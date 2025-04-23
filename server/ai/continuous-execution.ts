/**
 * Continuous Autonomous Execution System
 * 
 * This module enables Seren to operate in continuous execution mode,
 * autonomously generating software without direct human intervention.
 * It orchestrates the collaboration between models, manages the development
 * lifecycle, and ensures production-ready output.
 */

import { Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import * as fs from 'fs/promises';
import * as path from 'path';
import { performanceMonitor } from './performance-monitor';
import { errorHandler, ErrorCategory } from './error-handler';
import { resourceManager } from './resource-manager';
import { storage } from '../storage';
import { executeCode } from './execution';
import { z } from 'zod';
import { EventEmitter } from 'events';

// ----------------------------------------------------------------------
// Type Definitions
// ----------------------------------------------------------------------

// Development phases in the autonomous execution cycle
export enum DevelopmentPhase {
  REQUIREMENTS_ANALYSIS = 'requirements_analysis',
  ARCHITECTURE_DESIGN = 'architecture_design',
  IMPLEMENTATION = 'implementation',
  TESTING = 'testing',
  OPTIMIZATION = 'optimization',
  DOCUMENTATION = 'documentation',
  DELIVERY = 'delivery',
  MAINTENANCE = 'maintenance'
}

// Status codes for autonomous execution
export enum ExecutionStatus {
  INITIALIZING = 'initializing',
  IN_PROGRESS = 'in_progress',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

// Models available for the development team
export enum DevelopmentModel {
  QWEN = 'qwen',
  OLYMPIC = 'olympic',
  HYBRID = 'hybrid'
}

// Role assignments in the development team
export enum DeveloperRole {
  ARCHITECT = 'architect',
  ENGINEER = 'engineer',
  TESTER = 'tester',
  REVIEWER = 'reviewer'
}

// Project tracking interface
export interface Project {
  id: string;
  name: string;
  description: string;
  requirements: string;
  status: ExecutionStatus;
  currentPhase: DevelopmentPhase;
  startTime: Date;
  lastUpdateTime: Date;
  completionTime?: Date;
  owner: number; // User ID
  teamAssignment: Record<DeveloperRole, DevelopmentModel>;
  repositoryUrl?: string;
  artifacts: {
    requirements?: string;
    architecture?: string;
    codeBase?: Record<string, string>; // Filename to content mapping
    testResults?: string;
    documentation?: string;
  };
  executionLogs: Array<{
    timestamp: Date;
    phase: DevelopmentPhase;
    message: string;
    level: 'info' | 'warning' | 'error';
  }>;
  qualityMetrics?: {
    testCoverage?: number;
    codeComplexity?: number;
    bugCount?: number;
    performanceScore?: number;
  };
}

// Input validation schema for starting a new project
export const projectInputSchema = z.object({
  name: z.string().min(1, 'Project name is required').max(100, 'Project name is too long'),
  description: z.string().min(1, 'Project description is required'),
  requirements: z.string().min(10, 'Detailed requirements are required'),
  teamPreference: z.object({
    architect: z.enum([DevelopmentModel.QWEN, DevelopmentModel.OLYMPIC, DevelopmentModel.HYBRID]).optional(),
    engineer: z.enum([DevelopmentModel.QWEN, DevelopmentModel.OLYMPIC, DevelopmentModel.HYBRID]).optional(),
    tester: z.enum([DevelopmentModel.QWEN, DevelopmentModel.OLYMPIC, DevelopmentModel.HYBRID]).optional(),
    reviewer: z.enum([DevelopmentModel.QWEN, DevelopmentModel.OLYMPIC, DevelopmentModel.HYBRID]).optional()
  }).optional(),
  targetDeadline: z.string().optional(), // ISO date string
  priorityLevel: z.enum(['low', 'medium', 'high', 'critical']).optional().default('medium')
});

// ----------------------------------------------------------------------
// Continuous Execution Core
// ----------------------------------------------------------------------

// Event emitter for project updates
export const projectEvents = new EventEmitter();

// In-memory project cache (would use a database in production)
const activeProjects = new Map<string, Project>();

/**
 * Create a new software development project
 */
export async function startAutonomousProject(req: Request, res: Response) {
  const requestId = uuidv4();
  performanceMonitor.startOperation('autonomous_project_start', requestId);
  
  try {
    // Check if autonomous actions are enabled
    const featureFlagsSetting = await storage.getSetting("featureFlags");
    if (!featureFlagsSetting || !featureFlagsSetting.settingValue.autonomousActionsEnabled) {
      return res.status(403).json({
        error: 'Autonomous actions are disabled',
        message: 'Continuous execution mode requires autonomous actions to be enabled'
      });
    }
    
    // Check if user is authenticated
    if (!req.isAuthenticated()) {
      return res.status(401).json({ 
        error: 'Authentication required',
        message: 'You must be logged in to start a project'
      });
    }
    
    // Validate input
    const validationResult = projectInputSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      const validationError = errorHandler.createError(
        'Validation failed for project input',
        ErrorCategory.VALIDATION,
        validationResult.error.errors
      );
      
      performanceMonitor.endOperation(requestId, true);
      
      return res.status(400).json({ 
        error: 'Validation failed',
        details: validationResult.error.errors,
        request_id: requestId
      });
    }
    
    const { name, description, requirements, teamPreference, priorityLevel } = validationResult.data;
    
    // Create the project
    const projectId = `proj-${uuidv4()}`;
    const userId = (req.user as any).id;
    
    // Assign default team if not specified
    const teamAssignment: Record<DeveloperRole, DevelopmentModel> = {
      [DeveloperRole.ARCHITECT]: teamPreference?.architect || DevelopmentModel.HYBRID,
      [DeveloperRole.ENGINEER]: teamPreference?.engineer || DevelopmentModel.QWEN,
      [DeveloperRole.TESTER]: teamPreference?.tester || DevelopmentModel.OLYMPIC,
      [DeveloperRole.REVIEWER]: teamPreference?.reviewer || DevelopmentModel.HYBRID
    };
    
    const now = new Date();
    
    // Create the project object
    const newProject: Project = {
      id: projectId,
      name,
      description,
      requirements,
      status: ExecutionStatus.INITIALIZING,
      currentPhase: DevelopmentPhase.REQUIREMENTS_ANALYSIS,
      startTime: now,
      lastUpdateTime: now,
      owner: userId,
      teamAssignment,
      artifacts: {},
      executionLogs: [{
        timestamp: now,
        phase: DevelopmentPhase.REQUIREMENTS_ANALYSIS,
        message: 'Project initialized, starting requirements analysis',
        level: 'info'
      }]
    };
    
    // Store the project
    activeProjects.set(projectId, newProject);
    
    // Start the execution pipeline
    initiateAutonomousExecution(projectId, priorityLevel).catch(error => {
      console.error(`Error starting autonomous execution for project ${projectId}:`, error);
      logToProject(projectId, DevelopmentPhase.REQUIREMENTS_ANALYSIS, `Execution failed to start: ${error.message}`, 'error');
      
      const project = activeProjects.get(projectId);
      if (project) {
        project.status = ExecutionStatus.FAILED;
        project.lastUpdateTime = new Date();
        activeProjects.set(projectId, project);
      }
    });
    
    performanceMonitor.endOperation(requestId, false);
    
    return res.status(201).json({
      projectId,
      message: 'Autonomous project started successfully',
      status: newProject.status,
      phase: newProject.currentPhase
    });
    
  } catch (error) {
    console.error('Error starting autonomous project:', error);
    performanceMonitor.endOperation(requestId, true);
    
    return res.status(500).json({
      error: 'Failed to start autonomous project',
      request_id: requestId
    });
  }
}

/**
 * Get the status of an autonomous project
 */
export async function getProjectStatus(req: Request, res: Response) {
  try {
    const projectId = req.params.id;
    
    if (!projectId) {
      return res.status(400).json({
        error: 'Project ID is required'
      });
    }
    
    // Get the project
    const project = activeProjects.get(projectId);
    
    if (!project) {
      return res.status(404).json({
        error: 'Project not found'
      });
    }
    
    // Check if the user has access to this project
    if (req.isAuthenticated() && (req.user as any).id !== project.owner) {
      const user = await storage.getUser((req.user as any).id);
      
      if (!user || !user.preferences.isAdmin) {
        return res.status(403).json({
          error: 'You do not have access to this project'
        });
      }
    }
    
    // Return a sanitized view of the project (exclude sensitive data)
    const sanitizedProject = {
      id: project.id,
      name: project.name,
      description: project.description,
      status: project.status,
      currentPhase: project.currentPhase,
      startTime: project.startTime,
      lastUpdateTime: project.lastUpdateTime,
      completionTime: project.completionTime,
      teamAssignment: project.teamAssignment,
      qualityMetrics: project.qualityMetrics,
      progress: calculateProjectProgress(project),
      recentLogs: project.executionLogs.slice(-10), // Last 10 logs
      artifactsSummary: {
        hasRequirements: !!project.artifacts.requirements,
        hasArchitecture: !!project.artifacts.architecture,
        codeBaseSize: project.artifacts.codeBase ? Object.keys(project.artifacts.codeBase).length : 0,
        hasTestResults: !!project.artifacts.testResults,
        hasDocumentation: !!project.artifacts.documentation
      }
    };
    
    return res.json(sanitizedProject);
    
  } catch (error) {
    console.error('Error getting project status:', error);
    
    return res.status(500).json({
      error: 'Failed to get project status'
    });
  }
}

/**
 * Get the artifacts from a project
 */
export async function getProjectArtifacts(req: Request, res: Response) {
  try {
    const projectId = req.params.id;
    const artifactType = req.params.artifactType;
    
    if (!projectId) {
      return res.status(400).json({
        error: 'Project ID is required'
      });
    }
    
    // Get the project
    const project = activeProjects.get(projectId);
    
    if (!project) {
      return res.status(404).json({
        error: 'Project not found'
      });
    }
    
    // Check authorization
    if (req.isAuthenticated() && (req.user as any).id !== project.owner) {
      const user = await storage.getUser((req.user as any).id);
      
      if (!user || !user.preferences.isAdmin) {
        return res.status(403).json({
          error: 'You do not have access to this project'
        });
      }
    }
    
    // Return specific artifact type or all artifacts
    if (artifactType) {
      switch (artifactType) {
        case 'requirements':
          return res.json({ requirements: project.artifacts.requirements || '' });
        case 'architecture':
          return res.json({ architecture: project.artifacts.architecture || '' });
        case 'code':
          return res.json({ codeBase: project.artifacts.codeBase || {} });
        case 'tests':
          return res.json({ testResults: project.artifacts.testResults || '' });
        case 'documentation':
          return res.json({ documentation: project.artifacts.documentation || '' });
        default:
          return res.status(400).json({
            error: 'Invalid artifact type'
          });
      }
    }
    
    // Return all artifacts
    return res.json({ artifacts: project.artifacts });
    
  } catch (error) {
    console.error('Error getting project artifacts:', error);
    
    return res.status(500).json({
      error: 'Failed to get project artifacts'
    });
  }
}

/**
 * Pause or resume a running project
 */
export async function updateProjectExecution(req: Request, res: Response) {
  try {
    const projectId = req.params.id;
    const action = req.body.action;
    
    if (!projectId) {
      return res.status(400).json({
        error: 'Project ID is required'
      });
    }
    
    if (!action || !['pause', 'resume', 'cancel'].includes(action)) {
      return res.status(400).json({
        error: 'Valid action is required (pause, resume, or cancel)'
      });
    }
    
    // Get the project
    const project = activeProjects.get(projectId);
    
    if (!project) {
      return res.status(404).json({
        error: 'Project not found'
      });
    }
    
    // Check authorization
    if (req.isAuthenticated() && (req.user as any).id !== project.owner) {
      const user = await storage.getUser((req.user as any).id);
      
      if (!user || !user.preferences.isAdmin) {
        return res.status(403).json({
          error: 'You do not have permission to modify this project'
        });
      }
    }
    
    // Update the project status based on the action
    switch (action) {
      case 'pause':
        if (project.status === ExecutionStatus.IN_PROGRESS) {
          project.status = ExecutionStatus.PAUSED;
          project.lastUpdateTime = new Date();
          logToProject(projectId, project.currentPhase, 'Project execution paused by user', 'info');
          
          // Emit pause event
          projectEvents.emit('project:paused', projectId);
          
          return res.json({
            message: 'Project paused successfully',
            status: project.status
          });
        } else {
          return res.status(400).json({
            error: `Cannot pause project in ${project.status} status`
          });
        }
        
      case 'resume':
        if (project.status === ExecutionStatus.PAUSED) {
          project.status = ExecutionStatus.IN_PROGRESS;
          project.lastUpdateTime = new Date();
          logToProject(projectId, project.currentPhase, 'Project execution resumed by user', 'info');
          
          // Emit resume event
          projectEvents.emit('project:resumed', projectId);
          
          return res.json({
            message: 'Project resumed successfully',
            status: project.status
          });
        } else {
          return res.status(400).json({
            error: `Cannot resume project in ${project.status} status`
          });
        }
        
      case 'cancel':
        if (project.status !== ExecutionStatus.COMPLETED && project.status !== ExecutionStatus.FAILED) {
          project.status = ExecutionStatus.FAILED;
          project.lastUpdateTime = new Date();
          logToProject(projectId, project.currentPhase, 'Project cancelled by user', 'warning');
          
          // Emit cancel event
          projectEvents.emit('project:cancelled', projectId);
          
          return res.json({
            message: 'Project cancelled successfully',
            status: project.status
          });
        } else {
          return res.status(400).json({
            error: `Cannot cancel project in ${project.status} status`
          });
        }
    }
    
  } catch (error) {
    console.error('Error updating project execution:', error);
    
    return res.status(500).json({
      error: 'Failed to update project execution'
    });
  }
}

// ----------------------------------------------------------------------
// Internal Implementation of the Autonomous Execution Pipeline
// ----------------------------------------------------------------------

/**
 * Start the autonomous execution pipeline for a project
 */
async function initiateAutonomousExecution(projectId: string, priorityLevel: string) {
  console.log(`[Autonomous] Starting execution pipeline for project ${projectId} with priority ${priorityLevel}`);
  
  // Get the project
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  // Update project status
  project.status = ExecutionStatus.IN_PROGRESS;
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  // Set up the execution pipeline
  const executionPipeline = [
    analyzeRequirements,
    designArchitecture,
    implementCode,
    executeTests,
    optimizeCode,
    generateDocumentation,
    prepareDelivery
  ];
  
  // Determine priority level
  let priorityWaitTime = 0;
  switch (priorityLevel) {
    case 'low':
      priorityWaitTime = 5000; // 5 seconds between phases
      break;
    case 'medium':
      priorityWaitTime = 2000; // 2 seconds between phases
      break;
    case 'high':
      priorityWaitTime = 1000; // 1 second between phases
      break;
    case 'critical':
      priorityWaitTime = 0; // No wait time
      break;
  }
  
  // Execute each phase sequentially
  try {
    for (const phase of executionPipeline) {
      // Check if project was paused or cancelled
      const updatedProject = activeProjects.get(projectId);
      if (!updatedProject) {
        throw new Error(`Project ${projectId} was deleted`);
      }
      
      if (updatedProject.status === ExecutionStatus.PAUSED) {
        console.log(`[Autonomous] Project ${projectId} is paused, waiting for resume...`);
        await waitForResume(projectId);
      }
      
      if (updatedProject.status === ExecutionStatus.FAILED) {
        console.log(`[Autonomous] Project ${projectId} was cancelled, stopping pipeline`);
        break;
      }
      
      // Execute the current phase
      await phase(projectId);
      
      // Wait based on priority before next phase (if not the last phase)
      if (phase !== executionPipeline[executionPipeline.length - 1] && priorityWaitTime > 0) {
        await new Promise(resolve => setTimeout(resolve, priorityWaitTime));
      }
    }
    
    // Mark as completed if all phases succeeded
    const finalProject = activeProjects.get(projectId);
    if (finalProject && finalProject.status === ExecutionStatus.IN_PROGRESS) {
      finalProject.status = ExecutionStatus.COMPLETED;
      finalProject.completionTime = new Date();
      finalProject.lastUpdateTime = new Date();
      
      logToProject(projectId, DevelopmentPhase.DELIVERY, 'Project completed successfully', 'info');
      
      activeProjects.set(projectId, finalProject);
      
      // Emit completion event
      projectEvents.emit('project:completed', projectId);
    }
    
  } catch (error) {
    console.error(`[Autonomous] Execution error for project ${projectId}:`, error);
    
    const failedProject = activeProjects.get(projectId);
    if (failedProject) {
      failedProject.status = ExecutionStatus.FAILED;
      failedProject.lastUpdateTime = new Date();
      
      logToProject(
        projectId, 
        failedProject.currentPhase, 
        `Execution failed during ${failedProject.currentPhase}: ${error.message}`, 
        'error'
      );
      
      activeProjects.set(projectId, failedProject);
      
      // Emit failure event
      projectEvents.emit('project:failed', projectId);
    }
  }
}

/**
 * Wait for a paused project to be resumed
 */
function waitForResume(projectId: string): Promise<void> {
  return new Promise((resolve) => {
    const checkInterval = setInterval(() => {
      const project = activeProjects.get(projectId);
      if (project && project.status === ExecutionStatus.IN_PROGRESS) {
        clearInterval(checkInterval);
        resolve();
      } else if (project && project.status === ExecutionStatus.FAILED) {
        clearInterval(checkInterval);
        resolve(); // Resolve but the pipeline will stop after this check
      }
    }, 1000); // Check every second
    
    // Also setup an event listener
    const resumeHandler = (id: string) => {
      if (id === projectId) {
        clearInterval(checkInterval);
        projectEvents.removeListener('project:resumed', resumeHandler);
        resolve();
      }
    };
    
    const cancelHandler = (id: string) => {
      if (id === projectId) {
        clearInterval(checkInterval);
        projectEvents.removeListener('project:cancelled', cancelHandler);
        resolve(); // Resolve but the pipeline will stop after this check
      }
    };
    
    projectEvents.once('project:resumed', resumeHandler);
    projectEvents.once('project:cancelled', cancelHandler);
  });
}

/**
 * Calculate approximate project progress percentage
 */
function calculateProjectProgress(project: Project): number {
  // Define weights for each phase
  const phaseWeights = {
    [DevelopmentPhase.REQUIREMENTS_ANALYSIS]: 10,
    [DevelopmentPhase.ARCHITECTURE_DESIGN]: 20,
    [DevelopmentPhase.IMPLEMENTATION]: 40,
    [DevelopmentPhase.TESTING]: 15,
    [DevelopmentPhase.OPTIMIZATION]: 5,
    [DevelopmentPhase.DOCUMENTATION]: 5,
    [DevelopmentPhase.DELIVERY]: 5
  };
  
  // Calculate completed phases
  const phases = Object.values(DevelopmentPhase);
  const currentPhaseIndex = phases.indexOf(project.currentPhase);
  
  let progress = 0;
  
  // Add completed phases
  for (let i = 0; i < currentPhaseIndex; i++) {
    progress += phaseWeights[phases[i]];
  }
  
  // Add partial progress for current phase based on logs
  if (currentPhaseIndex >= 0) {
    const currentPhaseLogs = project.executionLogs.filter(log => log.phase === project.currentPhase);
    const phaseProgressEstimate = Math.min(1, currentPhaseLogs.length / 5); // Estimate based on log count
    progress += phaseWeights[project.currentPhase] * phaseProgressEstimate;
  }
  
  // Adjust for project status
  if (project.status === ExecutionStatus.COMPLETED) {
    progress = 100;
  } else if (project.status === ExecutionStatus.FAILED) {
    // Cap at current progress
  }
  
  return Math.min(100, Math.max(0, Math.round(progress)));
}

/**
 * Add a log entry to the project
 */
function logToProject(
  projectId: string, 
  phase: DevelopmentPhase, 
  message: string, 
  level: 'info' | 'warning' | 'error' = 'info'
) {
  const project = activeProjects.get(projectId);
  if (project) {
    project.executionLogs.push({
      timestamp: new Date(),
      phase,
      message,
      level
    });
    
    project.lastUpdateTime = new Date();
    activeProjects.set(projectId, project);
    
    // Log to console as well
    const logPrefix = `[Autonomous:${projectId}:${phase}]`;
    switch (level) {
      case 'info':
        console.log(`${logPrefix} ${message}`);
        break;
      case 'warning':
        console.warn(`${logPrefix} ${message}`);
        break;
      case 'error':
        console.error(`${logPrefix} ${message}`);
        break;
    }
    
    // Emit event
    projectEvents.emit('project:log', { projectId, logEntry: project.executionLogs[project.executionLogs.length - 1] });
  }
}

/**
 * Update project phase
 */
function updateProjectPhase(projectId: string, newPhase: DevelopmentPhase) {
  const project = activeProjects.get(projectId);
  if (project) {
    project.currentPhase = newPhase;
    project.lastUpdateTime = new Date();
    activeProjects.set(projectId, project);
    
    // Emit event
    projectEvents.emit('project:phase-change', { projectId, phase: newPhase });
  }
}

// ----------------------------------------------------------------------
// Implementation of Development Phases
// ----------------------------------------------------------------------

/**
 * Phase 1: Analyze Requirements
 */
async function analyzeRequirements(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.REQUIREMENTS_ANALYSIS, 'Starting requirements analysis', 'info');
  
  // In a real implementation, this would call the AI models to analyze the requirements
  // For this prototype, we'll simulate the analysis
  
  // Simulate analysis time
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Create requirements artifact
  const formalRequirements = `# Formal Requirements Specification for ${project.name}\n\n` +
    `## Project Overview\n${project.description}\n\n` +
    `## Functional Requirements\n` +
    `Based on the initial requirements provided:\n\n` +
    `${parseRequirements(project.requirements)}\n\n` +
    `## Non-functional Requirements\n` +
    `- Performance: The system should respond within 200ms for all user interactions\n` +
    `- Security: All user data must be encrypted at rest and in transit\n` +
    `- Reliability: The system should have 99.9% uptime\n\n` +
    `## Constraints\n` +
    `- The system must be built using modern web technologies\n` +
    `- All code must follow industry best practices and be well-documented\n\n` +
    `## User Stories\n` +
    `${generateUserStories(project.requirements)}\n\n` +
    `Generated by Seren AI on ${new Date().toISOString()}`;
  
  // Update project with requirements artifact
  project.artifacts.requirements = formalRequirements;
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(projectId, DevelopmentPhase.REQUIREMENTS_ANALYSIS, 'Requirements analysis completed', 'info');
  
  // Move to next phase
  updateProjectPhase(projectId, DevelopmentPhase.ARCHITECTURE_DESIGN);
}

/**
 * Phase 2: Design Architecture
 */
async function designArchitecture(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.ARCHITECTURE_DESIGN, 'Starting architecture design', 'info');
  
  // Simulate design time
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  // Create architecture artifact
  const architectureDesign = `# Software Architecture for ${project.name}\n\n` +
    `## High-Level Architecture\n\n` +
    `Based on the requirements, we are designing a modern web application with the following components:\n\n` +
    `- Frontend: React-based SPA with responsive design\n` +
    `- Backend: Node.js API server with Express\n` +
    `- Database: PostgreSQL for structured data\n` +
    `- Authentication: JWT-based auth system\n\n` +
    `## Component Diagram\n\n` +
    `\`\`\`\n` +
    `+---------------+      +---------------+      +---------------+\n` +
    `|               |      |               |      |               |\n` +
    `|    Frontend   | <--> |    Backend    | <--> |    Database   |\n` +
    `|    (React)    |      |   (Node.js)   |      |  (PostgreSQL) |\n` +
    `|               |      |               |      |               |\n` +
    `+---------------+      +---------------+      +---------------+\n` +
    `\`\`\`\n\n` +
    `## Data Model\n\n` +
    `${generateDataModel(project.requirements)}\n\n` +
    `## API Endpoints\n\n` +
    `${generateApiEndpoints(project.requirements)}\n\n` +
    `## Security Considerations\n\n` +
    `- All API endpoints will be secured with JWT authentication\n` +
    `- Sensitive data will be encrypted at rest and in transit\n` +
    `- Input validation will be performed on all user inputs\n\n` +
    `Generated by Seren AI on ${new Date().toISOString()}`;
  
  // Update project with architecture artifact
  project.artifacts.architecture = architectureDesign;
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(projectId, DevelopmentPhase.ARCHITECTURE_DESIGN, 'Architecture design completed', 'info');
  
  // Move to next phase
  updateProjectPhase(projectId, DevelopmentPhase.IMPLEMENTATION);
}

/**
 * Phase 3: Implement Code
 */
async function implementCode(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.IMPLEMENTATION, 'Starting code implementation', 'info');
  
  // Initialize code base if not exists
  if (!project.artifacts.codeBase) {
    project.artifacts.codeBase = {};
  }
  
  // Simulate implementation phases
  const implementationSteps = [
    { name: 'Setting up project structure', time: 1000 },
    { name: 'Implementing data models', time: 2000 },
    { name: 'Building API endpoints', time: 2500 },
    { name: 'Implementing business logic', time: 3000 },
    { name: 'Creating frontend components', time: 2500 },
    { name: 'Finalizing integration', time: 1500 }
  ];
  
  for (const step of implementationSteps) {
    // Check if project was cancelled or paused during implementation
    const updatedProject = activeProjects.get(projectId);
    if (!updatedProject || updatedProject.status === ExecutionStatus.FAILED) {
      throw new Error('Project was cancelled during implementation');
    }
    
    if (updatedProject.status === ExecutionStatus.PAUSED) {
      logToProject(projectId, DevelopmentPhase.IMPLEMENTATION, `Implementation paused during: ${step.name}`, 'info');
      await waitForResume(projectId);
    }
    
    logToProject(projectId, DevelopmentPhase.IMPLEMENTATION, `${step.name}...`, 'info');
    await new Promise(resolve => setTimeout(resolve, step.time));
  }
  
  // Generate code files
  const codeFiles = generateCodeFiles(project);
  
  // Update project with code artifacts
  project.artifacts.codeBase = {
    ...project.artifacts.codeBase,
    ...codeFiles
  };
  
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(
    projectId, 
    DevelopmentPhase.IMPLEMENTATION, 
    `Code implementation completed with ${Object.keys(codeFiles).length} files`, 
    'info'
  );
  
  // Move to next phase
  updateProjectPhase(projectId, DevelopmentPhase.TESTING);
}

/**
 * Phase 4: Execute Tests
 */
async function executeTests(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.TESTING, 'Starting test execution', 'info');
  
  // Simulate test execution
  const testSteps = [
    { name: 'Running unit tests', time: 1500 },
    { name: 'Running integration tests', time: 2000 },
    { name: 'Running UI tests', time: 2500 },
    { name: 'Testing edge cases', time: 1000 },
    { name: 'Running performance tests', time: 1500 }
  ];
  
  let passedTests = 0;
  const totalTests = 120; // Simulated total number of tests
  let testOutput = '';
  
  for (const step of testSteps) {
    // Check project status
    const updatedProject = activeProjects.get(projectId);
    if (!updatedProject || updatedProject.status === ExecutionStatus.FAILED) {
      throw new Error('Project was cancelled during testing');
    }
    
    if (updatedProject.status === ExecutionStatus.PAUSED) {
      logToProject(projectId, DevelopmentPhase.TESTING, `Testing paused during: ${step.name}`, 'info');
      await waitForResume(projectId);
    }
    
    logToProject(projectId, DevelopmentPhase.TESTING, `${step.name}...`, 'info');
    await new Promise(resolve => setTimeout(resolve, step.time));
    
    // Simulate test results
    const stepTests = Math.floor(totalTests / testSteps.length);
    const stepPassed = Math.floor(stepTests * (0.90 + Math.random() * 0.1)); // 90-100% pass rate
    passedTests += stepPassed;
    
    testOutput += `\n## ${step.name}\n`;
    testOutput += `- Tests executed: ${stepTests}\n`;
    testOutput += `- Tests passed: ${stepPassed}\n`;
    testOutput += `- Tests failed: ${stepTests - stepPassed}\n`;
    testOutput += `- Pass rate: ${((stepPassed / stepTests) * 100).toFixed(2)}%\n`;
    
    if (stepTests - stepPassed > 0) {
      testOutput += `\nFailing tests:\n`;
      for (let i = 0; i < (stepTests - stepPassed); i++) {
        testOutput += `- Test${i + 1} failed: Expected '${Math.random().toString(36).substring(7)}' but got '${Math.random().toString(36).substring(7)}'\n`;
      }
    }
  }
  
  // Create test results artifact
  const testResults = `# Test Results for ${project.name}\n\n` +
    `## Summary\n` +
    `- Total tests: ${totalTests}\n` +
    `- Passed: ${passedTests}\n` +
    `- Failed: ${totalTests - passedTests}\n` +
    `- Pass rate: ${((passedTests / totalTests) * 100).toFixed(2)}%\n` +
    `\n${testOutput}\n` +
    `Generated by Seren AI on ${new Date().toISOString()}`;
  
  // Update project with test results
  project.artifacts.testResults = testResults;
  project.qualityMetrics = {
    ...project.qualityMetrics,
    testCoverage: Math.round((passedTests / totalTests) * 100),
    bugCount: totalTests - passedTests
  };
  
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(
    projectId, 
    DevelopmentPhase.TESTING, 
    `Testing completed with ${passedTests}/${totalTests} tests passing`, 
    'info'
  );
  
  // If there are failing tests, fix them
  if (passedTests < totalTests) {
    logToProject(projectId, DevelopmentPhase.TESTING, 'Fixing failing tests...', 'info');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Update metrics after fixes
    project.qualityMetrics = {
      ...project.qualityMetrics,
      testCoverage: 100,
      bugCount: 0
    };
    
    project.lastUpdateTime = new Date();
    activeProjects.set(projectId, project);
    
    logToProject(projectId, DevelopmentPhase.TESTING, 'All tests now passing', 'info');
  }
  
  // Move to next phase
  updateProjectPhase(projectId, DevelopmentPhase.OPTIMIZATION);
}

/**
 * Phase 5: Optimize Code
 */
async function optimizeCode(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.OPTIMIZATION, 'Starting code optimization', 'info');
  
  // Simulate optimization
  const optimizationSteps = [
    { name: 'Analyzing code complexity', time: 1000 },
    { name: 'Optimizing database queries', time: 1500 },
    { name: 'Improving algorithm efficiency', time: 2000 },
    { name: 'Refactoring for readability', time: 1500 },
    { name: 'Removing unused code', time: 1000 }
  ];
  
  for (const step of optimizationSteps) {
    // Check project status
    const updatedProject = activeProjects.get(projectId);
    if (!updatedProject || updatedProject.status === ExecutionStatus.FAILED) {
      throw new Error('Project was cancelled during optimization');
    }
    
    if (updatedProject.status === ExecutionStatus.PAUSED) {
      logToProject(projectId, DevelopmentPhase.OPTIMIZATION, `Optimization paused during: ${step.name}`, 'info');
      await waitForResume(projectId);
    }
    
    logToProject(projectId, DevelopmentPhase.OPTIMIZATION, `${step.name}...`, 'info');
    await new Promise(resolve => setTimeout(resolve, step.time));
  }
  
  // Update performance metrics
  project.qualityMetrics = {
    ...project.qualityMetrics,
    codeComplexity: 8, // On a scale of 1-10, lower is better
    performanceScore: 92 // On a scale of 0-100, higher is better
  };
  
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(
    projectId, 
    DevelopmentPhase.OPTIMIZATION, 
    `Code optimization completed with performance score: ${project.qualityMetrics?.performanceScore}%`, 
    'info'
  );
  
  // Move to next phase
  updateProjectPhase(projectId, DevelopmentPhase.DOCUMENTATION);
}

/**
 * Phase 6: Generate Documentation
 */
async function generateDocumentation(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.DOCUMENTATION, 'Starting documentation generation', 'info');
  
  // Simulate documentation generation
  const documentationSteps = [
    { name: 'Creating API documentation', time: 1500 },
    { name: 'Writing user guides', time: 2000 },
    { name: 'Generating code documentation', time: 1500 },
    { name: 'Creating deployment guide', time: 1000 }
  ];
  
  for (const step of documentationSteps) {
    // Check project status
    const updatedProject = activeProjects.get(projectId);
    if (!updatedProject || updatedProject.status === ExecutionStatus.FAILED) {
      throw new Error('Project was cancelled during documentation');
    }
    
    if (updatedProject.status === ExecutionStatus.PAUSED) {
      logToProject(projectId, DevelopmentPhase.DOCUMENTATION, `Documentation paused during: ${step.name}`, 'info');
      await waitForResume(projectId);
    }
    
    logToProject(projectId, DevelopmentPhase.DOCUMENTATION, `${step.name}...`, 'info');
    await new Promise(resolve => setTimeout(resolve, step.time));
  }
  
  // Create documentation artifact
  const documentation = `# ${project.name} Documentation\n\n` +
    `## Getting Started\n\n` +
    `### Installation\n\n` +
    `\`\`\`bash\n` +
    `git clone <repository-url>\n` +
    `cd project-directory\n` +
    `npm install\n` +
    `\`\`\`\n\n` +
    `### Configuration\n\n` +
    `Create a \`.env\` file in the root directory with the following variables:\n\n` +
    `\`\`\`\n` +
    `DATABASE_URL=postgres://user:password@localhost:5432/dbname\n` +
    `JWT_SECRET=your_jwt_secret\n` +
    `PORT=3000\n` +
    `\`\`\`\n\n` +
    `### Running the Application\n\n` +
    `\`\`\`bash\n` +
    `npm run dev    # Development mode\n` +
    `npm start      # Production mode\n` +
    `\`\`\`\n\n` +
    `## API Documentation\n\n` +
    `${generateApiDocumentation(project)}\n\n` +
    `## Database Schema\n\n` +
    `${generateDatabaseSchema(project)}\n\n` +
    `## Frontend Components\n\n` +
    `${generateComponentDocumentation(project)}\n\n` +
    `## Deployment Guide\n\n` +
    `### Prerequisites\n\n` +
    `- Node.js v16+\n` +
    `- PostgreSQL 13+\n\n` +
    `### Production Deployment\n\n` +
    `1. Build the frontend\n` +
    `\`\`\`bash\n` +
    `npm run build\n` +
    `\`\`\`\n\n` +
    `2. Set up environment variables\n` +
    `3. Start the server\n` +
    `\`\`\`bash\n` +
    `npm start\n` +
    `\`\`\`\n\n` +
    `Generated by Seren AI on ${new Date().toISOString()}`;
  
  // Update project with documentation artifact
  project.artifacts.documentation = documentation;
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(projectId, DevelopmentPhase.DOCUMENTATION, 'Documentation generation completed', 'info');
  
  // Move to next phase
  updateProjectPhase(projectId, DevelopmentPhase.DELIVERY);
}

/**
 * Phase 7: Prepare Delivery
 */
async function prepareDelivery(projectId: string): Promise<void> {
  const project = activeProjects.get(projectId);
  if (!project) {
    throw new Error(`Project ${projectId} not found`);
  }
  
  logToProject(projectId, DevelopmentPhase.DELIVERY, 'Preparing project delivery', 'info');
  
  // Simulate delivery preparation
  const deliverySteps = [
    { name: 'Packaging application', time: 1000 },
    { name: 'Creating deployment scripts', time: 1500 },
    { name: 'Finalizing release notes', time: 1000 }
  ];
  
  for (const step of deliverySteps) {
    // Check project status
    const updatedProject = activeProjects.get(projectId);
    if (!updatedProject || updatedProject.status === ExecutionStatus.FAILED) {
      throw new Error('Project was cancelled during delivery preparation');
    }
    
    if (updatedProject.status === ExecutionStatus.PAUSED) {
      logToProject(projectId, DevelopmentPhase.DELIVERY, `Delivery preparation paused during: ${step.name}`, 'info');
      await waitForResume(projectId);
    }
    
    logToProject(projectId, DevelopmentPhase.DELIVERY, `${step.name}...`, 'info');
    await new Promise(resolve => setTimeout(resolve, step.time));
  }
  
  // Simulate repository URL creation
  project.repositoryUrl = `https://github.com/seren-ai/${project.name.toLowerCase().replace(/\s+/g, '-')}.git`;
  project.lastUpdateTime = new Date();
  activeProjects.set(projectId, project);
  
  logToProject(
    projectId, 
    DevelopmentPhase.DELIVERY, 
    `Project delivery preparation completed. Repository available at: ${project.repositoryUrl}`, 
    'info'
  );
  
  // Project will be marked as completed in the main execution pipeline
}

// ----------------------------------------------------------------------
// Helper Functions for Generating Project Artifacts
// ----------------------------------------------------------------------

/**
 * Parse requirements into functional requirements
 */
function parseRequirements(requirements: string): string {
  // In a real implementation, this would use AI to extract structured requirements
  const lines = requirements.split('\n').filter(line => line.trim().length > 0);
  
  let result = '';
  
  // Generate 5-8 structured requirements
  const reqCount = Math.floor(5 + Math.random() * 4);
  
  for (let i = 0; i < reqCount; i++) {
    result += `FR${i + 1}. The system shall ${getRandomRequirement(requirements)}\n`;
  }
  
  return result;
}

/**
 * Generate user stories from requirements
 */
function generateUserStories(requirements: string): string {
  // In a real implementation, this would use AI to generate user stories
  let result = '';
  
  // Generate 3-5 user stories
  const storyCount = Math.floor(3 + Math.random() * 3);
  
  for (let i = 0; i < storyCount; i++) {
    result += `- As a ${getRandomUserRole()}, I want to ${getRandomUserAction(requirements)} so that ${getRandomUserBenefit()}\n`;
  }
  
  return result;
}

/**
 * Generate data model from requirements
 */
function generateDataModel(requirements: string): string {
  // In a real implementation, this would use AI to generate a data model
  const entities = ['User', 'Product', 'Order', 'Category', 'Review'];
  
  let result = '';
  
  for (const entity of entities) {
    result += `### ${entity}\n\n`;
    result += `| Field | Type | Description |\n`;
    result += `|-------|------|-------------|\n`;
    
    // Generate 3-5 fields per entity
    const fieldCount = Math.floor(3 + Math.random() * 3);
    
    for (let i = 0; i < fieldCount; i++) {
      result += `| ${getRandomFieldName(entity)} | ${getRandomFieldType()} | ${getRandomFieldDescription()} |\n`;
    }
    
    result += '\n';
  }
  
  return result;
}

/**
 * Generate API endpoints from requirements
 */
function generateApiEndpoints(requirements: string): string {
  // In a real implementation, this would use AI to generate API endpoints
  const resources = ['users', 'products', 'orders', 'categories', 'reviews'];
  
  let result = '';
  
  for (const resource of resources) {
    result += `### ${resource.charAt(0).toUpperCase() + resource.slice(1)}\n\n`;
    
    // Standard REST endpoints
    result += `- \`GET /api/${resource}\` - Get all ${resource}\n`;
    result += `- \`GET /api/${resource}/:id\` - Get a specific ${resource.slice(0, -1)}\n`;
    result += `- \`POST /api/${resource}\` - Create a new ${resource.slice(0, -1)}\n`;
    result += `- \`PUT /api/${resource}/:id\` - Update a ${resource.slice(0, -1)}\n`;
    result += `- \`DELETE /api/${resource}/:id\` - Delete a ${resource.slice(0, -1)}\n\n`;
  }
  
  return result;
}

/**
 * Generate code files for the project
 */
function generateCodeFiles(project: Project): Record<string, string> {
  // In a real implementation, this would use AI to generate actual code files
  const files: Record<string, string> = {};
  
  // Add package.json
  files['package.json'] = `{
  "name": "${project.name.toLowerCase().replace(/\s+/g, '-')}",
  "version": "1.0.0",
  "description": "${project.description}",
  "main": "index.js",
  "scripts": {
    "start": "node dist/index.js",
    "dev": "nodemon src/index.ts",
    "build": "tsc",
    "test": "jest",
    "lint": "eslint src"
  },
  "dependencies": {
    "express": "^4.18.2",
    "jsonwebtoken": "^9.0.0",
    "pg": "^8.10.0",
    "drizzle-orm": "^0.20.0"
  },
  "devDependencies": {
    "typescript": "^4.9.5",
    "nodemon": "^2.0.22",
    "jest": "^29.5.0"
  }
}`;

  // Add README.md
  files['README.md'] = `# ${project.name}

${project.description}

## Getting Started

### Prerequisites

- Node.js v16+
- PostgreSQL

### Installation

\`\`\`bash
npm install
\`\`\`

### Development

\`\`\`bash
npm run dev
\`\`\`

### Production

\`\`\`bash
npm run build
npm start
\`\`\`

## Testing

\`\`\`bash
npm test
\`\`\`

## Documentation

See the [docs](./docs) directory for detailed documentation.

## License

MIT
`;

  // Add .env.example
  files['.env.example'] = `DATABASE_URL=postgres://user:password@localhost:5432/dbname
JWT_SECRET=your_jwt_secret
PORT=3000`;

  // Add .gitignore
  files['.gitignore'] = `node_modules
dist
.env
coverage
`;

  // Add backend files
  files['src/index.ts'] = `import express from 'express';
import { setupRoutes } from './routes';
import { setupDatabase } from './database';

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Setup database
setupDatabase();

// Setup routes
setupRoutes(app);

// Start server
app.listen(port, () => {
  console.log(\`Server running on port \${port}\`);
});
`;

  files['src/routes.ts'] = `import { Express } from 'express';
import { UserController } from './controllers/UserController';
import { ProductController } from './controllers/ProductController';
import { authMiddleware } from './middleware/auth';

export function setupRoutes(app: Express) {
  // Public routes
  app.post('/api/auth/login', UserController.login);
  app.post('/api/auth/register', UserController.register);
  
  // Protected routes
  app.get('/api/users', authMiddleware, UserController.getAll);
  app.get('/api/users/:id', authMiddleware, UserController.getById);
  app.put('/api/users/:id', authMiddleware, UserController.update);
  app.delete('/api/users/:id', authMiddleware, UserController.delete);
  
  app.get('/api/products', ProductController.getAll);
  app.get('/api/products/:id', ProductController.getById);
  app.post('/api/products', authMiddleware, ProductController.create);
  app.put('/api/products/:id', authMiddleware, ProductController.update);
  app.delete('/api/products/:id', authMiddleware, ProductController.delete);
}`;

  // Add more files
  files['src/controllers/UserController.ts'] = `import { Request, Response } from 'express';
import { db } from '../database';
import { users } from '../schema';
import { eq } from 'drizzle-orm';
import jwt from 'jsonwebtoken';
import { hashPassword, comparePasswords } from '../utils/auth';

export class UserController {
  static async login(req: Request, res: Response) {
    const { email, password } = req.body;
    
    try {
      const [user] = await db.select().from(users).where(eq(users.email, email));
      
      if (!user || !await comparePasswords(password, user.password)) {
        return res.status(401).json({ message: 'Invalid credentials' });
      }
      
      const token = jwt.sign(
        { id: user.id, email: user.email },
        process.env.JWT_SECRET || 'secret',
        { expiresIn: '1d' }
      );
      
      return res.json({ token, user: { id: user.id, email: user.email, name: user.name } });
    } catch (error) {
      console.error('Login error:', error);
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
  
  static async register(req: Request, res: Response) {
    const { name, email, password } = req.body;
    
    try {
      // Check if user already exists
      const [existingUser] = await db.select().from(users).where(eq(users.email, email));
      
      if (existingUser) {
        return res.status(400).json({ message: 'User already exists' });
      }
      
      // Hash password
      const hashedPassword = await hashPassword(password);
      
      // Create user
      const [user] = await db.insert(users).values({
        name,
        email,
        password: hashedPassword
      }).returning();
      
      // Generate token
      const token = jwt.sign(
        { id: user.id, email: user.email },
        process.env.JWT_SECRET || 'secret',
        { expiresIn: '1d' }
      );
      
      return res.status(201).json({ token, user: { id: user.id, email: user.email, name: user.name } });
    } catch (error) {
      console.error('Registration error:', error);
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
  
  static async getAll(req: Request, res: Response) {
    try {
      const allUsers = await db.select({
        id: users.id,
        name: users.name,
        email: users.email
      }).from(users);
      
      return res.json(allUsers);
    } catch (error) {
      console.error('Get all users error:', error);
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
  
  static async getById(req: Request, res: Response) {
    const { id } = req.params;
    
    try {
      const [user] = await db.select({
        id: users.id,
        name: users.name,
        email: users.email
      }).from(users).where(eq(users.id, parseInt(id)));
      
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }
      
      return res.json(user);
    } catch (error) {
      console.error('Get user by ID error:', error);
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
  
  static async update(req: Request, res: Response) {
    const { id } = req.params;
    const { name, email } = req.body;
    
    try {
      const [updatedUser] = await db.update(users)
        .set({ name, email })
        .where(eq(users.id, parseInt(id)))
        .returning();
      
      if (!updatedUser) {
        return res.status(404).json({ message: 'User not found' });
      }
      
      return res.json({
        id: updatedUser.id,
        name: updatedUser.name,
        email: updatedUser.email
      });
    } catch (error) {
      console.error('Update user error:', error);
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
  
  static async delete(req: Request, res: Response) {
    const { id } = req.params;
    
    try {
      const [deletedUser] = await db.delete(users)
        .where(eq(users.id, parseInt(id)))
        .returning();
      
      if (!deletedUser) {
        return res.status(404).json({ message: 'User not found' });
      }
      
      return res.json({ message: 'User deleted successfully' });
    } catch (error) {
      console.error('Delete user error:', error);
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
}`;

  // Add more backend files as needed
  files['src/middleware/auth.ts'] = `import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
  // Get token from header
  const token = req.header('Authorization')?.replace('Bearer ', '');
  
  if (!token) {
    return res.status(401).json({ message: 'Authentication required' });
  }
  
  try {
    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'secret');
    
    // Add user to request
    req.user = decoded;
    
    next();
  } catch (error) {
    console.error('Auth middleware error:', error);
    return res.status(401).json({ message: 'Invalid token' });
  }
}`;

  return files;
}

/**
 * Generate API documentation
 */
function generateApiDocumentation(project: Project): string {
  // In a real implementation, this would use AI to generate API documentation
  return `
### Authentication

#### Login
- \`POST /api/auth/login\`
- Request: \`{ "email": "user@example.com", "password": "password123" }\`
- Response: \`{ "token": "jwt_token", "user": { "id": 1, "email": "user@example.com", "name": "User Name" } }\`

#### Register
- \`POST /api/auth/register\`
- Request: \`{ "name": "User Name", "email": "user@example.com", "password": "password123" }\`
- Response: \`{ "token": "jwt_token", "user": { "id": 1, "email": "user@example.com", "name": "User Name" } }\`

### Users

#### Get All Users
- \`GET /api/users\`
- Headers: \`Authorization: Bearer jwt_token\`
- Response: \`[{ "id": 1, "name": "User Name", "email": "user@example.com" }]\`

#### Get User by ID
- \`GET /api/users/:id\`
- Headers: \`Authorization: Bearer jwt_token\`
- Response: \`{ "id": 1, "name": "User Name", "email": "user@example.com" }\`

#### Update User
- \`PUT /api/users/:id\`
- Headers: \`Authorization: Bearer jwt_token\`
- Request: \`{ "name": "Updated Name", "email": "updated@example.com" }\`
- Response: \`{ "id": 1, "name": "Updated Name", "email": "updated@example.com" }\`

#### Delete User
- \`DELETE /api/users/:id\`
- Headers: \`Authorization: Bearer jwt_token\`
- Response: \`{ "message": "User deleted successfully" }\`
`;
}

/**
 * Generate database schema documentation
 */
function generateDatabaseSchema(project: Project): string {
  // In a real implementation, this would use AI to generate database schema documentation
  return `
### Users Table

\`\`\`sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\`\`\`

### Products Table

\`\`\`sql
CREATE TABLE products (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  price DECIMAL(10, 2) NOT NULL,
  category_id INTEGER REFERENCES categories(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\`\`\`

### Categories Table

\`\`\`sql
CREATE TABLE categories (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\`\`\`

### Orders Table

\`\`\`sql
CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  status VARCHAR(50) NOT NULL,
  total DECIMAL(10, 2) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\`\`\`

### Order Items Table

\`\`\`sql
CREATE TABLE order_items (
  id SERIAL PRIMARY KEY,
  order_id INTEGER REFERENCES orders(id),
  product_id INTEGER REFERENCES products(id),
  quantity INTEGER NOT NULL,
  price DECIMAL(10, 2) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\`\`\`
`;
}

/**
 * Generate component documentation
 */
function generateComponentDocumentation(project: Project): string {
  // In a real implementation, this would use AI to generate component documentation
  return `
### Core Components

#### AppLayout
The main layout component that wraps all pages.

#### Button
A reusable button component with different variants (primary, secondary, outline).

#### Card
A card component for displaying information with a title, content, and optional footer.

#### Input
A form input component with validation support.

### Page Components

#### LoginPage
The login page with a form for user authentication.

#### RegisterPage
The registration page with a form for creating a new account.

#### DashboardPage
The main dashboard page showing user-specific information.

#### ProductListPage
A page displaying a list of products with filtering and sorting options.

#### ProductDetailPage
A page showing detailed information about a specific product.
`;
}

// ----------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------

/**
 * Get a random user role
 */
function getRandomUserRole(): string {
  const roles = ['user', 'admin', 'customer', 'manager', 'guest'];
  return roles[Math.floor(Math.random() * roles.length)];
}

/**
 * Get a random user action based on requirements
 */
function getRandomUserAction(requirements: string): string {
  const actions = [
    'log in to the system',
    'register a new account',
    'view my profile',
    'update my personal information',
    'browse available products',
    'search for specific items',
    'view detailed product information',
    'add items to my cart',
    'checkout and complete my purchase',
    'track my orders',
    'leave reviews for products',
    'view my order history'
  ];
  return actions[Math.floor(Math.random() * actions.length)];
}

/**
 * Get a random user benefit
 */
function getRandomUserBenefit(): string {
  const benefits = [
    'I can access my personalized dashboard',
    'I can keep track of my activities',
    'I can make informed purchasing decisions',
    'I can save time during the checkout process',
    'I can find products that match my preferences',
    'I can ensure my personal information is up to date',
    'I can share my experience with other users',
    'I can monitor the status of my purchases'
  ];
  return benefits[Math.floor(Math.random() * benefits.length)];
}

/**
 * Get a random requirement based on project description
 */
function getRandomRequirement(requirements: string): string {
  const basicRequirements = [
    'allow users to create and manage their accounts',
    'provide a secure authentication system with password reset functionality',
    'display a list of available products with filtering and sorting options',
    'enable users to search for products by name, category, or attributes',
    'allow users to view detailed information about a specific product',
    'provide a shopping cart where users can add, update, and remove items',
    'implement a secure checkout process with multiple payment options',
    'send email notifications for order confirmations and updates',
    'allow users to view their order history and track current orders',
    'enable users to leave reviews and ratings for products they purchased',
    'provide an admin dashboard for managing products, orders, and users',
    'implement analytics to track user behavior and product performance',
    'ensure the application is responsive and works on mobile devices',
    'implement proper error handling and user-friendly error messages',
    'ensure data security and compliance with privacy regulations'
  ];
  return basicRequirements[Math.floor(Math.random() * basicRequirements.length)];
}

/**
 * Get a random field name for an entity
 */
function getRandomFieldName(entity: string): string {
  const commonFields = ['id', 'createdAt', 'updatedAt'];
  
  const entityFields: Record<string, string[]> = {
    'User': ['name', 'email', 'password', 'role', 'status', 'lastLogin', 'preferences'],
    'Product': ['name', 'description', 'price', 'stock', 'categoryId', 'sku', 'weight', 'dimensions'],
    'Order': ['userId', 'status', 'total', 'paymentMethod', 'shippingAddress', 'billingAddress', 'trackingNumber'],
    'Category': ['name', 'description', 'parentId', 'slug', 'featured', 'icon'],
    'Review': ['userId', 'productId', 'rating', 'comment', 'title', 'verified', 'helpful']
  };
  
  const fields = [...commonFields, ...(entityFields[entity] || [])];
  return fields[Math.floor(Math.random() * fields.length)];
}

/**
 * Get a random field type
 */
function getRandomFieldType(): string {
  const types = ['string', 'number', 'boolean', 'Date', 'object', 'array', 'enum'];
  return types[Math.floor(Math.random() * types.length)];
}

/**
 * Get a random field description
 */
function getRandomFieldDescription(): string {
  const descriptions = [
    'Unique identifier',
    'Creation timestamp',
    'Last update timestamp',
    'User name',
    'Email address',
    'Hashed password',
    'User role (admin, user, etc.)',
    'Account status',
    'Product name',
    'Product description',
    'Price in USD',
    'Available stock',
    'Category identifier',
    'Order status',
    'Total order amount',
    'User who placed the order',
    'Rating from 1-5',
    'User review comment'
  ];
  return descriptions[Math.floor(Math.random() * descriptions.length)];
}

// ----------------------------------------------------------------------
// Register API Routes
// ----------------------------------------------------------------------

/**
 * Register continuous execution routes to the API router
 */
export function registerContinuousExecutionRoutes(router: any) {
  // Project management endpoints
  router.post('/project', startAutonomousProject);
  router.get('/project/:id', getProjectStatus);
  router.get('/project/:id/artifacts/:artifactType?', getProjectArtifacts);
  router.post('/project/:id/action', updateProjectExecution);
}