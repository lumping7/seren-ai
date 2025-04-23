/**
 * OpenManus Integration Module
 * 
 * This module integrates the OpenManus agent system with our Node.js backend,
 * creating a bridge between the Python-based agentic system and the TypeScript backend.
 * It enables end-to-end autonomous software development through role-specific agents.
 */

import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { v4 as uuidv4 } from 'uuid';
// Model integration interface for executing tasks
interface ModelIntegrationInterface {
  executeTask: (params: {
    model: string;
    role: string;
    content: string;
    context?: any;
  }) => Promise<string>;
}

// Error handler interface for handling errors
interface ErrorHandlerInterface {
  handleError: (source: string, error: any) => void;
}

// Performance monitor interface for tracking operations
interface PerformanceMonitorInterface {
  startOperation: (name: string, id?: string) => string;
  endOperation: (id: string, failed?: boolean) => void;
}

// Define agent roles matching Python implementation
export enum AgentRole {
  PLANNER = 'planner',
  CODER = 'coder',
  TESTER = 'tester',
  REVIEWER = 'reviewer',
  FIXER = 'fixer',
  RESEARCHER = 'researcher',
  INTEGRATOR = 'integrator'
}

// Define task status matching Python implementation
export enum TaskStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  BLOCKED = 'blocked',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

// Define task priority matching Python implementation
export enum TaskPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Interface for project requirements
export interface ProjectRequirements {
  name: string;
  description: string;
  language?: string;
  framework?: string;
  features?: string[];
  constraints?: string[];
  [key: string]: any;
}

// Interface for task definition
export interface Task {
  task_id: string;
  name: string;
  description: string;
  role: string;
  dependencies: string[];
  status: string;
  priority: string;
  parent_id?: string;
  context: any;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  assigned_to?: string;
  result?: any;
  error?: string;
  subtasks: Task[];
  logs: {
    timestamp: string;
    level: string;
    message: string;
  }[];
}

// Interface for project status
export interface ProjectStatus {
  project_id: string;
  project_name: string;
  status: string;
  progress: number;
  subtask_counts: {
    total: number;
    pending: number;
    in_progress: number;
    completed: number;
    failed: number;
  };
  started_at?: string;
  completed_at?: string;
  requirements: string;
  language: string;
  framework: string;
  recent_logs: any[];
  error?: string;
}

/**
 * OpenManus Integration Class
 * 
 * This class provides methods to communicate with the Python-based OpenManus agent system,
 * allowing the creation and management of autonomous software development projects.
 */
export class OpenManusIntegration {
  private agentProcess: ChildProcess | null = null;
  private modelIntegration: ModelIntegrationInterface;
  private errorHandler: ErrorHandlerInterface;
  private performanceMonitor: PerformanceMonitorInterface;
  private projectsDir: string;
  private projectStates: Map<string, ProjectStatus> = new Map();
  private messageQueue: any[] = [];
  private isProcessingQueue: boolean = false;
  private isInitialized: boolean = false;

  constructor(
    modelIntegration: ModelIntegration,
    errorHandler: ErrorHandler,
    performanceMonitor: PerformanceMonitor,
    projectsDir: string = path.join(process.cwd(), 'projects')
  ) {
    this.modelIntegration = modelIntegration;
    this.errorHandler = errorHandler;
    this.performanceMonitor = performanceMonitor;
    this.projectsDir = projectsDir;

    // Ensure projects directory exists
    if (!existsSync(this.projectsDir)) {
      mkdirSync(this.projectsDir, { recursive: true });
    }
  }

  /**
   * Initialize the OpenManus integration by starting the Python agent system
   */
  public async initialize(): Promise<boolean> {
    if (this.isInitialized) {
      console.log('[OpenManus] Already initialized');
      return true;
    }

    try {
      // For logging and debugging only - we'll properly initialize through wrapper functions
      console.log('[OpenManus] Initializing OpenManus integration');
      this.isInitialized = true;
      return true;
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.initialize', error);
      return false;
    }
  }

  /**
   * Create a new software development project
   * @param requirements Project requirements
   * @returns Project ID
   */
  public async createProject(requirements: ProjectRequirements): Promise<string> {
    try {
      await this.initialize();
      
      const operationId = this.performanceMonitor.startOperation('createProject');
      
      // Generate a unique project ID
      const projectId = uuidv4();
      
      // Create project directory
      const projectDir = path.join(this.projectsDir, projectId);
      if (!existsSync(projectDir)) {
        mkdirSync(projectDir, { recursive: true });
      }
      
      // Save project requirements
      const reqPath = path.join(projectDir, 'requirements.json');
      writeFileSync(reqPath, JSON.stringify(requirements, null, 2));
      
      // Initialize project status
      const projectStatus: ProjectStatus = {
        project_id: projectId,
        project_name: requirements.name,
        status: TaskStatus.PENDING,
        progress: 0,
        subtask_counts: {
          total: 0,
          pending: 0,
          in_progress: 0,
          completed: 0,
          failed: 0
        },
        requirements: requirements.description,
        language: requirements.language || '',
        framework: requirements.framework || '',
        recent_logs: [
          {
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Project created'
          }
        ]
      };
      
      this.projectStates.set(projectId, projectStatus);
      
      // Create task decomposition using the models
      const decomposition = await this.decomposeProject(requirements);
      
      // Update project status
      projectStatus.subtask_counts.total = decomposition.subtasks ? decomposition.subtasks.length : 0;
      projectStatus.subtask_counts.pending = projectStatus.subtask_counts.total;
      projectStatus.status = TaskStatus.IN_PROGRESS;
      projectStatus.started_at = new Date().toISOString();
      
      // Save task decomposition
      const taskPath = path.join(projectDir, 'tasks.json');
      writeFileSync(taskPath, JSON.stringify(decomposition, null, 2));
      
      this.performanceMonitor.endOperation(operationId);
      
      // Start autonomous execution in background
      this.startProjectExecution(projectId);
      
      return projectId;
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.createProject', error);
      throw error;
    }
  }

  /**
   * Get the status of a project
   * @param projectId Project ID
   * @returns Project status
   */
  public getProjectStatus(projectId: string): ProjectStatus {
    try {
      const status = this.projectStates.get(projectId);
      if (!status) {
        // Try to load from file system
        const projectDir = path.join(this.projectsDir, projectId);
        if (existsSync(projectDir)) {
          const statusPath = path.join(projectDir, 'status.json');
          if (existsSync(statusPath)) {
            const statusData = JSON.parse(readFileSync(statusPath, 'utf8'));
            this.projectStates.set(projectId, statusData);
            return statusData;
          }
        }
        
        throw new Error(`Project not found: ${projectId}`);
      }
      
      return status;
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.getProjectStatus', error);
      throw error;
    }
  }

  /**
   * Get all projects
   * @returns Array of project statuses
   */
  public getAllProjects(): ProjectStatus[] {
    try {
      return Array.from(this.projectStates.values());
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.getAllProjects', error);
      throw error;
    }
  }

  /**
   * Get the generated files for a project
   * @param projectId Project ID
   * @returns Object mapping file paths to file contents
   */
  public getProjectFiles(projectId: string): Record<string, string> {
    try {
      const projectDir = path.join(this.projectsDir, projectId);
      if (!existsSync(projectDir)) {
        throw new Error(`Project directory not found: ${projectId}`);
      }
      
      const outputDir = path.join(projectDir, 'output');
      if (!existsSync(outputDir)) {
        return {};
      }
      
      // This function would recursively read all files in the output directory
      // For brevity, we're not implementing it here but would return something like:
      // { 'src/main.js': '...content...', 'src/utils.js': '...content...' }
      
      return {}; // Placeholder implementation
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.getProjectFiles', error);
      throw error;
    }
  }

  /**
   * Decompose a project into tasks using the architecture model
   * @param requirements Project requirements
   * @returns Task decomposition
   */
  private async decomposeProject(requirements: ProjectRequirements): Promise<Task> {
    try {
      const operationId = this.performanceMonitor.startOperation('decomposeProject');
      
      // Create the main project task
      const mainTask: Task = {
        task_id: uuidv4(),
        name: `Create ${requirements.name}`,
        description: requirements.description,
        role: AgentRole.PLANNER,
        dependencies: [],
        status: TaskStatus.PENDING,
        priority: TaskPriority.HIGH,
        context: requirements,
        created_at: new Date().toISOString(),
        subtasks: [],
        logs: [
          {
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Project decomposition started'
          }
        ]
      };
      
      // Generate the task breakdown using the Qwen model (architecture model)
      const prompt = `
      You are an expert software architect and project planner. Given the following project requirements,
      break down the project into a structured set of tasks for autonomous implementation.
      
      Project Name: ${requirements.name}
      Description: ${requirements.description}
      Language: ${requirements.language || 'Not specified'}
      Framework: ${requirements.framework || 'Not specified'}
      
      For each task, specify:
      1. A clear name and description
      2. The role needed to complete it (planner, coder, tester, reviewer, fixer, researcher, integrator)
      3. Dependencies (which tasks must be completed first)
      4. Priority (low, medium, high, critical)
      
      Start with high-level planning tasks, then architecture design, 
      then break down implementation into logical components.
      Include tasks for testing, documentation, and quality assurance.
      
      Format your response as a nested JSON structure.
      `;
      
      // Use the model to generate the task breakdown
      const result = await this.modelIntegration.executeTask({
        model: 'qwen2.5-7b-omni',
        role: 'architect',
        content: prompt,
      });
      
      // Parse the JSON content from the result
      try {
        let jsonContent;
        const jsonMatch = result.match(/```json([\s\S]*?)```/);
        
        if (jsonMatch) {
          jsonContent = jsonMatch[1].trim();
        } else {
          jsonContent = result;
        }
        
        // Parse the tasks
        const taskStructure = JSON.parse(jsonContent);
        
        // Recursively convert the structure to our Task interface
        const convertTaskStructure = (task: any, parentId?: string): Task => {
          const newTask: Task = {
            task_id: uuidv4(),
            name: task.name || 'Unnamed Task',
            description: task.description || '',
            role: task.role || AgentRole.PLANNER,
            dependencies: task.dependencies || [],
            status: TaskStatus.PENDING,
            priority: task.priority || TaskPriority.MEDIUM,
            parent_id: parentId,
            context: task.context || {},
            created_at: new Date().toISOString(),
            subtasks: [],
            logs: []
          };
          
          if (task.subtasks) {
            newTask.subtasks = task.subtasks.map((subtask: any) => 
              convertTaskStructure(subtask, newTask.task_id)
            );
          }
          
          return newTask;
        };
        
        // Apply the subtasks to the main task
        mainTask.subtasks = taskStructure.subtasks.map((subtask: any) => 
          convertTaskStructure(subtask, mainTask.task_id)
        );
      } catch (parseError) {
        console.error('[OpenManus] Error parsing task structure:', parseError);
        // Add placeholder subtasks if parsing fails
        mainTask.subtasks = [
          {
            task_id: uuidv4(),
            name: 'Architecture Design',
            description: 'Design the high-level architecture for the project',
            role: AgentRole.PLANNER,
            dependencies: [],
            status: TaskStatus.PENDING,
            priority: TaskPriority.HIGH,
            parent_id: mainTask.task_id,
            context: requirements,
            created_at: new Date().toISOString(),
            subtasks: [],
            logs: []
          },
          {
            task_id: uuidv4(),
            name: 'Implementation',
            description: 'Implement the core functionality of the project',
            role: AgentRole.CODER,
            dependencies: [mainTask.subtasks[0].task_id],
            status: TaskStatus.PENDING,
            priority: TaskPriority.HIGH,
            parent_id: mainTask.task_id,
            context: requirements,
            created_at: new Date().toISOString(),
            subtasks: [],
            logs: []
          },
          {
            task_id: uuidv4(),
            name: 'Testing',
            description: 'Test the implemented functionality',
            role: AgentRole.TESTER,
            dependencies: [mainTask.subtasks[1].task_id],
            status: TaskStatus.PENDING,
            priority: TaskPriority.MEDIUM,
            parent_id: mainTask.task_id,
            context: requirements,
            created_at: new Date().toISOString(),
            subtasks: [],
            logs: []
          }
        ];
      }
      
      this.performanceMonitor.endOperation(operationId);
      
      return mainTask;
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.decomposeProject', error);
      throw error;
    }
  }

  /**
   * Start the autonomous execution of a project
   * @param projectId Project ID
   */
  private async startProjectExecution(projectId: string): Promise<void> {
    try {
      const projectDir = path.join(this.projectsDir, projectId);
      const taskPath = path.join(projectDir, 'tasks.json');
      const statusPath = path.join(projectDir, 'status.json');
      
      if (!existsSync(taskPath)) {
        throw new Error(`Task file not found for project: ${projectId}`);
      }
      
      const projectTasks = JSON.parse(readFileSync(taskPath, 'utf8'));
      const status = this.projectStates.get(projectId) as ProjectStatus;
      
      // Create output directory
      const outputDir = path.join(projectDir, 'output');
      if (!existsSync(outputDir)) {
        mkdirSync(outputDir, { recursive: true });
      }
      
      // Update status
      status.status = TaskStatus.IN_PROGRESS;
      status.recent_logs.push({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: 'Project execution started'
      });
      
      // Save status
      writeFileSync(statusPath, JSON.stringify(status, null, 2));
      
      // Execute tasks in a background process to not block the server
      // This would be a more complex implementation in a real system
      // For now, we'll simulate async task execution
      
      // Start with the architecture design task
      const architectureTask = projectTasks.subtasks.find(
        (task: Task) => task.role === AgentRole.PLANNER
      );
      
      if (architectureTask) {
        // Queue the architecture task for execution
        this.queueTaskExecution(projectId, architectureTask);
      }
      
      console.log(`[OpenManus] Started project execution for project: ${projectId}`);
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.startProjectExecution', error);
    }
  }

  /**
   * Queue a task for execution
   * @param projectId Project ID
   * @param task Task to execute
   */
  private queueTaskExecution(projectId: string, task: Task): void {
    this.messageQueue.push({
      type: 'execute_task',
      projectId,
      task
    });
    
    // Start processing the queue if not already processing
    if (!this.isProcessingQueue) {
      this.processQueue();
    }
  }

  /**
   * Process the message queue
   */
  private async processQueue(): Promise<void> {
    if (this.messageQueue.length === 0) {
      this.isProcessingQueue = false;
      return;
    }
    
    this.isProcessingQueue = true;
    const message = this.messageQueue.shift();
    
    try {
      if (message.type === 'execute_task') {
        await this.executeTask(message.projectId, message.task);
      }
    } catch (error) {
      this.errorHandler.handleError('OpenManusIntegration.processQueue', error);
    }
    
    // Process next message
    setTimeout(() => this.processQueue(), 100);
  }

  /**
   * Execute a task using the appropriate model
   * @param projectId Project ID
   * @param task Task to execute
   */
  private async executeTask(projectId: string, task: Task): Promise<void> {
    try {
      console.log(`[OpenManus] Executing task: ${task.name} (${task.task_id})`);
      
      const operationId = this.performanceMonitor.startOperation(`executeTask_${task.task_id}`);
      
      // Update task status
      task.status = TaskStatus.IN_PROGRESS;
      task.started_at = new Date().toISOString();
      
      // Update project status
      const status = this.projectStates.get(projectId) as ProjectStatus;
      status.subtask_counts.pending--;
      status.subtask_counts.in_progress++;
      status.recent_logs.push({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: `Executing task: ${task.name}`
      });
      
      // Save status
      const statusPath = path.join(this.projectsDir, projectId, 'status.json');
      writeFileSync(statusPath, JSON.stringify(status, null, 2));
      
      // Choose model based on role
      let model;
      let role;
      switch (task.role) {
        case AgentRole.PLANNER:
        case AgentRole.REVIEWER:
        case AgentRole.RESEARCHER:
          model = 'qwen2.5-7b-omni';
          role = 'architect';
          break;
        case AgentRole.CODER:
        case AgentRole.FIXER:
        case AgentRole.INTEGRATOR:
          model = 'olympiccoder-7b';
          role = 'builder';
          break;
        case AgentRole.TESTER:
          model = 'hybrid';  // Use both models
          role = 'tester';
          break;
        default:
          model = 'qwen2.5-7b-omni';
          role = 'assistant';
      }
      
      // Generate prompt based on task
      const prompt = this.generateTaskPrompt(task);
      
      // Execute with the model
      const result = await this.modelIntegration.executeTask({
        model,
        role,
        content: prompt,
        context: {
          projectId,
          taskId: task.task_id,
          role: task.role
        }
      });
      
      // Process the result
      const processedResult = this.processTaskResult(task, result);
      
      // Complete the task
      task.status = TaskStatus.COMPLETED;
      task.completed_at = new Date().toISOString();
      task.result = processedResult;
      
      // Update project status
      status.subtask_counts.in_progress--;
      status.subtask_counts.completed++;
      status.progress = (status.subtask_counts.completed / status.subtask_counts.total) * 100;
      status.recent_logs.push({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: `Completed task: ${task.name}`
      });
      
      // Save status and task
      writeFileSync(statusPath, JSON.stringify(status, null, 2));
      
      // Get task file and update it
      const taskPath = path.join(this.projectsDir, projectId, 'tasks.json');
      const taskData = JSON.parse(readFileSync(taskPath, 'utf8'));
      this.updateTaskInTree(taskData, task);
      writeFileSync(taskPath, JSON.stringify(taskData, null, 2));
      
      // Save output if applicable
      if (processedResult.files) {
        const outputDir = path.join(this.projectsDir, projectId, 'output');
        if (!existsSync(outputDir)) {
          mkdirSync(outputDir, { recursive: true });
        }
        
        for (const [filePath, content] of Object.entries(processedResult.files)) {
          const fullPath = path.join(outputDir, filePath);
          const dirName = path.dirname(fullPath);
          
          // Create directories if needed
          if (!existsSync(dirName)) {
            mkdirSync(dirName, { recursive: true });
          }
          
          writeFileSync(fullPath, content);
        }
      }
      
      this.performanceMonitor.endOperation(operationId);
      
      // Queue dependent tasks
      this.queueDependentTasks(projectId, taskData, task.task_id);
      
      // Check if project is complete
      if (status.subtask_counts.completed + status.subtask_counts.failed === status.subtask_counts.total) {
        status.status = TaskStatus.COMPLETED;
        status.completed_at = new Date().toISOString();
        status.recent_logs.push({
          timestamp: new Date().toISOString(),
          level: 'info',
          message: 'Project completed'
        });
        
        writeFileSync(statusPath, JSON.stringify(status, null, 2));
      }
    } catch (error) {
      this.errorHandler.handleError(`OpenManusIntegration.executeTask_${task.task_id}`, error);
      
      // Mark task as failed
      task.status = TaskStatus.FAILED;
      task.error = error.message;
      
      // Update project status
      const status = this.projectStates.get(projectId) as ProjectStatus;
      status.subtask_counts.in_progress--;
      status.subtask_counts.failed++;
      status.recent_logs.push({
        timestamp: new Date().toISOString(),
        level: 'error',
        message: `Failed task: ${task.name} - ${error.message}`
      });
      
      // Save status and task
      const statusPath = path.join(this.projectsDir, projectId, 'status.json');
      writeFileSync(statusPath, JSON.stringify(status, null, 2));
      
      const taskPath = path.join(this.projectsDir, projectId, 'tasks.json');
      const taskData = JSON.parse(readFileSync(taskPath, 'utf8'));
      this.updateTaskInTree(taskData, task);
      writeFileSync(taskPath, JSON.stringify(taskData, null, 2));
    }
  }

  /**
   * Generate a prompt for a task based on its role
   * @param task Task
   * @returns Prompt string
   */
  private generateTaskPrompt(task: Task): string {
    switch (task.role) {
      case AgentRole.PLANNER:
        return `
        You are an expert software architect and planner.
        
        Task: ${task.name}
        Description: ${task.description}
        
        Project name: ${task.context.name || 'Software Project'}
        
        Requirements:
        ${task.context.description || 'No specific requirements provided.'}
        
        Language: ${task.context.language || 'Not specified'}
        Framework: ${task.context.framework || 'Not specified'}
        
        Your job is to:
        1. Analyze the requirements thoroughly
        2. Design a high-level architecture for the software
        3. Break down the project into specific, actionable components
        4. Prioritize these components in a logical order
        5. Provide a detailed technical specification
        
        Please provide a comprehensive project plan and architecture design.
        Include file structure, key components, data models, and API endpoints.
        
        Format your response as markdown, with code blocks for diagrams and examples.
        Also include a JSON section with a list of files that need to be created:
        
        \`\`\`json
        {
          "files": {
            "example.js": "content of the file",
            "subfolder/example2.js": "content of the second file"
          }
        }
        \`\`\`
        `;
      
      case AgentRole.CODER:
        return `
        You are an expert software developer.
        
        Task: ${task.name}
        Description: ${task.description}
        
        Project context:
        ${JSON.stringify(task.context, null, 2)}
        
        Your job is to:
        1. Implement the code according to the specification
        2. Follow best practices for ${task.context.language || 'the specified language'}
        3. Include appropriate comments and documentation
        4. Ensure the code is efficient, secure, and maintainable
        5. Handle potential edge cases and errors
        
        Please provide the implementation code.
        Format your response with each file in its own code block, preceded by the filename:
        
        # Filename: example.js
        \`\`\`javascript
        // Code content here
        \`\`\`
        
        Then provide a JSON summary of all files:
        
        \`\`\`json
        {
          "files": {
            "example.js": "// Code content here",
            "subfolder/example2.js": "// More code here"
          }
        }
        \`\`\`
        `;
      
      case AgentRole.TESTER:
        return `
        You are an expert software tester.
        
        Task: ${task.name}
        Description: ${task.description}
        
        Project context:
        ${JSON.stringify(task.context, null, 2)}
        
        Your job is to:
        1. Create comprehensive test cases for the code
        2. Test for correctness, performance, and security
        3. Identify any bugs or issues
        4. Ensure the code meets all requirements
        5. Verify edge cases and error handling
        
        Please provide the test cases and any testing code in appropriate files.
        
        Format your response with each file in its own code block, preceded by the filename,
        and include a summary of test cases and findings.
        
        Then provide a JSON summary of all test files:
        
        \`\`\`json
        {
          "files": {
            "tests/example.test.js": "// Test code here"
          },
          "findings": [
            {
              "type": "bug",
              "severity": "medium",
              "description": "Description of the issue",
              "file": "path/to/file.js",
              "line": 42
            }
          ]
        }
        \`\`\`
        `;
      
      case AgentRole.REVIEWER:
        return `
        You are an expert code reviewer.
        
        Task: ${task.name}
        Description: ${task.description}
        
        Project context:
        ${JSON.stringify(task.context, null, 2)}
        
        Your job is to:
        1. Review the code for clarity, efficiency, and correctness
        2. Identify any potential bugs, security issues, or performance problems
        3. Suggest improvements to the code structure and organization
        4. Ensure the code follows industry best practices
        5. Verify that the code meets all requirements
        
        Please provide a detailed code review in markdown format.
        Organize your review by file, and include specific line references for each issue.
        
        For each issue, provide:
        1. The severity (critical, high, medium, low)
        2. A clear description of the problem
        3. A suggested fix or improvement
        
        Then provide a JSON summary of all issues:
        
        \`\`\`json
        {
          "files": {
            "example.js": [
              {
                "line": 42,
                "severity": "medium",
                "issue": "Description of the issue",
                "suggestion": "Suggested fix"
              }
            ]
          }
        }
        \`\`\`
        `;
      
      case AgentRole.FIXER:
        return `
        You are an expert debugger and bug fixer.
        
        Task: ${task.name}
        Description: ${task.description}
        
        Project context:
        ${JSON.stringify(task.context, null, 2)}
        
        Your job is to:
        1. Identify the root cause of the bug or issue
        2. Fix the problem without introducing new issues
        3. Ensure the solution is compatible with the existing codebase
        4. Verify that the fix resolves the issue completely
        5. Document the fix and the reason for the bug
        
        Please provide the fixed code files and an explanation of each issue.
        
        Format your response with each file in its own code block, preceded by the filename.
        For each file, highlight the changes you made and explain why.
        
        Then provide a JSON summary of all fixed files:
        
        \`\`\`json
        {
          "files": {
            "example.js": "// Fixed code content"
          },
          "fixes": [
            {
              "file": "example.js",
              "line": 42,
              "issue": "Description of the issue",
              "fix": "Description of the fix"
            }
          ]
        }
        \`\`\`
        `;
      
      default:
        return `
        Task: ${task.name}
        Description: ${task.description}
        
        Context:
        ${JSON.stringify(task.context, null, 2)}
        
        Please complete this task based on the given requirements and context.
        Provide a detailed response with any relevant files, code, or documentation.
        `;
    }
  }

  /**
   * Process the result of a task execution
   * @param task Task that was executed
   * @param result Result from the model
   * @returns Processed result
   */
  private processTaskResult(task: Task, result: string): any {
    try {
      // Default result structure
      const processedResult = {
        output: result,
        files: {}
      };
      
      // Try to extract JSON data
      const jsonMatch = result.match(/```json([\s\S]*?)```/);
      if (jsonMatch) {
        try {
          const jsonData = JSON.parse(jsonMatch[1].trim());
          
          // Check for files property
          if (jsonData.files) {
            processedResult.files = jsonData.files;
          }
          
          // Check for other properties
          for (const key in jsonData) {
            if (key !== 'files') {
              processedResult[key] = jsonData[key];
            }
          }
        } catch (jsonError) {
          console.error('[OpenManus] Failed to parse JSON from result:', jsonError);
        }
      }
      
      // If we have a coder task but no files were extracted, try to extract them from code blocks
      if (task.role === AgentRole.CODER && Object.keys(processedResult.files).length === 0) {
        // Find filename headers and code blocks
        const fileRegex = /# Filename: (\S+)\n```(?:\w+)?\n([\s\S]*?)```/g;
        let match;
        
        while ((match = fileRegex.exec(result)) !== null) {
          const filename = match[1];
          const content = match[2];
          
          processedResult.files[filename] = content;
        }
      }
      
      return processedResult;
    } catch (error) {
      console.error('[OpenManus] Error processing task result:', error);
      return { output: result };
    }
  }

  /**
   * Update a task in the task tree
   * @param taskTree Task tree
   * @param updatedTask Updated task
   * @returns Whether the task was found and updated
   */
  private updateTaskInTree(taskTree: Task, updatedTask: Task): boolean {
    if (taskTree.task_id === updatedTask.task_id) {
      // Update all fields except subtasks
      for (const key in updatedTask) {
        if (key !== 'subtasks') {
          taskTree[key] = updatedTask[key];
        }
      }
      return true;
    }
    
    // Check subtasks
    if (taskTree.subtasks) {
      for (let i = 0; i < taskTree.subtasks.length; i++) {
        if (this.updateTaskInTree(taskTree.subtasks[i], updatedTask)) {
          return true;
        }
      }
    }
    
    return false;
  }

  /**
   * Queue dependent tasks for execution
   * @param projectId Project ID
   * @param taskTree Task tree
   * @param completedTaskId ID of the completed task
   */
  private queueDependentTasks(projectId: string, taskTree: Task, completedTaskId: string): void {
    // Recursively search for tasks that depend on the completed task
    const findDependentTasks = (task: Task): Task[] => {
      let dependentTasks: Task[] = [];
      
      // Check if this task depends on the completed task
      if (task.dependencies && task.dependencies.includes(completedTaskId) && task.status === TaskStatus.PENDING) {
        // Check if all dependencies are completed
        const allDepsCompleted = task.dependencies.every(depId => {
          const depStatus = this.findTaskStatus(taskTree, depId);
          return depStatus === TaskStatus.COMPLETED;
        });
        
        if (allDepsCompleted) {
          dependentTasks.push(task);
        }
      }
      
      // Check subtasks
      if (task.subtasks) {
        for (const subtask of task.subtasks) {
          dependentTasks = [...dependentTasks, ...findDependentTasks(subtask)];
        }
      }
      
      return dependentTasks;
    };
    
    const dependentTasks = findDependentTasks(taskTree);
    
    // Queue each dependent task
    for (const task of dependentTasks) {
      this.queueTaskExecution(projectId, task);
    }
  }

  /**
   * Find the status of a task in the task tree
   * @param taskTree Task tree
   * @param taskId Task ID
   * @returns Task status or undefined if not found
   */
  private findTaskStatus(taskTree: Task, taskId: string): string | undefined {
    if (taskTree.task_id === taskId) {
      return taskTree.status;
    }
    
    // Check subtasks
    if (taskTree.subtasks) {
      for (const subtask of taskTree.subtasks) {
        const status = this.findTaskStatus(subtask, taskId);
        if (status) {
          return status;
        }
      }
    }
    
    return undefined;
  }
}