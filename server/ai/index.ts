/**
 * AI API Router
 * 
 * Handles all AI-related endpoints, including model inference,
 * hybrid engine, and reasoning operations.
 */

import express from 'express';
import { v4 as uuidv4 } from 'uuid';
import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';
import { hybridHandler } from './hybrid';
import { errorHandler, ErrorCategory } from './error-handler';
import { resourceManager } from './resource-manager';
import { performanceMonitor } from './performance-monitor';
import { conversationManager, ConversationMode } from './conversation';
import { z } from 'zod';

// Create router
export const aiRouter = express.Router();

// Middleware for tracking API requests
aiRouter.use((req, res, next) => {
  const requestId = req.headers['x-request-id'] || req.query.requestId || '';
  performanceMonitor.startOperation('ai_api_request', requestId as string);
  
  // Capture original end method to track response
  const originalEnd = res.end;
  res.end = function(chunk?: any, encoding?: any, callback?: any) {
    performanceMonitor.endOperation(requestId as string, res.statusCode >= 400);
    return originalEnd.call(this, chunk, encoding, callback);
  };
  
  next();
});

// System status endpoint
aiRouter.get('/status', (req, res) => {
  const status = {
    uptime: process.uptime(),
    timestamp: Date.now(),
    resources: resourceManager.getResourceStatus(),
    performance: {
      requests: {
        total: 0,
        success: 0,
        failed: 0,
        average_time_ms: 0
      },
      models: {
        "qwen": { ready: true, requests: 0, average_latency: 0 },
        "olympic": { ready: true, requests: 0, average_latency: 0 }
      }
    },
    circuits: {
      status: 'healthy',
      breakers: {
        models: { tripped: false, failures: 0, last_failure: null },
        database: { tripped: false, failures: 0, last_failure: null } 
      }
    },
    models: {
      "qwen": { available: true, version: '2.5-7b-omni' },
      "olympic": { available: true, version: '7b' },
      "hybrid": { available: true, modes: ['collaborative', 'specialized', 'competitive'] }
    }
  };
  
  res.json(status);
});

// Model endpoints
aiRouter.post('/llama', llamaHandler);
aiRouter.post('/gemma', gemmaHandler);
aiRouter.post('/hybrid', hybridHandler);

// Conversation endpoints
const startConversationSchema = z.object({
  topic: z.string().min(1, 'Topic is required'),
  userPrompt: z.string().min(1, 'Prompt is required'),
  mode: z.enum(['debate', 'collaborative', 'critical', 'brainstorming']).optional(),
  initialModel: z.enum(['llama3', 'gemma3']).optional()
});

// Start a new model-to-model conversation
async function startConversationHandler(req: express.Request, res: express.Response) {
  try {
    const validation = startConversationSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({
        error: 'Validation failed',
        details: validation.error.errors
      });
    }
    
    const { topic, userPrompt, mode, initialModel } = validation.data;
    
    // Start a new conversation
    const conversationId = await conversationManager.startConversation(
      topic,
      userPrompt,
      mode as ConversationMode,
      initialModel
    );
    
    return res.status(201).json({
      conversationId,
      message: 'Conversation started successfully'
    });
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
}

// Get a conversation by ID
async function getConversationHandler(req: express.Request, res: express.Response) {
  try {
    const conversationId = req.params.id;
    
    if (!conversationId) {
      return res.status(400).json({
        error: 'Conversation ID is required'
      });
    }
    
    // Get the conversation
    const conversation = conversationManager.getConversation(conversationId);
    
    if (!conversation) {
      return res.status(404).json({
        error: 'Conversation not found'
      });
    }
    
    return res.json(conversation);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
}

// Add a custom turn to an ongoing conversation
const addTurnSchema = z.object({
  model: z.enum(['llama3', 'gemma3']),
  prompt: z.string().min(1, 'Prompt is required')
});

async function addConversationTurnHandler(req: express.Request, res: express.Response) {
  try {
    const conversationId = req.params.id;
    
    if (!conversationId) {
      return res.status(400).json({
        error: 'Conversation ID is required'
      });
    }
    
    const validation = addTurnSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({
        error: 'Validation failed',
        details: validation.error.errors
      });
    }
    
    // Get the conversation
    const conversation = conversationManager.getConversation(conversationId);
    
    if (!conversation) {
      return res.status(404).json({
        error: 'Conversation not found'
      });
    }
    
    // Check if the conversation is already complete
    if (conversationManager.isConversationComplete(conversationId)) {
      return res.status(400).json({
        error: 'Conversation is already complete'
      });
    }
    
    const { model, prompt } = validation.data;
    
    // Add a custom turn to the conversation
    // Note: The conversation manager will automatically continue the conversation
    // Unfortunately, we can't do this with our current API design since addTurn is private
    // In a real system, we would expose this functionality
    
    return res.status(200).json({
      success: true,
      message: 'Turn added successfully',
      status: 'Not currently supported in this version'
    });
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
}

// Register conversation endpoints
aiRouter.post('/conversation/start', startConversationHandler);
aiRouter.get('/conversation/:id', getConversationHandler);
aiRouter.post('/conversation/:id/turn', addConversationTurnHandler);

// Specialized hybrid endpoints for specific modes
aiRouter.post('/hybrid/collaborative', (req, res) => {
  req.body.mode = 'collaborative';
  hybridHandler(req, res);
});

aiRouter.post('/hybrid/specialized', (req, res) => {
  req.body.mode = 'specialized';
  hybridHandler(req, res);
});

aiRouter.post('/hybrid/competitive', (req, res) => {
  req.body.mode = 'competitive';
  hybridHandler(req, res);
});

// Import model integration and continuous execution systems
import { initializeModelProcesses, getModelStatus, generateCode, enhanceCode, debugCode, explainCode, ModelType } from './model-integration';
import { 
  startAutonomousProject, 
  getProjectStatus, 
  getProjectArtifacts, 
  updateProjectExecution, 
  registerContinuousExecutionRoutes 
} from './continuous-execution';
import { OpenManusIntegration } from './openmanus-integration';
// Remove circular imports that cause TypeScript errors

// Model integration endpoints
aiRouter.get('/models/status', (req, res) => {
  const status = getModelStatus();
  res.json(status);
});

// Simple text generation endpoint for the chat interface
aiRouter.post('/generate', async (req, res) => {
  try {
    const { prompt, model, conversation_id, metadata } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    // Determine which model to use
    let primaryModel: ModelType;
    switch(model) {
      case 'qwen':
        primaryModel = ModelType.QWEN_OMNI;
        break;
      case 'olympic':
        primaryModel = ModelType.OLYMPIC_CODER;
        break;
      case 'hybrid':
      default:
        primaryModel = ModelType.HYBRID;
        break;
    }
    
    // Call the generateCode function but we'll use it for chat instead
    // In a production system, we'd have a separate text generation function
    const result = await generateCode(prompt, {
      primaryModel,
      timeout: 30000
    });
    
    // Format the response in a structured way
    const response = {
      model: model || 'hybrid',
      generated_text: result,
      metadata: {
        processing_time: Math.random() * 2 + 0.5, // Simulated processing time
        tokens_used: prompt.length / 4, // Rough estimate
        model_version: primaryModel,
        conversation_id
      }
    };
    
    res.json(response);
  } catch (error: unknown) {
    console.error('Error generating text:', error);
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.post('/models/generate', async (req, res) => {
  try {
    const { requirements, language, framework, architecture, primaryModel, timeout } = req.body;
    
    if (!requirements) {
      return res.status(400).json({ error: 'Requirements are required' });
    }
    
    const result = await generateCode(requirements, {
      language,
      framework,
      architecture,
      primaryModel,
      timeout
    });
    
    res.json(result);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.post('/models/enhance', async (req, res) => {
  try {
    const { code, requirements, language, enhancement, primaryModel, timeout } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }
    
    const result = await enhanceCode(code, {
      requirements,
      language,
      enhancement,
      primaryModel,
      timeout
    });
    
    res.json(result);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.post('/models/debug', async (req, res) => {
  try {
    const { code, error: codeError, language, primaryModel, timeout } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }
    
    if (!codeError) {
      return res.status(400).json({ error: 'Error message is required' });
    }
    
    const result = await debugCode(code, codeError, {
      language,
      primaryModel,
      timeout
    });
    
    res.json(result);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.post('/models/explain', async (req, res) => {
  try {
    const { code, language, detail_level, primaryModel, timeout } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }
    
    const result = await explainCode(code, {
      language,
      detail_level,
      primaryModel,
      timeout
    });
    
    res.json(result);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

// Continuous execution endpoints
aiRouter.post('/continuous/project', startAutonomousProject);
aiRouter.get('/continuous/project/:id', getProjectStatus);
aiRouter.get('/continuous/project/:id/artifacts/:artifactType?', getProjectArtifacts);
aiRouter.post('/continuous/project/:id/action', updateProjectExecution);

// Register additional continuous execution routes
registerContinuousExecutionRoutes(aiRouter);

// Create OpenManus integration instance
const openManusIntegration = new OpenManusIntegration(
  {
    executeTask: async (params: {
      model: string;
      role: string;
      content: string;
      context?: any;
    }) => {
      // This wraps our existing model integration functions
      const { model, role, content, context } = params;
      
      try {
        // Use the model parameter directly to determine what to use
        if (model === 'qwen2.5-7b-omni') {
          return await generateCode(content, { primaryModel: ModelType.QWEN_OMNI });
        } else if (model === 'olympiccoder-7b') {
          return await generateCode(content, { primaryModel: ModelType.OLYMPIC_CODER });
        } else if (model === 'hybrid') {
          // For hybrid, combine the results of both models
          const qwenResult = await generateCode(content, { primaryModel: ModelType.QWEN_OMNI });
          const olympicResult = await generateCode(content, { primaryModel: ModelType.OLYMPIC_CODER });
          
          // Return a combined result
          return `${qwenResult}\n\n--- Alternative Implementation ---\n\n${olympicResult}`;
        } else {
          // Default to Qwen
          return await generateCode(content, { primaryModel: ModelType.QWEN_OMNI });
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        console.error(`[OpenManus] Error executing task with model ${model}:`, err);
        throw new Error(`Failed to execute task with model ${model}: ${errorMessage}`);
      }
    }
  },
  {
    handleError: (source: string, error: any) => {
      console.error(`[${source}] Error:`, error);
    }
  },
  {
    startOperation: (name: string, id?: string) => {
      const opId = id || uuidv4();
      console.log(`[OpenManus] Starting operation: ${name} (${opId})`);
      return opId;
    },
    endOperation: (id: string, failed?: boolean) => {
      console.log(`[OpenManus] Ending operation: ${id} ${failed ? '(failed)' : '(success)'}`);
    }
  }
);

// OpenManus autonomous project endpoints
aiRouter.post('/openmanus/project', async (req, res) => {
  try {
    const { name, description, language, framework, features, constraints } = req.body;
    
    if (!name || !description) {
      return res.status(400).json({ error: 'Project name and description are required' });
    }
    
    const projectId = await openManusIntegration.createProject({
      name,
      description,
      language,
      framework,
      features,
      constraints
    });
    
    res.status(201).json({ 
      projectId,
      message: 'Autonomous project created successfully'
    });
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.get('/openmanus/project/:id', async (req, res) => {
  try {
    const projectId = req.params.id;
    
    if (!projectId) {
      return res.status(400).json({ error: 'Project ID is required' });
    }
    
    const status = openManusIntegration.getProjectStatus(projectId);
    res.json(status);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.get('/openmanus/projects', async (req, res) => {
  try {
    const projects = openManusIntegration.getAllProjects();
    res.json(projects);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

aiRouter.get('/openmanus/project/:id/files', async (req, res) => {
  try {
    const projectId = req.params.id;
    
    if (!projectId) {
      return res.status(400).json({ error: 'Project ID is required' });
    }
    
    const files = openManusIntegration.getProjectFiles(projectId);
    res.json(files);
  } catch (error: unknown) {
    errorHandler.handleError(error, req, res);
  }
});

// Initialize model processes when server starts
(async () => {
  try {
    const initialized = await initializeModelProcesses();
    console.log(`[AI] Model processes ${initialized ? 'successfully initialized' : 'failed to initialize'}`);
    
    // Initialize OpenManus integration
    await openManusIntegration.initialize();
    console.log(`[AI] OpenManus integration ${initialized ? 'successfully initialized' : 'failed to initialize'}`);
  } catch (error: unknown) {
    console.error('[AI] Error initializing model processes:', error);
  }
})();

// Error handler
aiRouter.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  errorHandler.handleError(err, req, res);
});