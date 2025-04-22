import express from 'express';
import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';
import { hybridHandler } from './hybrid';
import { memoryHandler } from './memory';
import { reasoningHandler } from './reasoning';
import { executeCodeHandler } from './execution';

// Create AI router
const aiRouter = express.Router();

// Basic health check endpoint
aiRouter.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    models: ['llama3', 'gemma3', 'hybrid'],
    version: '1.0.0'
  });
});

// AI Models routes
aiRouter.post('/llama', llamaHandler);
aiRouter.post('/gemma', gemmaHandler);
aiRouter.post('/hybrid', hybridHandler);

// This endpoint decides which model to use based on the input
aiRouter.post('/generate', async (req, res) => {
  try {
    const { prompt, model = 'hybrid', options = {} } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    switch (model) {
      case 'llama3':
        return await llamaHandler(req, res);
      case 'gemma3':
        return await gemmaHandler(req, res);
      case 'hybrid':
        return await hybridHandler(req, res);
      default:
        return res.status(400).json({ 
          error: 'Invalid model specified',
          valid_models: ['llama3', 'gemma3', 'hybrid'] 
        });
    }
  } catch (error: any) {
    console.error('Error generating AI response:', error);
    return res.status(500).json({ 
      error: 'Failed to generate AI response',
      details: error.message || 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
});

// Memory system routes
aiRouter.get('/memory', memoryHandler.getMemories);
aiRouter.post('/memory', memoryHandler.storeMemory);
aiRouter.get('/memory/search', memoryHandler.searchMemories);

// Reasoning system routes
aiRouter.post('/reason', reasoningHandler.performReasoning);
aiRouter.get('/reason/operations', reasoningHandler.getOperationSets);
aiRouter.get('/reason/history', reasoningHandler.getReasoningHistory);

// Code execution routes
aiRouter.post('/execute', executeCodeHandler);

// System status and capabilities
aiRouter.get('/status', (req, res) => {
  res.json({
    status: 'operational',
    models: [
      {
        id: 'llama3',
        version: '3.1.0',
        status: 'active',
        capabilities: ['logical reasoning', 'architectural design', 'code generation', 'technical documentation'],
        context_length: 8192
      },
      {
        id: 'gemma3',
        version: '3.0.2',
        status: 'active',
        capabilities: ['creative writing', 'summarization', 'UI/UX suggestions', 'implementation details'],
        context_length: 8192
      },
      {
        id: 'hybrid',
        version: '1.0.0',
        status: 'active',
        capabilities: ['collaborative design', 'full-stack implementation', 'seamless architecture-to-code flow'],
        context_length: 8192,
        components: ['llama3', 'gemma3', 'reasoning']
      }
    ],
    capabilities: {
      reasoning: true,
      memory: true,
      code_execution: true,
      neuro_symbolic: true
    },
    system_load: Math.random() * 0.5 + 0.3, // Simulated load between 30% and 80%
    last_updated: new Date().toISOString()
  });
});

export { aiRouter };
