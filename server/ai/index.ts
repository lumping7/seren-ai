import express from 'express';
import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';
import { memoryHandler } from './memory';
import { reasoningHandler } from './reasoning';
import { executeCodeHandler } from './execution';

// Create AI router
const aiRouter = express.Router();

// Basic health check endpoint
aiRouter.get('/health', (req, res) => {
  res.json({ status: 'ok', models: ['llama3', 'gemma3', 'hybrid'] });
});

// AI Models routes
aiRouter.post('/llama', llamaHandler);
aiRouter.post('/gemma', gemmaHandler);

// This endpoint decides which model to use based on the input
aiRouter.post('/generate', async (req, res) => {
  try {
    const { prompt, model = 'hybrid', options = {} } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    let response;
    
    switch (model) {
      case 'llama3':
        response = await llamaHandler(req, res);
        break;
      case 'gemma3':
        response = await gemmaHandler(req, res);
        break;
      case 'hybrid':
        // For hybrid, we'd implement more sophisticated model switching
        // For now, we'll simulate it
        if (prompt.includes('reasoning') || prompt.includes('logic')) {
          response = await llamaHandler(req, res);
        } else {
          response = await gemmaHandler(req, res);
        }
        break;
      default:
        return res.status(400).json({ error: 'Invalid model specified' });
    }
    
    return response;
  } catch (error) {
    console.error('Error generating AI response:', error);
    return res.status(500).json({ error: 'Failed to generate AI response' });
  }
});

// Memory system routes
aiRouter.get('/memory', memoryHandler.getMemories);
aiRouter.post('/memory', memoryHandler.storeMemory);
aiRouter.get('/memory/search', memoryHandler.searchMemories);

// Reasoning system routes
aiRouter.post('/reason', reasoningHandler.performReasoning);

// Code execution routes
aiRouter.post('/execute', executeCodeHandler);

export { aiRouter };
