import express from 'express';
import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';
import { hybridHandler } from './hybrid';
import { memoryHandler } from './memory';
import { reasoningHandler } from './reasoning';
import { executeCodeHandler } from './execution';
import { knowledgeApi } from './knowledge-api';
import { conversationManager } from './conversation';
import { resourceManager } from './resource-manager';
import { systemApi } from './system-api';

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
    const { prompt, model = 'hybrid', options = {}, enhanceWithKnowledge = false } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    // Track resource usage for adaptive scaling
    resourceManager.startRequest();
    
    // Check if we can process this request based on current load
    if (!resourceManager.canProcessRequest()) {
      return res.status(429).json({
        error: 'System currently at capacity',
        retry_after: 5, // Retry after 5 seconds
        current_profile: resourceManager.getProfile()
      });
    }
    
    // Optionally enhance prompt with relevant knowledge
    let enhancedPrompt = prompt;
    if (enhanceWithKnowledge) {
      try {
        const result = await knowledgeApi.enhancePromptWithKnowledge(
          prompt,
          options.domains
        );
        enhancedPrompt = result;
      } catch (knowledgeError) {
        console.error('Error enhancing prompt with knowledge:', knowledgeError);
        // Continue with original prompt on error
      }
    }
    
    // Create a modified request with the enhanced prompt if needed
    const modifiedReq = enhanceWithKnowledge 
      ? { ...req, body: { ...req.body, prompt: enhancedPrompt } }
      : req;
    
    try {
      let result;
      switch (model) {
        case 'llama3':
          result = await llamaHandler(modifiedReq, res);
          break;
        case 'gemma3':
          result = await gemmaHandler(modifiedReq, res);
          break;
        case 'hybrid':
          result = await hybridHandler(modifiedReq, res);
          break;
        default:
          return res.status(400).json({ 
            error: 'Invalid model specified',
            valid_models: ['llama3', 'gemma3', 'hybrid'] 
          });
      }
      
      // End resource tracking
      resourceManager.endRequest();
      
      return result;
    } catch (modelError: any) {
      resourceManager.endRequest();
      throw modelError;
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
aiRouter.post('/reason', reasoningHandler);

// Code execution routes
aiRouter.post('/execute', executeCodeHandler);

// Knowledge system routes
aiRouter.post('/knowledge', knowledgeApi.addKnowledge);
aiRouter.post('/knowledge/user', knowledgeApi.addUserKnowledge);
aiRouter.post('/knowledge/extract', knowledgeApi.extractKnowledge);
aiRouter.get('/knowledge/retrieve', knowledgeApi.retrieveKnowledge);
aiRouter.get('/knowledge/similar', knowledgeApi.findSimilarKnowledge);
aiRouter.post('/knowledge/domains', knowledgeApi.createDomain);
aiRouter.get('/knowledge/domains', knowledgeApi.getDomains);
aiRouter.get('/knowledge/domains/:domain', knowledgeApi.getKnowledgeByDomain);
aiRouter.get('/knowledge/:id/related', knowledgeApi.getRelatedKnowledge);
aiRouter.get('/knowledge/graph', knowledgeApi.createKnowledgeGraph);
aiRouter.post('/knowledge/enhance-prompt', knowledgeApi.enhancePrompt);

// Conversation system (multi-turn AI-to-AI discussion)
aiRouter.post('/conversation', async (req, res) => {
  try {
    const { topic, userPrompt, mode } = req.body;
    
    if (!topic || !userPrompt) {
      return res.status(400).json({ error: 'Topic and userPrompt are required' });
    }
    
    const userId = req.isAuthenticated() ? (req.user as any).id : undefined;
    
    // Start a conversation between models
    const conversationId = await conversationManager.startConversation(
      topic,
      userPrompt,
      mode || 'collaborative',
      userId
    );
    
    return res.json({
      success: true,
      conversationId,
      message: 'Conversation started'
    });
  } catch (error: any) {
    console.error('Error starting conversation:', error);
    return res.status(500).json({ error: error.message || 'Unknown error' });
  }
});

aiRouter.get('/conversation/:id', async (req, res) => {
  try {
    const conversationId = req.params.id;
    const conversation = conversationManager.getConversation(conversationId);
    
    if (!conversation) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    return res.json({
      success: true,
      conversation,
      isComplete: conversationManager.isConversationComplete(conversationId)
    });
  } catch (error: any) {
    console.error('Error getting conversation:', error);
    return res.status(500).json({ error: error.message || 'Unknown error' });
  }
});

// System information and management
aiRouter.get('/system/resources', systemApi.getResources);

// System status and capabilities
aiRouter.get('/status', systemApi.getStatus);

export { aiRouter };
