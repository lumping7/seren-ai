/**
 * AI API Router
 * 
 * Handles all AI-related endpoints, including model inference,
 * hybrid engine, and reasoning operations.
 */

import express from 'express';
import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';
import { hybridHandler } from './hybrid';
import { errorHandler } from './error-handler';
import { resourceManager } from './resource-manager';
import { performanceMonitor } from './performance-monitor';

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
    performance: performanceMonitor.getSummaryStats(),
    circuits: errorHandler.getCircuitBreakerStatus(),
    models: {
      llama3: { available: true, versions: ['3.1.0-8b', '3.1.0-70b'] },
      gemma3: { available: true, versions: ['3.1.0-7b', '3.1.0-27b'] },
      hybrid: { available: true, modes: ['collaborative', 'specialized', 'competitive'] }
    }
  };
  
  res.json(status);
});

// Model endpoints
aiRouter.post('/llama', llamaHandler);
aiRouter.post('/gemma', gemmaHandler);
aiRouter.post('/hybrid', hybridHandler);

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

// Error handler
aiRouter.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  errorHandler.handleError(err, req, res);
});