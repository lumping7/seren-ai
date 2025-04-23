import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { setupAuth } from "./auth";
import { aiRouter } from "./ai";
import { securityRouter } from "./security";
import { healthRouter } from "./health";
import { isDatabaseAvailable } from "./db";
import { modelServices } from "./ai";
import { generateDirectResponse, ModelType } from './ai/direct-integration';

export async function registerRoutes(app: Express): Promise<Server> {
  // Setup authentication
  setupAuth(app);
  
  // Set up AI routes
  app.use("/api/ai", aiRouter);
  
  // Set up security routes
  app.use("/api/security", securityRouter);
  
  // Set up health monitoring routes
  app.use("/api/health", healthRouter);
  
  // Direct AI endpoint (virtual computer system)
  // This is a production-ready, completely self-contained AI system
  // that doesn't rely on external services or API calls
  
  app.post("/api/virtual-computer", async (req, res) => {
    try {
      const { prompt, model, operationId } = req.body;
      
      if (!prompt) {
        return res.status(400).json({ 
          error: "Missing prompt",
          message: "A prompt is required to generate a response" 
        });
      }
      
      // Map model string to ModelType enum
      let modelType: ModelType;
      if (model === 'qwen') {
        modelType = ModelType.QWEN_OMNI;
      } else if (model === 'olympic') {
        modelType = ModelType.OLYMPIC_CODER;
      } else {
        modelType = ModelType.HYBRID;
      }
      
      // Generate response using our direct integration
      const response = await generateDirectResponse(prompt, {
        model: modelType,
        operationId
      });
      
      // Return structured response
      res.json({
        model: model || 'hybrid',
        generated_text: response,
        metadata: {
          system: "virtual-computer",
          timestamp: new Date().toISOString(),
          model_version: modelType,
          operation_id: operationId
        }
      });
    } catch (error) {
      console.error("Error in virtual computer endpoint:", error);
      res.status(500).json({
        error: "Internal server error",
        message: "An error occurred processing your request"
      });
    }
  });
  
  // REST API endpoint for chat (used when WebSocket is unavailable)
  app.post("/api/chat", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      const { message } = req.body;
      
      if (!message || !message.content) {
        return res.status(400).json({ message: "Invalid message format" });
      }
      
      // Save user message to database
      const user = req.user as Express.User;
      const savedMessage = await storage.createMessage({
        conversationId: message.conversationId,
        role: 'user',
        content: message.content,
        model: message.model || 'hybrid',
        userId: user.id,
        metadata: message.metadata || {}
      });
      
      // Process via AI (simplified version of WebSocket handler logic)
      try {
        // Determine which AI model to use
        const modelType = message.model || 'hybrid';
        
        // Generate direct response
        const aiResponseText = await generateDirectResponse(message.content, {
          model: modelType === 'qwen' ? ModelType.QWEN_OMNI :
                 modelType === 'olympic' ? ModelType.OLYMPIC_CODER :
                 ModelType.HYBRID
        });
        
        // Create AI response message
        const aiResponse = await storage.createMessage({
          conversationId: message.conversationId,
          role: 'assistant',
          content: aiResponseText,
          model: modelType,
          userId: user.id,
          metadata: { 
            modelType,
            generatedVia: "http"
          }
        });
        
        // Return both messages
        res.status(200).json({ 
          userMessage: savedMessage,
          aiResponse: aiResponse
        });
      } catch (aiError: any) {
        console.error('AI processing error:', aiError);
        
        // Create error response
        const errorResponse = await storage.createMessage({
          conversationId: message.conversationId,
          role: 'assistant',
          content: `I'm sorry, but there was an error processing your message: ${aiError.message || "Unknown error"}`,
          model: message.model || 'hybrid',
          userId: user.id,
          metadata: { error: aiError.message || "Unknown error", fallback: true }
        });
        
        res.status(500).json({ 
          userMessage: savedMessage,
          aiResponse: errorResponse,
          error: "AI processing error"
        });
      }
    } catch (error: any) {
      console.error('Error in chat API:', error);
      res.status(500).json({ 
        message: "Failed to process chat message",
        error: error.message 
      });
    }
  });
  
  // AI Settings API
  app.get("/api/settings/:key", async (req, res) => {
    try {
      const key = req.params.key;
      const setting = await storage.getSetting(key);
      
      if (!setting) {
        return res.status(404).json({ message: "Setting not found" });
      }
      
      res.json(setting);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch setting" });
    }
  });
  
  app.put("/api/settings/:key", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      const key = req.params.key;
      const value = req.body.value;
      
      if (!value) {
        return res.status(400).json({ message: "Setting value is required" });
      }
      
      const user = req.user as Express.User;
      const setting = await storage.updateSetting(key, value, user.id);
      res.json(setting);
    } catch (error) {
      res.status(500).json({ message: "Failed to update setting" });
    }
  });

  // Create HTTP server
  const httpServer = createServer(app);
  
  // Define a flag to enable or disable WebSockets
  // In production VDS environment, WebSockets might not be available
  // so we'll use HTTP as our primary communication method
  const ENABLE_WEBSOCKETS = process.env.ENABLE_WEBSOCKETS === 'true' || 
                           process.env.NODE_ENV === 'development';
  
  let wss: WebSocketServer | null = null;
  
  if (ENABLE_WEBSOCKETS) {
    try {
      // Setup WebSocket server for real-time chat only if enabled
      console.log('Initializing WebSocket server for real-time chat');
      wss = new WebSocketServer({ server: httpServer, path: '/ws' });
    } catch (error) {
      console.error('Failed to initialize WebSocket server:', error);
      console.log('Falling back to HTTP-only mode');
      wss = null;
    }
  } else {
    console.log('WebSockets are disabled. Using HTTP-only mode');
  }
  
  // Only set up WebSocket handlers if WebSockets are enabled and initialized successfully
  if (wss) {
    wss.on('connection', (ws) => {
      console.log('Client connected to WebSocket');
      
      ws.on('message', async (message) => {
        try {
          console.log('WebSocket message received:', message.toString());
          const data = JSON.parse(message.toString());
          
          if (data.type === 'chat-message' && data.message) {
            console.log('Processing chat message:', data.message.conversationId);
            
            // Store message in database
            const savedMessage = await storage.createMessage({
              conversationId: data.message.conversationId,
              role: data.message.role,
              content: data.message.content,
              model: data.message.model,
              userId: data.message.userId,
              metadata: data.message.metadata || {}
            });
            
            // Broadcast message to all connected clients
            if (wss) {
              wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                  client.send(JSON.stringify({
                    type: 'new-message',
                    message: savedMessage
                  }));
                }
              });
            }
            
            // If it's a user message, generate AI response
            if (data.message.role === 'user') {
              // Process message with the appropriate AI model
              try {
                // Determine which AI model to use
                const modelType = data.message.model || 'hybrid';
                
                // Direct AI generation using our local AI router
                let aiResponseText = '';
                
                try {
                  // Generate direct response
                  aiResponseText = await generateDirectResponse(data.message.content, {
                    model: modelType === 'qwen' ? ModelType.QWEN_OMNI :
                           modelType === 'olympic' ? ModelType.OLYMPIC_CODER :
                           ModelType.HYBRID
                  });
                } catch (aiError: any) {
                  console.error("[OpenManus] Error generating response:", aiError);
                  
                  // Provide a more detailed error message
                  aiResponseText = `I'm experiencing a technical issue with the OpenManus integration. Error details: ${aiError.message || "Unknown error"}`;
                }
                
                // Create a response object
                const aiResponse = {
                  generated_text: aiResponseText,
                  metadata: {
                    model_version: modelType,
                    processing_time: 0,
                    tokens_used: 0
                  }
                };
                
                // If AI response was successful, save it and broadcast it
                if (aiResponse && aiResponse.generated_text) {
                  const savedAiResponse = await storage.createMessage({
                    conversationId: data.message.conversationId,
                    role: 'assistant',
                    content: aiResponse.generated_text,
                    model: modelType,
                    userId: data.message.userId,
                    metadata: {
                      processingTime: aiResponse.metadata?.processing_time || 0,
                      tokensUsed: aiResponse.metadata?.tokens_used || 0,
                      modelVersion: aiResponse.metadata?.model_version || modelType
                    }
                  });
                  
                  // Broadcast AI response
                  if (wss) {
                    wss.clients.forEach((client) => {
                      if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify({
                          type: 'new-message',
                          message: savedAiResponse
                        }));
                      }
                    });
                  }
                } else {
                  // If AI generation failed, use a fallback response
                  const fallbackResponse = await storage.createMessage({
                    conversationId: data.message.conversationId,
                    role: 'assistant',
                    content: "I apologize for the technical issue. The OpenManus system is in offline mode with locally hosted models as requested. Please try your query again with more specific instructions for the AI model.",
                    model: modelType,
                    userId: data.message.userId,
                    metadata: { error: "AI generation failed", fallback: true }
                  });
                  
                  // Broadcast fallback response
                  if (wss) {
                    wss.clients.forEach((client) => {
                      if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify({
                          type: 'new-message',
                          message: fallbackResponse
                        }));
                      }
                    });
                  }
                }
              } catch (error: any) {
                console.error('AI processing error:', error);
                
                // Send error response to client
                const errorResponse = await storage.createMessage({
                  conversationId: data.message.conversationId,
                  role: 'assistant',
                  content: "I'm sorry, but there was an error processing your message. Please try again later.",
                  model: data.message.model || 'hybrid',
                  userId: data.message.userId,
                  metadata: { error: error.message || "Unknown error", fallback: true }
                });
                
                if (wss) {
                  wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                      client.send(JSON.stringify({
                        type: 'new-message',
                        message: errorResponse
                      }));
                    }
                  });
                }
              }
            }
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      });
      
      ws.on('close', () => {
        console.log('Client disconnected from WebSocket');
      });
    });
  }

  return httpServer;
}