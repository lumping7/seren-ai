import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { setupAuth } from "./auth";
import { aiRouter } from "./ai";
import { securityRouter } from "./security";

export async function registerRoutes(app: Express): Promise<Server> {
  // Setup authentication
  setupAuth(app);
  
  // Set up AI routes
  app.use("/api/ai", aiRouter);
  
  // Set up security routes
  app.use("/api/security", securityRouter);
  
  // AI Memory API
  app.get("/api/memories", async (req, res) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : undefined;
      const memories = await storage.getMemories(limit);
      res.json(memories);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch memories" });
    }
  });
  
  app.get("/api/memories/user/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const limit = req.query.limit ? parseInt(req.query.limit as string) : undefined;
      const memories = await storage.getMemoriesByUser(userId, limit);
      res.json(memories);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch user memories" });
    }
  });
  
  app.post("/api/memories", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      const memory = await storage.createMemory({
        ...req.body,
        userId: req.user?.id
      });
      
      res.status(201).json(memory);
    } catch (error) {
      res.status(500).json({ message: "Failed to create memory" });
    }
  });
  
  // AI Messages API
  app.get("/api/messages/:conversationId", async (req, res) => {
    try {
      const conversationId = req.params.conversationId;
      const messages = await storage.getMessages(conversationId);
      res.json(messages);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch messages" });
    }
  });
  
  app.post("/api/messages", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      const message = await storage.createMessage({
        ...req.body,
        userId: req.user?.id
      });
      
      res.status(201).json(message);
    } catch (error) {
      res.status(500).json({ message: "Failed to create message" });
    }
  });
  
  // REST API endpoint for chat (fallback when WebSocket is unavailable)
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
      const savedMessage = await storage.createMessage({
        conversationId: message.conversationId,
        role: 'user',
        content: message.content,
        model: message.model || 'hybrid',
        userId: req.user.id,
        metadata: message.metadata || {}
      });
      
      // Process via AI (simplified version of WebSocket handler logic)
      try {
        // Create fallback response (normally would call AI model)
        const aiResponse = await storage.createMessage({
          conversationId: message.conversationId,
          role: 'assistant',
          content: `I received your message: "${message.content}". This is a fallback response since we're using the REST API instead of WebSockets.`,
          model: message.model || 'hybrid',
          userId: req.user.id,
          metadata: { 
            fallback: true,
            note: "Using REST API fallback instead of WebSockets"
          }
        });
        
        // Return both messages
        res.status(200).json({ 
          userMessage: savedMessage,
          aiResponse: aiResponse
        });
      } catch (aiError) {
        console.error('AI processing error:', aiError);
        res.status(500).json({ message: "Error processing message with AI" });
      }
    } catch (error) {
      console.error('Error in chat API:', error);
      res.status(500).json({ message: "Failed to process chat message" });
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
      
      const setting = await storage.updateSetting(key, value, req.user?.id);
      res.json(setting);
    } catch (error) {
      res.status(500).json({ message: "Failed to update setting" });
    }
  });

  const httpServer = createServer(app);
  
  // Setup WebSocket server for real-time chat
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  
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
          wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({
                type: 'new-message',
                message: savedMessage
              }));
            }
          });
          
          // If it's a user message, generate AI response
          if (data.message.role === 'user') {
            // Process message with the appropriate AI model
            try {
              // Determine which AI model to use
              const modelType = data.message.model || 'hybrid';
              
              // Direct AI generation using our local AI router
              let aiResponseText = '';
              
              try {
                // Import the offline direct integration module
                // This implements the OpenManus structure with locally hosted models
                const { generateDirectResponse, ModelType } = await import('./ai/direct-integration');
                
                // Map the modelType string to our ModelType enum
                let directModel: ModelType;
                switch (modelType) {
                  case 'qwen':
                    directModel = ModelType.QWEN_OMNI;
                    break;
                  case 'olympic':
                    directModel = ModelType.OLYMPIC_CODER;
                    break;
                  case 'hybrid':
                  default:
                    directModel = ModelType.HYBRID;
                    break;
                }
                
                // Log full request details for debugging
                console.log(`[OpenManus] Processing request with model: ${directModel}`);
                console.log(`[OpenManus] Request content: ${data.message.content.substring(0, 100)}${data.message.content.length > 100 ? '...' : ''}`);
                console.log(`[OpenManus] Conversation ID: ${data.message.conversationId}`);
                
                // Generate text using our bleeding-edge framework
                // This is completely offline and self-contained
                const response = await generateDirectResponse(
                  data.message.content,
                  directModel
                );
                
                console.log(`[OpenManus] Generated response of ${response.length} characters`);
                
                // Set the response text
                aiResponseText = response;
              } catch (aiError) {
                console.error("[OpenManus] Error generating response:", aiError);
                
                // Provide a more detailed error message
                aiResponseText = `I'm experiencing a technical issue with the OpenManus integration. Error details: ${aiError.message}`;
              }
              
              // Create a response object similar to what the API would return
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
                wss.clients.forEach((client) => {
                  if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({
                      type: 'new-message',
                      message: savedAiResponse
                    }));
                  }
                });
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
                wss.clients.forEach((client) => {
                  if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({
                      type: 'new-message',
                      message: fallbackResponse
                    }));
                  }
                });
              }
            } catch (error) {
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
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    });
    
    ws.on('close', () => {
      console.log('Client disconnected from WebSocket');
    });
  });

  return httpServer;
}
