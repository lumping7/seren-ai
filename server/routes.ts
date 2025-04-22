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
        const data = JSON.parse(message.toString());
        
        if (data.type === 'chat-message') {
          // Store message in database
          const savedMessage = await storage.createMessage({
            conversationId: data.conversationId,
            role: data.role,
            content: data.content,
            model: data.model,
            userId: data.userId,
            metadata: data.metadata || {}
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
          if (data.role === 'user') {
            // In a real implementation, this would call the AI engine
            // For now, we'll simulate an AI response
            setTimeout(async () => {
              const aiResponse = await storage.createMessage({
                conversationId: data.conversationId,
                role: 'assistant',
                content: `This is a simulated response to: "${data.content}"`,
                model: 'hybrid',
                userId: data.userId,
                metadata: { processingTime: 0.5 }
              });
              
              // Broadcast AI response
              wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                  client.send(JSON.stringify({
                    type: 'new-message',
                    message: aiResponse
                  }));
                }
              });
            }, 1000);
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
