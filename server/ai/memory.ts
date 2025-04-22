import { Request, Response } from 'express';
import { storage } from '../storage';

/**
 * Memory System Handler
 * Provides functions for storing and retrieving AI memories
 * In a real implementation, this would interface with vector databases
 */
export const memoryHandler = {
  /**
   * Get memories from the system
   */
  getMemories: async (req: Request, res: Response) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 10;
      const userId = req.query.userId ? parseInt(req.query.userId as string) : undefined;
      
      let memories;
      if (userId) {
        memories = await storage.getMemoriesByUser(userId, limit);
      } else {
        memories = await storage.getMemories(limit);
      }
      
      return res.json(memories);
    } catch (error) {
      console.error('Error fetching memories:', error);
      return res.status(500).json({ error: 'Failed to fetch memories' });
    }
  },
  
  /**
   * Store a new memory
   */
  storeMemory: async (req: Request, res: Response) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ error: 'Authentication required' });
      }
      
      const { title, content, type, metadata } = req.body;
      
      if (!title || !content || !type) {
        return res.status(400).json({ error: 'Title, content, and type are required' });
      }
      
      const memory = await storage.createMemory({
        title,
        content,
        type,
        userId: req.user?.id,
        metadata
      });
      
      return res.status(201).json(memory);
    } catch (error) {
      console.error('Error storing memory:', error);
      return res.status(500).json({ error: 'Failed to store memory' });
    }
  },
  
  /**
   * Search memories by content
   * In a real implementation, this would use vector similarity search
   */
  searchMemories: async (req: Request, res: Response) => {
    try {
      const query = req.query.q as string;
      
      if (!query) {
        return res.status(400).json({ error: 'Search query is required' });
      }
      
      // Simulate vector search with simple text search
      const allMemories = await storage.getMemories();
      const results = allMemories.filter(memory => 
        memory.title.toLowerCase().includes(query.toLowerCase()) || 
        memory.content.toLowerCase().includes(query.toLowerCase())
      );
      
      return res.json(results);
    } catch (error) {
      console.error('Error searching memories:', error);
      return res.status(500).json({ error: 'Failed to search memories' });
    }
  }
};
