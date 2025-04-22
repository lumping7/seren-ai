/**
 * Knowledge API Endpoints
 * 
 * Provides REST endpoints for interacting with the knowledge system:
 * - Adding new knowledge
 * - Retrieving relevant knowledge
 * - Managing knowledge domains
 * - Visualizing knowledge graphs
 */

import { Request, Response } from 'express';
import { knowledgeManager } from './knowledge-manager';
import { z } from 'zod';
import { insertKnowledgeSchema, KnowledgeSource } from '@shared/schema';

// Define validation schemas
const addKnowledgeSchema = insertKnowledgeSchema.extend({
  domains: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
  importanceScore: z.number().min(0).max(1).optional()
});

const retrieveKnowledgeSchema = z.object({
  query: z.string().min(1),
  domains: z.array(z.string()).optional(),
  limit: z.number().min(1).max(50).optional(),
  minImportance: z.number().min(0).max(1).optional()
});

const createDomainSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional()
});

const knowledgeGraphSchema = z.object({
  centralDomain: z.string().optional(),
  depth: z.number().min(1).max(5).optional(),
  maxNodes: z.number().min(5).max(200).optional()
});

/**
 * Knowledge API handler functions
 */
export const knowledgeApi = {
  /**
   * Add new knowledge from external sources or user input
   * POST /api/knowledge
   */
  addKnowledge: async (req: Request, res: Response) => {
    try {
      const validationResult = addKnowledgeSchema.safeParse(req.body);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { content, source, domains, tags, importanceScore, metadata } = validationResult.data;
      
      // Get user ID if authenticated
      const userId = req.isAuthenticated() ? (req.user as any).id : undefined;
      
      // Add to knowledge base
      const entry = await knowledgeManager.addKnowledge(
        content,
        source,
        domains || [],
        userId,
        metadata || {},
        importanceScore
      );
      
      return res.status(201).json({
        success: true,
        entry,
        message: 'Knowledge added successfully'
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error adding knowledge:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to add knowledge',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Add knowledge directly from user
   * POST /api/knowledge/user
   */
  addUserKnowledge: async (req: Request, res: Response) => {
    try {
      // Must be authenticated to add user knowledge
      if (!req.isAuthenticated()) {
        return res.status(401).json({
          error: 'Authentication required to add user knowledge'
        });
      }
      
      const validationResult = z.object({
        content: z.string().min(10),
        domains: z.array(z.string()).optional(),
        importanceScore: z.number().min(0).max(1).optional(),
        metadata: z.record(z.any()).optional()
      }).safeParse(req.body);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { content, domains, importanceScore, metadata } = validationResult.data;
      const userId = (req.user as any).id;
      
      // Add to knowledge base
      const entry = await knowledgeManager.addUserKnowledge(
        content,
        domains || [],
        userId,
        metadata || {},
        importanceScore
      );
      
      return res.status(201).json({
        success: true,
        entry,
        message: 'User knowledge added successfully'
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error adding user knowledge:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to add user knowledge',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Extract knowledge from a conversation
   * POST /api/knowledge/extract
   */
  extractKnowledge: async (req: Request, res: Response) => {
    try {
      const validationResult = z.object({
        conversationId: z.string().min(1)
      }).safeParse(req.body);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { conversationId } = validationResult.data;
      
      // Get user ID if authenticated
      const userId = req.isAuthenticated() ? (req.user as any).id : undefined;
      
      // Extract knowledge from conversation
      const entries = await knowledgeManager.extractKnowledgeFromConversation(
        conversationId,
        userId
      );
      
      return res.json({
        success: true,
        entriesCount: entries.length,
        entries,
        message: `Extracted ${entries.length} knowledge entries from conversation`
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error extracting knowledge:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to extract knowledge',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Retrieve relevant knowledge for a query
   * GET /api/knowledge/retrieve
   */
  retrieveKnowledge: async (req: Request, res: Response) => {
    try {
      const validationResult = retrieveKnowledgeSchema.safeParse(req.query);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { query, domains, limit, minImportance } = validationResult.data;
      
      // Retrieve knowledge
      const entries = await knowledgeManager.retrieveRelevantKnowledge(
        query,
        domains,
        limit,
        minImportance
      );
      
      return res.json({
        success: true,
        query,
        entriesCount: entries.length,
        entries
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error retrieving knowledge:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to retrieve knowledge',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Find similar knowledge entries
   * GET /api/knowledge/similar
   */
  findSimilarKnowledge: async (req: Request, res: Response) => {
    try {
      const validationResult = z.object({
        content: z.string().min(10),
        threshold: z.number().min(0).max(1).optional(),
        limit: z.number().min(1).max(20).optional()
      }).safeParse(req.query);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { content, threshold, limit } = validationResult.data;
      
      // Find similar knowledge
      const entries = await knowledgeManager.findSimilarKnowledge(
        content,
        threshold,
        limit
      );
      
      return res.json({
        success: true,
        entriesCount: entries.length,
        entries
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error finding similar knowledge:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to find similar knowledge',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Create a new knowledge domain
   * POST /api/knowledge/domains
   */
  createDomain: async (req: Request, res: Response) => {
    try {
      const validationResult = createDomainSchema.safeParse(req.body);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { name, description } = validationResult.data;
      
      // Get user ID if authenticated
      const userId = req.isAuthenticated() ? (req.user as any).id : undefined;
      
      // Create domain
      const domain = await knowledgeManager.createDomain(
        name,
        description || `Domain for ${name}`,
        userId
      );
      
      return res.status(201).json({
        success: true,
        domain,
        message: 'Domain created successfully'
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error creating domain:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to create domain',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Get all knowledge domains
   * GET /api/knowledge/domains
   */
  getDomains: async (req: Request, res: Response) => {
    try {
      const domains = await knowledgeManager.getDomains();
      
      return res.json({
        success: true,
        domainsCount: domains.length,
        domains
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error getting domains:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to get domains',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Get knowledge entries for a specific domain
   * GET /api/knowledge/domains/:domain
   */
  getKnowledgeByDomain: async (req: Request, res: Response) => {
    try {
      const domain = req.params.domain;
      
      if (!domain) {
        return res.status(400).json({
          error: 'Domain name is required'
        });
      }
      
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 100;
      
      // Get knowledge by domain
      const entries = await knowledgeManager.getKnowledgeByDomain(domain, limit);
      
      return res.json({
        success: true,
        domain,
        entriesCount: entries.length,
        entries
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error getting knowledge by domain:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to get knowledge by domain',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Get related knowledge entries
   * GET /api/knowledge/:id/related
   */
  getRelatedKnowledge: async (req: Request, res: Response) => {
    try {
      const knowledgeId = parseInt(req.params.id);
      
      if (isNaN(knowledgeId)) {
        return res.status(400).json({
          error: 'Valid knowledge ID is required'
        });
      }
      
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 5;
      
      // Get related knowledge
      const entries = await knowledgeManager.getRelatedKnowledge(knowledgeId, limit);
      
      return res.json({
        success: true,
        knowledgeId,
        entriesCount: entries.length,
        entries
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error getting related knowledge:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to get related knowledge',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Create a knowledge graph for visualization
   * GET /api/knowledge/graph
   */
  createKnowledgeGraph: async (req: Request, res: Response) => {
    try {
      const validationResult = knowledgeGraphSchema.safeParse(req.query);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { centralDomain, depth, maxNodes } = validationResult.data;
      
      // Create knowledge graph
      const graph = await knowledgeManager.createKnowledgeGraph(
        centralDomain,
        depth,
        maxNodes
      );
      
      return res.json({
        success: true,
        graph
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error creating knowledge graph:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to create knowledge graph',
        details: error.message || 'Unknown server error'
      });
    }
  },
  
  /**
   * Enhance a prompt with relevant knowledge
   * POST /api/knowledge/enhance-prompt
   */
  enhancePrompt: async (req: Request, res: Response) => {
    try {
      const validationResult = z.object({
        prompt: z.string().min(1),
        domains: z.array(z.string()).optional()
      }).safeParse(req.body);
      
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Validation failed',
          details: validationResult.error.errors
        });
      }
      
      const { prompt, domains } = validationResult.data;
      
      // Enhance prompt with knowledge
      const enhancedPrompt = await knowledgeManager.enhancePromptWithKnowledge(
        prompt,
        domains
      );
      
      return res.json({
        success: true,
        originalPrompt: prompt,
        enhancedPrompt,
        wasEnhanced: enhancedPrompt !== prompt
      });
    } catch (error: any) {
      console.error('[KnowledgeAPI] Error enhancing prompt:', error.message || 'Unknown error');
      return res.status(500).json({
        error: 'Failed to enhance prompt',
        details: error.message || 'Unknown server error'
      });
    }
  }
};