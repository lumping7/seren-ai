/**
 * System API Endpoints
 * 
 * Provides REST endpoints for system status, resource management,
 * and configuration.
 */

import { Request, Response } from 'express';
import { resourceManager } from './resource-manager';

/**
 * System API handler functions
 */
export const systemApi = {
  /**
   * Get system resource information
   * GET /api/ai/system/resources
   */
  getResources: async (req: Request, res: Response) => {
    try {
      const resources = resourceManager.getSystemResources();
      const profile = resourceManager.getProfile();
      const limits = resourceManager.getLimits();
      
      return res.status(200).json({
        success: true,
        resources,
        profile,
        limits
      });
    } catch (error) {
      console.error("Error getting system resources:", error);
      return res.status(500).json({
        success: false,
        error: "Failed to retrieve system resources",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  },
  
  /**
   * Get system status and capabilities
   * GET /api/ai/status
   */
  getStatus: async (req: Request, res: Response) => {
    try {
      // Get basic system information
      const resources = resourceManager.getSystemResources();
      const profile = resourceManager.getProfile();
      
      // Determine available capabilities based on profile
      const capabilities = {
        maxContextSize: resourceManager.getMaxContextSize(),
        maxTokens: resourceManager.getMaxTokens(),
        parallelProcessing: resourceManager.isParallelInferenceEnabled(),
        quantization: resourceManager.isQuantizationEnabled(),
        availableModels: ['llama3', 'gemma3', 'hybrid'],
        availableReasoningModes: [
          'logical_analysis',
          'causal_reasoning',
          'temporal_reasoning',
          'counterfactual_reasoning',
          'mathematical_reasoning',
          'ethical_reasoning',
          'scientific_reasoning'
        ],
        availableConversationModes: [
          'collaborative',
          'debate',
          'critical',
          'brainstorming'
        ]
      };
      
      // Get system health metrics
      const health = {
        status: 'operational',
        uptime: process.uptime(),
        memoryUsagePercent: Math.round(100 - (resources.availableMemory / resources.totalMemory * 100)),
        cpuUsagePercent: Math.round(resources.cpuUsage),
        activeRequests: resources.activeRequests
      };
      
      return res.status(200).json({
        success: true,
        version: '1.0.0',
        profile,
        capabilities,
        health,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error("Error getting system status:", error);
      return res.status(500).json({
        success: false,
        error: "Failed to retrieve system status",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  }
};