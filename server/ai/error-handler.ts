/**
 * Error Handler for AI Module
 * 
 * This module provides standardized error handling for the AI system,
 * ensuring consistent error responses and logging.
 */

import { Request, Response } from 'express';

export enum ErrorCategory {
  VALIDATION = 'validation',
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  MODEL = 'model',
  SYSTEM = 'system',
  NETWORK = 'network',
  RESOURCE = 'resource',
  TIMEOUT = 'timeout',
  UNKNOWN = 'unknown'
}

export interface EnhancedError extends Error {
  category?: ErrorCategory;
  details?: any;
  statusCode?: number;
}

class ErrorHandler {
  private initialized: boolean = false;

  /**
   * Initialize the error handler
   */
  initialize(): void {
    if (this.initialized) {
      return;
    }

    this.initialized = true;
    console.log('[ErrorHandler] Initialized');
  }

  /**
   * Create an enhanced error with additional metadata
   */
  createError(
    message: string, 
    category: ErrorCategory = ErrorCategory.UNKNOWN, 
    details?: any, 
    statusCode: number = 500
  ): EnhancedError {
    const error = new Error(message) as EnhancedError;
    error.category = category;
    error.details = details;
    error.statusCode = statusCode;
    return error;
  }

  /**
   * Handle an error and send an appropriate response
   */
  handleError(error: Error | EnhancedError, req?: Request, res?: Response): void {
    const enhancedError = error as EnhancedError;
    
    // Log the error
    console.error('[ErrorHandler]', {
      message: error.message,
      category: enhancedError.category || ErrorCategory.UNKNOWN,
      details: enhancedError.details,
      stack: error.stack
    });
    
    // If response object is provided, send error response
    if (req && res) {
      const statusCode = enhancedError.statusCode || 500;
      
      res.status(statusCode).json({
        error: {
          message: error.message,
          category: enhancedError.category || ErrorCategory.UNKNOWN,
          details: enhancedError.details,
          request_id: req.headers['x-request-id'] || 'unknown'
        }
      });
    }
  }
}

// Singleton instance
export const errorHandler = new ErrorHandler();
errorHandler.initialize();