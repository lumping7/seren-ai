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
  handleError(error: Error | EnhancedError | unknown, req?: Request, res?: Response): void {
    // Ensure we have an Error object
    const normalizedError: EnhancedError = this.normalizeError(error);
    
    // Log the error
    console.error('[ErrorHandler]', {
      message: normalizedError.message,
      category: normalizedError.category || ErrorCategory.UNKNOWN,
      details: normalizedError.details,
      stack: normalizedError.stack
    });
    
    // If response object is provided, send error response
    if (req && res) {
      const statusCode = normalizedError.statusCode || 500;
      
      res.status(statusCode).json({
        error: {
          message: normalizedError.message,
          category: normalizedError.category || ErrorCategory.UNKNOWN,
          details: normalizedError.details,
          request_id: req.headers['x-request-id'] || 'unknown'
        }
      });
    }
  }
  
  /**
   * Normalize any error type to an EnhancedError
   */
  normalizeError(error: unknown): EnhancedError {
    if (error instanceof Error) {
      return error as EnhancedError;
    } else if (typeof error === 'string') {
      return this.createError(error);
    } else if (error && typeof error === 'object') {
      const errObj = error as any;
      const message = errObj.message || 'Unknown error';
      return this.createError(
        message, 
        errObj.category || ErrorCategory.UNKNOWN,
        errObj,
        errObj.statusCode || 500
      );
    } else {
      return this.createError('Unknown error');
    }
  }
}

// Singleton instance
export const errorHandler = new ErrorHandler();
errorHandler.initialize();