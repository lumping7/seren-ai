/**
 * Error Handler
 * 
 * Comprehensive error handling system with centralized logging, error categorization,
 * circuit breakers, and consistent error response formatting.
 */

import fs from 'fs';
import path from 'path';
import { Request, Response, NextFunction } from 'express';

// Error categories for proper classification and handling
export enum ErrorCategory {
  VALIDATION = 'validation_error',
  API_ERROR = 'api_error',
  AUTHENTICATION = 'authentication_error',
  AUTHORIZATION = 'authorization_error',
  RESOURCE_LIMIT = 'resource_limit',
  RATE_LIMIT = 'rate_limit',
  TIMEOUT = 'timeout',
  NETWORK = 'network_error',
  STORAGE = 'storage_error',
  DATABASE = 'database_error',
  INTERNAL_ERROR = 'internal_error',
  SYSTEM_ERROR = 'system_error',
  UNKNOWN = 'unknown_error'
}

// Error with metadata for improved diagnostics
export interface EnhancedError extends Error {
  statusCode?: number;
  category: ErrorCategory;
  timestamp: Date;
  metadata?: any;
  originalError?: Error;
}

// Circuit breaker states
type CircuitState = 'closed' | 'open' | 'half-open';

// Circuit breaker configuration
interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeout: number;
  halfOpenMaxCalls: number;
}

// Circuit breaker tracking
interface CircuitBreaker {
  service: string;
  state: CircuitState;
  failures: number;
  lastFailure: number;
  halfOpenCalls: number;
  config: CircuitBreakerConfig;
}

/**
 * Error Handler Class
 * 
 * Singleton class for centralized error handling across the application.
 * Provides consistent error formatting, logging, and circuit breaker functionality.
 */
class ErrorHandler {
  private static instance: ErrorHandler;
  
  // Error logs directory
  private logsDir: string;
  private errorLogPath: string;
  
  // Circuit breakers for external services
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  
  // Default circuit breaker configuration
  private defaultCircuitConfig: CircuitBreakerConfig = {
    failureThreshold: 5,
    resetTimeout: 30000, // 30 seconds
    halfOpenMaxCalls: 3
  };
  
  /**
   * Private constructor to prevent direct instantiation
   */
  private constructor() {
    // Set up error logging
    this.logsDir = path.join(process.cwd(), 'logs');
    this.errorLogPath = path.join(this.logsDir, 'errors.log');
    
    // Ensure logs directory exists
    this.ensureLogsDirectory();
    
    // Initialize default circuit breakers
    this.initializeCircuitBreakers();
    
    console.log('[ErrorHandler] Initialized');
  }
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }
  
  /**
   * Create an enhanced error object with additional metadata
   */
  public createError(
    message: string,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    metadata?: any,
    originalError?: Error
  ): EnhancedError {
    const error: EnhancedError = new Error(message) as EnhancedError;
    error.name = category;
    error.category = category;
    error.timestamp = new Date();
    error.metadata = metadata;
    error.originalError = originalError;
    
    // Set appropriate HTTP status code based on category
    error.statusCode = this.getStatusCodeForCategory(category);
    
    return error;
  }
  
  /**
   * Get appropriate HTTP status code for error category
   */
  private getStatusCodeForCategory(category: ErrorCategory): number {
    switch (category) {
      case ErrorCategory.VALIDATION:
        return 400; // Bad Request
      case ErrorCategory.AUTHENTICATION:
        return 401; // Unauthorized
      case ErrorCategory.AUTHORIZATION:
        return 403; // Forbidden
      case ErrorCategory.RESOURCE_LIMIT:
      case ErrorCategory.RATE_LIMIT:
        return 429; // Too Many Requests
      case ErrorCategory.TIMEOUT:
        return 408; // Request Timeout
      case ErrorCategory.NETWORK:
        return 502; // Bad Gateway
      case ErrorCategory.STORAGE:
      case ErrorCategory.DATABASE:
        return 500; // Internal Server Error
      case ErrorCategory.API_ERROR:
        return 400; // Bad Request
      default:
        return 500; // Internal Server Error
    }
  }
  
  /**
   * Handle errors consistently throughout the application
   */
  public handleError(
    error: Error | EnhancedError,
    req?: Request,
    res?: Response
  ): { category: ErrorCategory, statusCode: number } {
    // Convert to enhanced error if not already
    const enhancedError = this.ensureEnhancedError(error);
    
    // Log the error
    this.logError(enhancedError, req);
    
    // Update circuit breaker if applicable
    if (enhancedError.metadata?.service) {
      this.recordServiceFailure(enhancedError.metadata.service);
    }
    
    // Send response if response object is provided
    if (res) {
      const statusCode = enhancedError.statusCode || 500;
      
      res.status(statusCode).json({
        error: {
          message: enhancedError.message,
          type: enhancedError.category,
          id: `error-${Date.now()}`,
          timestamp: enhancedError.timestamp.toISOString()
        }
      });
    }
    
    // Return the error category and status code for further handling
    return {
      category: enhancedError.category,
      statusCode: enhancedError.statusCode || 500
    };
  }
  
  /**
   * Express middleware for handling errors
   */
  public errorMiddleware() {
    return (err: Error, req: Request, res: Response, next: NextFunction) => {
      this.handleError(err, req, res);
    };
  }
  
  /**
   * Convert a standard error to an enhanced error
   */
  private ensureEnhancedError(error: Error | EnhancedError): EnhancedError {
    if ((error as EnhancedError).category) {
      return error as EnhancedError;
    }
    
    // Attempt to determine error category from error name or message
    let category = ErrorCategory.UNKNOWN;
    
    if (error.name === 'ValidationError' || error.message.includes('validation')) {
      category = ErrorCategory.VALIDATION;
    } else if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
      category = ErrorCategory.TIMEOUT;
    } else if (error.name === 'NetworkError' || error.message.includes('network')) {
      category = ErrorCategory.NETWORK;
    } else if (error.message.includes('database') || error.message.includes('sql')) {
      category = ErrorCategory.DATABASE;
    } else if (error.message.includes('permission') || error.message.includes('access')) {
      category = ErrorCategory.AUTHORIZATION;
    }
    
    // Create a new enhanced error
    return this.createError(
      error.message,
      category,
      undefined,
      error
    );
  }
  
  /**
   * Log error to file and console
   */
  private logError(error: EnhancedError, req?: Request): void {
    // Create a structured log entry
    const logEntry = {
      timestamp: error.timestamp.toISOString(),
      level: 'error',
      message: error.message,
      category: error.category,
      stack: error.stack,
      metadata: error.metadata,
      request: req ? {
        method: req.method,
        url: req.url,
        params: req.params,
        query: req.query,
        user: req.user ? { id: (req.user as any).id } : undefined
      } : undefined
    };
    
    // Log to console
    console.error(`[Error][${error.category}] ${error.message}`, {
      metadata: error.metadata,
      stack: error.stack
    });
    
    // Log to file
    this.appendToErrorLog(logEntry);
  }
  
  /**
   * Append error log to file
   */
  private appendToErrorLog(logEntry: any): void {
    try {
      fs.appendFileSync(
        this.errorLogPath,
        JSON.stringify(logEntry) + '\n'
      );
    } catch (fileError) {
      console.error(`Failed to write to error log: ${fileError}`);
    }
  }
  
  /**
   * Ensure logs directory exists
   */
  private ensureLogsDirectory(): void {
    try {
      if (!fs.existsSync(this.logsDir)) {
        fs.mkdirSync(this.logsDir, { recursive: true });
      }
    } catch (error) {
      console.error(`Failed to create logs directory: ${error}`);
    }
  }
  
  /**
   * Initialize circuit breakers for critical services
   */
  private initializeCircuitBreakers(): void {
    // Initialize circuit breakers for external services
    const services = [
      'llama3_api',
      'gemma3_api',
      'database',
      'storage',
      'authentication'
    ];
    
    for (const service of services) {
      this.circuitBreakers.set(service, {
        service,
        state: 'closed',
        failures: 0,
        lastFailure: 0,
        halfOpenCalls: 0,
        config: { ...this.defaultCircuitConfig }
      });
    }
  }
  
  /**
   * Record a service failure and potentially open circuit breaker
   */
  public recordServiceFailure(service: string): boolean {
    let circuitBreaker = this.circuitBreakers.get(service);
    
    // Create circuit breaker if it doesn't exist
    if (!circuitBreaker) {
      circuitBreaker = {
        service,
        state: 'closed',
        failures: 0,
        lastFailure: 0,
        halfOpenCalls: 0,
        config: { ...this.defaultCircuitConfig }
      };
      this.circuitBreakers.set(service, circuitBreaker);
    }
    
    // If circuit is already open, nothing to do
    if (circuitBreaker.state === 'open') {
      return false;
    }
    
    // Update failure count and timestamp
    circuitBreaker.failures++;
    circuitBreaker.lastFailure = Date.now();
    
    // If in half-open state, open the circuit again
    if (circuitBreaker.state === 'half-open') {
      circuitBreaker.state = 'open';
      circuitBreaker.failures = circuitBreaker.config.failureThreshold;
      console.warn(`[CircuitBreaker] Service ${service} failed in half-open state, circuit opened again`);
      
      // Set timeout to try half-open again
      setTimeout(() => {
        if (circuitBreaker!.state === 'open') {
          circuitBreaker!.state = 'half-open';
          circuitBreaker!.halfOpenCalls = 0;
          console.log(`[CircuitBreaker] Service ${service} transitioned from open to half-open`);
        }
      }, circuitBreaker.config.resetTimeout);
      
      return false;
    }
    
    // Check if we need to open the circuit
    if (circuitBreaker.failures >= circuitBreaker.config.failureThreshold) {
      circuitBreaker.state = 'open';
      console.warn(`[CircuitBreaker] Circuit opened for service ${service} after ${circuitBreaker.failures} failures`);
      
      // Set timeout to try half-open
      setTimeout(() => {
        if (circuitBreaker!.state === 'open') {
          circuitBreaker!.state = 'half-open';
          circuitBreaker!.halfOpenCalls = 0;
          console.log(`[CircuitBreaker] Service ${service} transitioned from open to half-open`);
        }
      }, circuitBreaker.config.resetTimeout);
      
      return false;
    }
    
    return true;
  }
  
  /**
   * Check if a service is available (circuit closed or half-open)
   */
  public isServiceAvailable(service: string): boolean {
    const circuitBreaker = this.circuitBreakers.get(service);
    
    // If no circuit breaker exists, assume service is available
    if (!circuitBreaker) {
      return true;
    }
    
    // If circuit is open, service is not available
    if (circuitBreaker.state === 'open') {
      return false;
    }
    
    // If circuit is half-open, allow limited calls
    if (circuitBreaker.state === 'half-open') {
      // Increment call count
      circuitBreaker.halfOpenCalls++;
      
      // Allow call if under the threshold
      if (circuitBreaker.halfOpenCalls <= circuitBreaker.config.halfOpenMaxCalls) {
        return true;
      }
      
      return false;
    }
    
    // Circuit is closed, service is available
    return true;
  }
  
  /**
   * Record a successful service call
   */
  public recordServiceSuccess(service: string): void {
    const circuitBreaker = this.circuitBreakers.get(service);
    
    // If no circuit breaker exists or circuit is closed, nothing to do
    if (!circuitBreaker || circuitBreaker.state === 'closed') {
      return;
    }
    
    // If circuit is half-open, check if we should close it
    if (circuitBreaker.state === 'half-open') {
      // Reset circuit on successful call in half-open state
      circuitBreaker.state = 'closed';
      circuitBreaker.failures = 0;
      circuitBreaker.halfOpenCalls = 0;
      console.log(`[CircuitBreaker] Circuit closed for service ${service} after successful half-open call`);
    }
  }
  
  /**
   * Get circuit breaker status for all services
   */
  public getCircuitBreakerStatus(): Record<string, { state: CircuitState; failures: number; lastFailure: number }> {
    const status: Record<string, { state: CircuitState; failures: number; lastFailure: number }> = {};
    
    for (const [service, circuitBreaker] of this.circuitBreakers.entries()) {
      status[service] = {
        state: circuitBreaker.state,
        failures: circuitBreaker.failures,
        lastFailure: circuitBreaker.lastFailure
      };
    }
    
    return status;
  }
}

// Export singleton instance
export const errorHandler = ErrorHandler.getInstance();