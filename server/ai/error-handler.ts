/**
 * Error Handling and Resilience System
 * 
 * Provides a comprehensive error handling framework for the AI system, 
 * including error categorization, retry mechanisms, circuit breaking,
 * and graceful degradation capabilities.
 */

import { Request, Response, NextFunction } from 'express';
import { performanceMonitor } from './performance-monitor';
import fs from 'fs';
import path from 'path';

// Error categorization
export enum ErrorCategory {
  VALIDATION = 'validation',
  RESOURCE_LIMIT = 'resource_limit',
  MODEL_ERROR = 'model_error',
  DATABASE_ERROR = 'database_error',
  NETWORK_ERROR = 'network_error',
  AUTHORIZATION_ERROR = 'authorization_error',
  INTERNAL_ERROR = 'internal_error',
  UNKNOWN_ERROR = 'unknown_error'
}

// Error severity levels
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Structured error data
export interface ErrorData {
  category: ErrorCategory;
  severity: ErrorSeverity;
  message: string;
  details?: any;
  stack?: string;
  requestId?: string;
  timestamp: Date;
  path?: string;
  retryable: boolean;
  resolution?: string;
}

// Circuit breaker states
enum CircuitState {
  CLOSED,  // Normal operation - requests flow through
  OPEN,    // Failure threshold exceeded - requests are blocked
  HALF_OPEN // Testing if system has recovered
}

// Circuit breaker tracking for different services
interface CircuitBreaker {
  service: string;
  state: CircuitState;
  failureCount: number;
  failureThreshold: number;
  resetTimeout: number;
  lastFailureTime: number;
  openUntil: number;
}

/**
 * AI Error Handler Class
 * 
 * Provides centralized error handling, tracking, and resilience
 */
export class ErrorHandler {
  private static instance: ErrorHandler;
  private errorLog: ErrorData[] = [];
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private logEnabled: boolean;
  private logPath: string;
  private maxLogSize: number = 100; // Max number of errors to keep in memory
  
  /**
   * Private constructor - use getInstance()
   */
  private constructor() {
    this.logEnabled = process.env.ENABLE_ERROR_LOGS === 'true';
    this.logPath = process.env.ERROR_LOG_PATH || './logs/errors';
    
    // Ensure log directory exists
    if (this.logEnabled) {
      try {
        fs.mkdirSync(this.logPath, { recursive: true });
        console.log(`[ErrorHandler] Log directory created: ${this.logPath}`);
      } catch (error) {
        console.error(`[ErrorHandler] Error creating log directory: ${error}`);
        this.logEnabled = false;
      }
    }
    
    // Initialize circuit breakers for key services
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
   * Initialize circuit breakers for key services
   */
  private initializeCircuitBreakers(): void {
    const services = [
      'llama3', 'gemma3', 'hybrid', 'database', 
      'reasoning', 'knowledge', 'conversation'
    ];
    
    services.forEach(service => {
      this.circuitBreakers.set(service, {
        service,
        state: CircuitState.CLOSED,
        failureCount: 0,
        failureThreshold: 5, // 5 consecutive failures
        resetTimeout: 30000, // 30 seconds
        lastFailureTime: 0,
        openUntil: 0
      });
    });
  }
  
  /**
   * Handle and process an error
   */
  public handleError(error: any, req?: Request, res?: Response): ErrorData {
    // Create structured error data
    const errorData = this.createErrorData(error, req);
    
    // Track error
    this.trackError(errorData);
    
    // Log error
    this.logError(errorData);
    
    // Update circuit breaker if service-specific
    if (errorData.details?.service) {
      this.updateCircuitBreaker(errorData.details.service);
    }
    
    // Send error response if response object provided
    if (res && !res.headersSent) {
      const statusCode = this.getHttpStatusCode(errorData.category);
      
      // Send appropriate HTTP response
      res.status(statusCode).json({
        error: errorData.message,
        category: errorData.category,
        timestamp: errorData.timestamp,
        request_id: errorData.requestId,
        retryable: errorData.retryable,
        resolution: errorData.resolution
      });
    }
    
    return errorData;
  }
  
  /**
   * Create structured error data object from an error
   */
  private createErrorData(error: any, req?: Request): ErrorData {
    // Determine error category
    const category = this.categorizeError(error);
    
    // Determine error severity
    const severity = this.assessSeverity(category, error);
    
    // Determine if error is retryable
    const retryable = this.isRetryable(category, error);
    
    // Create error data
    const errorData: ErrorData = {
      category,
      severity,
      message: error.message || 'An unknown error occurred',
      details: error.details || {},
      stack: error.stack,
      requestId: req?.headers['x-request-id'] as string || error.requestId,
      timestamp: new Date(),
      path: req?.path,
      retryable,
      resolution: this.suggestResolution(category, error)
    };
    
    return errorData;
  }
  
  /**
   * Categorize an error based on its properties
   */
  private categorizeError(error: any): ErrorCategory {
    if (error.name === 'ValidationError' || error.code === 'VALIDATION_ERROR') {
      return ErrorCategory.VALIDATION;
    }
    
    if (error.name === 'ResourceLimitError' || error.code === 'RESOURCE_LIMIT_EXCEEDED') {
      return ErrorCategory.RESOURCE_LIMIT;
    }
    
    if (error.name === 'ModelError' || error.code === 'MODEL_ERROR') {
      return ErrorCategory.MODEL_ERROR;
    }
    
    if (error.name === 'DatabaseError' || error.code?.startsWith('DB_') || error.code?.startsWith('PG_')) {
      return ErrorCategory.DATABASE_ERROR;
    }
    
    if (error.name === 'NetworkError' || error.code === 'NETWORK_ERROR' || error.code === 'ECONNREFUSED') {
      return ErrorCategory.NETWORK_ERROR;
    }
    
    if (error.name === 'AuthorizationError' || error.status === 401 || error.status === 403) {
      return ErrorCategory.AUTHORIZATION_ERROR;
    }
    
    if (error.name === 'InternalError' || error.code === 'INTERNAL_ERROR') {
      return ErrorCategory.INTERNAL_ERROR;
    }
    
    return ErrorCategory.UNKNOWN_ERROR;
  }
  
  /**
   * Assess error severity based on category and details
   */
  private assessSeverity(category: ErrorCategory, error: any): ErrorSeverity {
    switch (category) {
      case ErrorCategory.VALIDATION:
        return ErrorSeverity.LOW;
        
      case ErrorCategory.RESOURCE_LIMIT:
        return error.details?.critical ? ErrorSeverity.HIGH : ErrorSeverity.MEDIUM;
        
      case ErrorCategory.MODEL_ERROR:
        return ErrorSeverity.MEDIUM;
        
      case ErrorCategory.DATABASE_ERROR:
        return error.details?.connection ? ErrorSeverity.HIGH : ErrorSeverity.MEDIUM;
        
      case ErrorCategory.NETWORK_ERROR:
        return ErrorSeverity.HIGH;
        
      case ErrorCategory.AUTHORIZATION_ERROR:
        return ErrorSeverity.LOW;
        
      case ErrorCategory.INTERNAL_ERROR:
        return ErrorSeverity.HIGH;
        
      case ErrorCategory.UNKNOWN_ERROR:
        return ErrorSeverity.MEDIUM;
        
      default:
        return ErrorSeverity.MEDIUM;
    }
  }
  
  /**
   * Determine if an error is retryable
   */
  private isRetryable(category: ErrorCategory, error: any): boolean {
    // Check if error has explicit retryable property
    if (typeof error.retryable === 'boolean') {
      return error.retryable;
    }
    
    // Determine based on category
    switch (category) {
      case ErrorCategory.VALIDATION:
        return false; // Validation errors won't succeed on retry
        
      case ErrorCategory.RESOURCE_LIMIT:
        return true; // Resource limits may clear up
        
      case ErrorCategory.MODEL_ERROR:
        return !error.details?.permanent; // Model errors may be temporary
        
      case ErrorCategory.DATABASE_ERROR:
        return !error.details?.data_integrity; // Data integrity issues aren't retryable
        
      case ErrorCategory.NETWORK_ERROR:
        return true; // Network errors are often temporary
        
      case ErrorCategory.AUTHORIZATION_ERROR:
        return false; // Auth errors require user action
        
      case ErrorCategory.INTERNAL_ERROR:
        return false; // Internal errors likely need fixes
        
      case ErrorCategory.UNKNOWN_ERROR:
        return true; // Unknown errors - safe to retry
        
      default:
        return false;
    }
  }
  
  /**
   * Suggest resolution steps based on error category
   */
  private suggestResolution(category: ErrorCategory, error: any): string {
    switch (category) {
      case ErrorCategory.VALIDATION:
        return 'Check input parameters and ensure they match the required format.';
        
      case ErrorCategory.RESOURCE_LIMIT:
        return 'The system is currently at capacity. Please retry later or reduce the complexity of your request.';
        
      case ErrorCategory.MODEL_ERROR:
        return 'The AI model encountered an issue. Please try with different input parameters or a different model.';
        
      case ErrorCategory.DATABASE_ERROR:
        return 'Database operation failed. Please ensure data is valid and try again.';
        
      case ErrorCategory.NETWORK_ERROR:
        return 'Network connection issue detected. Please check your connection and retry.';
        
      case ErrorCategory.AUTHORIZATION_ERROR:
        return 'Authentication or authorization failed. Please check your credentials.';
        
      case ErrorCategory.INTERNAL_ERROR:
        return 'An internal system error occurred. Please contact support if the issue persists.';
        
      case ErrorCategory.UNKNOWN_ERROR:
        return 'An unexpected error occurred. Please retry or contact support if the issue persists.';
        
      default:
        return 'Please retry or contact support if the issue persists.';
    }
  }
  
  /**
   * Get appropriate HTTP status code for an error category
   */
  private getHttpStatusCode(category: ErrorCategory): number {
    switch (category) {
      case ErrorCategory.VALIDATION:
        return 400; // Bad Request
        
      case ErrorCategory.RESOURCE_LIMIT:
        return 429; // Too Many Requests
        
      case ErrorCategory.MODEL_ERROR:
        return 422; // Unprocessable Entity
        
      case ErrorCategory.DATABASE_ERROR:
        return 500; // Internal Server Error
        
      case ErrorCategory.NETWORK_ERROR:
        return 503; // Service Unavailable
        
      case ErrorCategory.AUTHORIZATION_ERROR:
        return 401; // Unauthorized
        
      case ErrorCategory.INTERNAL_ERROR:
        return 500; // Internal Server Error
        
      case ErrorCategory.UNKNOWN_ERROR:
        return 500; // Internal Server Error
        
      default:
        return 500; // Internal Server Error
    }
  }
  
  /**
   * Track error in memory
   */
  private trackError(errorData: ErrorData): void {
    // Add to in-memory error log
    this.errorLog.push(errorData);
    
    // Limit size of error log
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog.shift();
    }
    
    // Track in performance monitor for metrics
    performanceMonitor.endOperation(
      errorData.requestId || 'unknown', 
      true
    );
  }
  
  /**
   * Log error to file
   */
  private logError(errorData: ErrorData): void {
    // Always log critical errors to console
    if (errorData.severity === ErrorSeverity.CRITICAL) {
      console.error(`[CRITICAL ERROR] ${errorData.message}`, {
        category: errorData.category,
        path: errorData.path,
        requestId: errorData.requestId,
        timestamp: errorData.timestamp
      });
    } else if (errorData.severity === ErrorSeverity.HIGH) {
      console.error(`[ERROR] ${errorData.message}`, {
        category: errorData.category,
        path: errorData.path,
        requestId: errorData.requestId
      });
    }
    
    // Write to log file if enabled
    if (this.logEnabled) {
      try {
        const logFile = path.join(this.logPath, `errors-${new Date().toISOString().split('T')[0]}.log`);
        fs.appendFileSync(logFile, JSON.stringify({
          ...errorData,
          timestamp: errorData.timestamp.toISOString()
        }) + '\n');
      } catch (error) {
        console.error(`[ErrorHandler] Error writing to log file: ${error}`);
      }
    }
  }
  
  /**
   * Update circuit breaker state for a service
   */
  private updateCircuitBreaker(service: string): void {
    const breaker = this.circuitBreakers.get(service);
    if (!breaker) return;
    
    const now = Date.now();
    
    switch (breaker.state) {
      case CircuitState.CLOSED:
        // In closed state, increment failure count
        breaker.failureCount++;
        breaker.lastFailureTime = now;
        
        // Check if failure threshold is reached
        if (breaker.failureCount >= breaker.failureThreshold) {
          breaker.state = CircuitState.OPEN;
          breaker.openUntil = now + breaker.resetTimeout;
          console.warn(`[ErrorHandler] Circuit breaker opened for service: ${service}`);
        }
        break;
        
      case CircuitState.OPEN:
        // In open state, check if it's time to try again
        if (now > breaker.openUntil) {
          breaker.state = CircuitState.HALF_OPEN;
          console.info(`[ErrorHandler] Circuit breaker half-open for service: ${service}`);
        }
        break;
        
      case CircuitState.HALF_OPEN:
        // In half-open state, failure means going back to open
        breaker.state = CircuitState.OPEN;
        breaker.openUntil = now + breaker.resetTimeout;
        console.warn(`[ErrorHandler] Circuit breaker re-opened for service: ${service}`);
        break;
    }
    
    // Update the circuit breaker
    this.circuitBreakers.set(service, breaker);
  }
  
  /**
   * Register success with circuit breaker
   */
  public registerSuccess(service: string): void {
    const breaker = this.circuitBreakers.get(service);
    if (!breaker) return;
    
    switch (breaker.state) {
      case CircuitState.CLOSED:
        // Reset failure count on success
        breaker.failureCount = 0;
        break;
        
      case CircuitState.HALF_OPEN:
        // In half-open state, success means going back to closed
        breaker.state = CircuitState.CLOSED;
        breaker.failureCount = 0;
        console.info(`[ErrorHandler] Circuit breaker closed for service: ${service}`);
        break;
    }
    
    // Update the circuit breaker
    this.circuitBreakers.set(service, breaker);
  }
  
  /**
   * Check if circuit breaker allows request
   */
  public canMakeRequest(service: string): boolean {
    const breaker = this.circuitBreakers.get(service);
    if (!breaker) return true;
    
    const now = Date.now();
    
    switch (breaker.state) {
      case CircuitState.CLOSED:
        return true;
        
      case CircuitState.OPEN:
        // Check if it's time to try again
        if (now > breaker.openUntil) {
          breaker.state = CircuitState.HALF_OPEN;
          this.circuitBreakers.set(service, breaker);
          console.info(`[ErrorHandler] Circuit breaker half-open for service: ${service}`);
          return true;
        }
        return false;
        
      case CircuitState.HALF_OPEN:
        // In half-open state, allow one request to test
        return true;
        
      default:
        return true;
    }
  }
  
  /**
   * Get recent errors
   */
  public getRecentErrors(): ErrorData[] {
    return [...this.errorLog];
  }
  
  /**
   * Get errors by category
   */
  public getErrorsByCategory(category: ErrorCategory): ErrorData[] {
    return this.errorLog.filter(error => error.category === category);
  }
  
  /**
   * Get error count by category
   */
  public getErrorCountByCategory(): Record<string, number> {
    const counts: Record<string, number> = {};
    
    // Initialize all categories to 0
    Object.values(ErrorCategory).forEach(category => {
      counts[category] = 0;
    });
    
    // Count errors by category
    this.errorLog.forEach(error => {
      counts[error.category]++;
    });
    
    return counts;
  }
  
  /**
   * Get circuit breaker status for all services
   */
  public getCircuitBreakerStatus(): Record<string, any> {
    const status: Record<string, any> = {};
    
    this.circuitBreakers.forEach((breaker, service) => {
      status[service] = {
        state: CircuitState[breaker.state],
        failureCount: breaker.failureCount,
        threshold: breaker.failureThreshold,
        openUntil: breaker.state === CircuitState.OPEN ? new Date(breaker.openUntil).toISOString() : null
      };
    });
    
    return status;
  }
  
  /**
   * Reset circuit breakers (for testing or maintenance)
   */
  public resetCircuitBreakers(): void {
    this.circuitBreakers.forEach((breaker, service) => {
      breaker.state = CircuitState.CLOSED;
      breaker.failureCount = 0;
      breaker.openUntil = 0;
      
      this.circuitBreakers.set(service, breaker);
    });
    
    console.log('[ErrorHandler] Circuit breakers reset');
  }
  
  /**
   * Express middleware for error handling
   */
  public errorMiddleware() {
    return (err: any, req: Request, res: Response, next: NextFunction) => {
      this.handleError(err, req, res);
    };
  }
  
  /**
   * Create custom error with specific properties
   */
  public createError(
    message: string, 
    category: ErrorCategory, 
    details?: any
  ): Error & { category: ErrorCategory; details?: any } {
    const error: Error & { category: ErrorCategory; details?: any } = new Error(message) as any;
    error.category = category;
    error.details = details;
    return error;
  }
}

export const errorHandler = ErrorHandler.getInstance();