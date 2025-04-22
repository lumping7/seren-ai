/**
 * Error Handling and Resilience System
 * 
 * Provides a comprehensive error handling framework for the AI system, 
 * including error categorization, retry mechanisms, circuit breaking,
 * and graceful degradation capabilities.
 */

import fs from 'fs';
import path from 'path';
import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';

// Error categories
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

// Circuit breaker configuration
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
  private logEnabled: boolean = true;
  private logPath: string;
  private maxLogSize: number = 100; // Max number of errors to keep in memory
  
  private constructor() {
    this.logPath = path.join(process.cwd(), 'logs', 'errors.log');
    
    // Create log directory if it doesn't exist
    const logDir = path.dirname(this.logPath);
    if (!fs.existsSync(logDir)) {
      try {
        fs.mkdirSync(logDir, { recursive: true });
      } catch (error) {
        console.error(`[ErrorHandler] Failed to create log directory: ${error}`);
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
    // Define circuit breakers for critical services
    const services = [
      'llama3_api',
      'gemma3_api',
      'database',
      'knowledge_base',
      'reasoning_engine'
    ];
    
    for (const service of services) {
      this.circuitBreakers.set(service, {
        service,
        state: CircuitState.CLOSED,
        failureCount: 0,
        failureThreshold: 5, // After 5 failures
        resetTimeout: 30000, // 30 seconds
        lastFailureTime: 0,
        openUntil: 0
      });
    }
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
    if (this.logEnabled) {
      this.logError(errorData);
    }
    
    // Update circuit breaker if applicable
    if (error.service) {
      this.updateCircuitBreaker(error.service);
    }
    
    // If response object provided, send error response
    if (res) {
      const statusCode = this.getHttpStatusCode(errorData.category);
      res.status(statusCode).json({
        error: errorData.message,
        error_id: errorData.requestId,
        category: errorData.category,
        timestamp: errorData.timestamp,
        resolution: errorData.resolution
      });
    }
    
    return errorData;
  }
  
  /**
   * Create structured error data object from an error
   */
  private createErrorData(error: any, req?: Request): ErrorData {
    // Extract original error message if it's wrapped
    let message = error.message || 'Unknown error occurred';
    if (typeof error === 'string') {
      message = error;
    }
    
    // Generate a request ID if not provided
    const requestId = 
      (req && req.headers['x-request-id'] as string) || 
      `err-${uuidv4().split('-')[0]}`;
    
    // Get path from request if available
    const path = req ? req.path : undefined;
    
    // Categorize error
    const category = 
      error.category || 
      this.categorizeError(error);
    
    // Assess severity
    const severity = this.assessSeverity(category, error);
    
    // Determine if retryable
    const retryable = this.isRetryable(category, error);
    
    // Suggest resolution
    const resolution = this.suggestResolution(category, error);
    
    const errorData: ErrorData = {
      category,
      severity,
      message,
      details: error.details || undefined,
      stack: error.stack,
      requestId,
      timestamp: new Date(),
      path,
      retryable,
      resolution
    };
    
    return errorData;
  }
  
  /**
   * Categorize an error based on its properties
   */
  private categorizeError(error: any): ErrorCategory {
    // Check for validation errors
    if (
      error.name === 'ValidationError' || 
      error.name === 'ZodError' ||
      (error.details && error.details.some) ||
      error.message?.includes('validation')
    ) {
      return ErrorCategory.VALIDATION;
    }
    
    // Check for resource limit errors
    if (
      error.message?.includes('memory') ||
      error.message?.includes('capacity') ||
      error.message?.includes('limit') ||
      error.message?.includes('quota') ||
      error.code === 'ENOMEM'
    ) {
      return ErrorCategory.RESOURCE_LIMIT;
    }
    
    // Check for model errors
    if (
      error.message?.includes('model') ||
      error.message?.includes('inference') ||
      error.message?.includes('prediction')
    ) {
      return ErrorCategory.MODEL_ERROR;
    }
    
    // Check for database errors
    if (
      error.name === 'SequelizeError' ||
      error.name === 'MongoError' ||
      error.code === 'SQLITE_ERROR' ||
      error.message?.includes('database') ||
      error.message?.includes('query') ||
      error.message?.includes('SQL')
    ) {
      return ErrorCategory.DATABASE_ERROR;
    }
    
    // Check for network errors
    if (
      error.code === 'ECONNREFUSED' ||
      error.code === 'ECONNRESET' ||
      error.code === 'ETIMEDOUT' ||
      error.name === 'NetworkError' ||
      error.message?.includes('network') ||
      error.message?.includes('connection') ||
      error.message?.includes('timeout')
    ) {
      return ErrorCategory.NETWORK_ERROR;
    }
    
    // Check for authorization errors
    if (
      error.name === 'UnauthorizedError' ||
      error.status === 401 ||
      error.status === 403 ||
      error.message?.includes('auth') ||
      error.message?.includes('permission') ||
      error.message?.includes('token')
    ) {
      return ErrorCategory.AUTHORIZATION_ERROR;
    }
    
    // Internal errors include any errors explicitly marked as internal or thrown by our code
    if (
      error.internal === true ||
      error.isInternal ||
      error.message?.includes('internal')
    ) {
      return ErrorCategory.INTERNAL_ERROR;
    }
    
    // Default to unknown
    return ErrorCategory.UNKNOWN_ERROR;
  }
  
  /**
   * Assess error severity based on category and details
   */
  private assessSeverity(category: ErrorCategory, error: any): ErrorSeverity {
    // Critical errors
    if (
      category === ErrorCategory.RESOURCE_LIMIT ||
      (category === ErrorCategory.DATABASE_ERROR && !error.message?.includes('transient'))
    ) {
      return ErrorSeverity.CRITICAL;
    }
    
    // High severity errors
    if (
      category === ErrorCategory.MODEL_ERROR ||
      category === ErrorCategory.NETWORK_ERROR ||
      category === ErrorCategory.INTERNAL_ERROR
    ) {
      return ErrorSeverity.HIGH;
    }
    
    // Medium severity errors
    if (
      category === ErrorCategory.AUTHORIZATION_ERROR ||
      (category === ErrorCategory.DATABASE_ERROR && error.message?.includes('transient'))
    ) {
      return ErrorSeverity.MEDIUM;
    }
    
    // Low severity errors
    if (
      category === ErrorCategory.VALIDATION ||
      category === ErrorCategory.UNKNOWN_ERROR
    ) {
      return ErrorSeverity.LOW;
    }
    
    return ErrorSeverity.MEDIUM;
  }
  
  /**
   * Determine if an error is retryable
   */
  private isRetryable(category: ErrorCategory, error: any): boolean {
    // Generally, these categories are retryable
    if (
      category === ErrorCategory.NETWORK_ERROR ||
      category === ErrorCategory.RESOURCE_LIMIT
    ) {
      return true;
    }
    
    // Some database errors are retryable
    if (
      category === ErrorCategory.DATABASE_ERROR && 
      (
        error.message?.includes('deadlock') ||
        error.message?.includes('lock') ||
        error.message?.includes('timeout') ||
        error.message?.includes('transient')
      )
    ) {
      return true;
    }
    
    // Some model errors might be retryable
    if (
      category === ErrorCategory.MODEL_ERROR &&
      (
        error.message?.includes('timeout') ||
        error.message?.includes('overloaded')
      )
    ) {
      return true;
    }
    
    // Non-retryable categories
    if (
      category === ErrorCategory.VALIDATION ||
      category === ErrorCategory.AUTHORIZATION_ERROR
    ) {
      return false;
    }
    
    return false;
  }
  
  /**
   * Suggest resolution steps based on error category
   */
  private suggestResolution(category: ErrorCategory, error: any): string {
    switch (category) {
      case ErrorCategory.VALIDATION:
        return 'Check input parameters for validity and correct format.';
      
      case ErrorCategory.RESOURCE_LIMIT:
        return 'Reduce request complexity or wait and try again later.';
      
      case ErrorCategory.MODEL_ERROR:
        return 'Retry with simplified input or try an alternative model.';
      
      case ErrorCategory.DATABASE_ERROR:
        return 'Check database connection and retry operation.';
      
      case ErrorCategory.NETWORK_ERROR:
        return 'Check network connectivity and retry operation.';
      
      case ErrorCategory.AUTHORIZATION_ERROR:
        return 'Verify credentials and permissions.';
      
      case ErrorCategory.INTERNAL_ERROR:
        return 'Report issue to system administrators.';
      
      default:
        return 'Review the request and try again.';
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
        return 500; // Internal Server Error
      
      case ErrorCategory.DATABASE_ERROR:
        return 503; // Service Unavailable
      
      case ErrorCategory.NETWORK_ERROR:
        return 503; // Service Unavailable
      
      case ErrorCategory.AUTHORIZATION_ERROR:
        return 401; // Unauthorized
      
      case ErrorCategory.INTERNAL_ERROR:
        return 500; // Internal Server Error
      
      default:
        return 500; // Internal Server Error
    }
  }
  
  /**
   * Track error in memory
   */
  private trackError(errorData: ErrorData): void {
    this.errorLog.push(errorData);
    
    // Limit the size of the error log
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog.shift();
    }
  }
  
  /**
   * Log error to file
   */
  private logError(errorData: ErrorData): void {
    try {
      const logEntry = JSON.stringify({
        timestamp: errorData.timestamp.toISOString(),
        requestId: errorData.requestId,
        category: errorData.category,
        severity: errorData.severity,
        message: errorData.message,
        path: errorData.path,
        details: errorData.details,
        stack: errorData.stack,
        retryable: errorData.retryable,
        resolution: errorData.resolution
      });
      
      fs.appendFileSync(this.logPath, logEntry + '\n');
    } catch (error) {
      console.error('[ErrorHandler] Failed to log error:', error);
    }
  }
  
  /**
   * Update circuit breaker state for a service
   */
  private updateCircuitBreaker(service: string): void {
    const breaker = this.circuitBreakers.get(service);
    if (!breaker) return;
    
    const now = Date.now();
    
    // Update based on current state
    switch (breaker.state) {
      case CircuitState.CLOSED:
        // Increment failure count
        breaker.failureCount++;
        breaker.lastFailureTime = now;
        
        // Check if threshold exceeded
        if (breaker.failureCount >= breaker.failureThreshold) {
          breaker.state = CircuitState.OPEN;
          breaker.openUntil = now + breaker.resetTimeout;
          console.warn(`[ErrorHandler] Circuit breaker opened for ${service}`);
        }
        break;
      
      case CircuitState.OPEN:
        // Check if it's time to try again
        if (now > breaker.openUntil) {
          breaker.state = CircuitState.HALF_OPEN;
          console.log(`[ErrorHandler] Circuit breaker half-opened for ${service}`);
        }
        break;
      
      case CircuitState.HALF_OPEN:
        // Failed again in half-open state
        breaker.state = CircuitState.OPEN;
        breaker.openUntil = now + breaker.resetTimeout;
        console.warn(`[ErrorHandler] Circuit breaker re-opened for ${service}`);
        break;
    }
  }
  
  /**
   * Register success with circuit breaker
   */
  public registerSuccess(service: string): void {
    const breaker = this.circuitBreakers.get(service);
    if (!breaker) return;
    
    // Update based on current state
    switch (breaker.state) {
      case CircuitState.CLOSED:
        // Reset failure count after successful operation
        breaker.failureCount = 0;
        break;
      
      case CircuitState.HALF_OPEN:
        // Success in half-open state, reset and close
        breaker.state = CircuitState.CLOSED;
        breaker.failureCount = 0;
        console.log(`[ErrorHandler] Circuit breaker closed for ${service}`);
        break;
      
      // No action needed for OPEN state
    }
  }
  
  /**
   * Check if circuit breaker allows request
   */
  public canMakeRequest(service: string): boolean {
    const breaker = this.circuitBreakers.get(service);
    if (!breaker) return true; // No breaker defined, allow request
    
    const now = Date.now();
    
    // Check if it's time to transition from OPEN to HALF_OPEN
    if (breaker.state === CircuitState.OPEN && now > breaker.openUntil) {
      breaker.state = CircuitState.HALF_OPEN;
      console.log(`[ErrorHandler] Circuit breaker half-opened for ${service}`);
    }
    
    // Allow requests in CLOSED state or HALF_OPEN state
    return breaker.state === CircuitState.CLOSED || breaker.state === CircuitState.HALF_OPEN;
  }
  
  /**
   * Get recent errors
   */
  public getRecentErrors(): ErrorData[] {
    return [...this.errorLog].reverse(); // Most recent first
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
    
    for (const error of this.errorLog) {
      const category = error.category;
      counts[category] = (counts[category] || 0) + 1;
    }
    
    return counts;
  }
  
  /**
   * Get circuit breaker status for all services
   */
  public getCircuitBreakerStatus(): Record<string, any> {
    const status: Record<string, any> = {};
    
    for (const [service, breaker] of this.circuitBreakers.entries()) {
      status[service] = {
        state: CircuitState[breaker.state],
        failureCount: breaker.failureCount,
        failureThreshold: breaker.failureThreshold,
        lastFailureTime: breaker.lastFailureTime,
        openUntil: breaker.openUntil
      };
    }
    
    return status;
  }
  
  /**
   * Reset circuit breakers (for testing or maintenance)
   */
  public resetCircuitBreakers(): void {
    for (const breaker of this.circuitBreakers.values()) {
      breaker.state = CircuitState.CLOSED;
      breaker.failureCount = 0;
      breaker.lastFailureTime = 0;
      breaker.openUntil = 0;
    }
    console.log('[ErrorHandler] All circuit breakers reset');
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
    if (details !== undefined) {
      error.details = details;
    }
    return error;
  }
}

export const errorHandler = ErrorHandler.getInstance();