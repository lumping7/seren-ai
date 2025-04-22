/**
 * Performance Monitoring System
 * 
 * Provides detailed performance tracking for all AI operations,
 * enabling optimization, bottleneck detection, and system health monitoring.
 */

import { Request, Response, NextFunction } from 'express';
import { resourceManager } from './resource-manager';
import os from 'os';
import fs from 'fs';
import path from 'path';

// Performance metrics for different operations
interface OperationMetrics {
  count: number;
  totalTime: number;
  averageTime: number;
  minTime: number;
  maxTime: number;
  lastExecutionTime: number;
  lastMemoryUsage: number;
  lastCpuUsage: number;
  errorCount: number;
  successRate: number;
}

// System health metrics
interface SystemHealth {
  cpuUsage: number;
  memoryUsage: number;
  freeMemory: number;
  activeRequests: number;
  requestsPerMinute: number;
  errorRate: number;
  avgResponseTime: number;
  timestamp: Date;
}

// Types of operations we're tracking
type OperationType = 
  'llama3' | 
  'gemma3' | 
  'hybrid' | 
  'reasoning' | 
  'knowledge_retrieve' | 
  'knowledge_add' | 
  'conversation' | 
  'memory' |
  'api_request';

/**
 * Performance Monitor class
 * 
 * Tracks and analyzes system performance metrics
 */
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metricsMap: Map<OperationType, OperationMetrics>;
  private healthHistory: SystemHealth[];
  private requestsInLastMinute: number[];
  private lastMinuteTimestamp: number;
  private logEnabled: boolean;
  private logPath: string;
  private startTime: number;
  
  /**
   * Private constructor - use getInstance()
   */
  private constructor() {
    this.metricsMap = new Map<OperationType, OperationMetrics>();
    this.healthHistory = [];
    this.requestsInLastMinute = [];
    this.lastMinuteTimestamp = Date.now();
    this.logEnabled = process.env.ENABLE_PERFORMANCE_LOGS === 'true';
    this.logPath = process.env.PERFORMANCE_LOG_PATH || './logs/performance';
    this.startTime = Date.now();
    
    // Initialize metrics for all operation types
    const operationTypes: OperationType[] = [
      'llama3', 'gemma3', 'hybrid', 'reasoning', 
      'knowledge_retrieve', 'knowledge_add', 'conversation', 
      'memory', 'api_request'
    ];
    
    operationTypes.forEach(type => {
      this.metricsMap.set(type, this.createEmptyMetrics());
    });
    
    // Ensure log directory exists
    if (this.logEnabled) {
      try {
        fs.mkdirSync(this.logPath, { recursive: true });
        console.log(`[PerformanceMonitor] Log directory created: ${this.logPath}`);
      } catch (error) {
        console.error(`[PerformanceMonitor] Error creating log directory: ${error}`);
        this.logEnabled = false;
      }
    }
    
    // Set up periodic health checks
    setInterval(() => this.recordSystemHealth(), 60000); // Every minute
    
    console.log('[PerformanceMonitor] Initialized');
  }
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }
  
  /**
   * Create empty metrics object with initial values
   */
  private createEmptyMetrics(): OperationMetrics {
    return {
      count: 0,
      totalTime: 0,
      averageTime: 0,
      minTime: Number.MAX_VALUE,
      maxTime: 0,
      lastExecutionTime: 0,
      lastMemoryUsage: 0,
      lastCpuUsage: 0,
      errorCount: 0,
      successRate: 100
    };
  }
  
  /**
   * Record start of operation for tracking
   */
  public startOperation(type: OperationType, requestId?: string): string {
    const id = requestId || `${type}-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    // Store start time in a private symbol property of the request object
    const key = `${id}-start`;
    (global as any)[key] = {
      startTime: Date.now(),
      startMemory: process.memoryUsage().heapUsed,
      type
    };
    
    this.recordRequest();
    
    return id;
  }
  
  /**
   * Record end of operation and update metrics
   */
  public endOperation(id: string, error: boolean = false): void {
    const key = `${id}-start`;
    const startData = (global as any)[key];
    
    if (!startData) {
      console.warn(`[PerformanceMonitor] No start data found for operation ${id}`);
      return;
    }
    
    const { startTime, startMemory, type } = startData;
    const endTime = Date.now();
    const executionTime = endTime - startTime;
    const memoryUsed = process.memoryUsage().heapUsed - startMemory;
    
    // Update metrics
    const metrics = this.metricsMap.get(type) || this.createEmptyMetrics();
    
    metrics.count += 1;
    metrics.totalTime += executionTime;
    metrics.lastExecutionTime = executionTime;
    metrics.lastMemoryUsage = memoryUsed;
    metrics.lastCpuUsage = resourceManager.getSystemResources().cpuUsage;
    
    if (error) {
      metrics.errorCount += 1;
    }
    
    // Update min/max times
    if (executionTime < metrics.minTime) {
      metrics.minTime = executionTime;
    }
    if (executionTime > metrics.maxTime) {
      metrics.maxTime = executionTime;
    }
    
    // Calculate averages
    metrics.averageTime = metrics.totalTime / metrics.count;
    metrics.successRate = ((metrics.count - metrics.errorCount) / metrics.count) * 100;
    
    // Update the metrics map
    this.metricsMap.set(type, metrics);
    
    // Clean up
    delete (global as any)[key];
    
    // Log if enabled
    if (this.logEnabled) {
      this.logOperation(type, id, executionTime, memoryUsed, error);
    }
    
    // Log slow operations
    if (executionTime > 2000) { // More than 2 seconds
      console.warn(`[PerformanceMonitor] Slow operation detected: ${type} ${id} took ${executionTime}ms`);
    }
  }
  
  /**
   * Record API request for request rate tracking
   */
  private recordRequest(): void {
    const now = Date.now();
    this.requestsInLastMinute.push(now);
    
    // Clean up old requests (older than 1 minute)
    const oneMinuteAgo = now - 60000;
    this.requestsInLastMinute = this.requestsInLastMinute.filter(time => time > oneMinuteAgo);
  }
  
  /**
   * Get current request rate (requests per minute)
   */
  public getRequestRate(): number {
    // Clean up old requests first
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    this.requestsInLastMinute = this.requestsInLastMinute.filter(time => time > oneMinuteAgo);
    
    return this.requestsInLastMinute.length;
  }
  
  /**
   * Record current system health metrics
   */
  private recordSystemHealth(): void {
    const resources = resourceManager.getSystemResources();
    const activeRequests = resources.activeRequests;
    const requestRate = this.getRequestRate();
    
    // Calculate average response time and error rate across all operations
    let totalResponseTime = 0;
    let totalCount = 0;
    let totalErrors = 0;
    
    this.metricsMap.forEach(metrics => {
      totalResponseTime += metrics.totalTime;
      totalCount += metrics.count;
      totalErrors += metrics.errorCount;
    });
    
    const avgResponseTime = totalCount > 0 ? totalResponseTime / totalCount : 0;
    const errorRate = totalCount > 0 ? (totalErrors / totalCount) * 100 : 0;
    
    // Create health snapshot
    const health: SystemHealth = {
      cpuUsage: resources.cpuUsage,
      memoryUsage: (resources.totalMemory - resources.availableMemory) / resources.totalMemory * 100,
      freeMemory: resources.availableMemory,
      activeRequests,
      requestsPerMinute: requestRate,
      errorRate,
      avgResponseTime,
      timestamp: new Date()
    };
    
    // Add to history, keeping last 60 entries (1 hour of data)
    this.healthHistory.push(health);
    if (this.healthHistory.length > 60) {
      this.healthHistory.shift();
    }
    
    // Log if enabled
    if (this.logEnabled) {
      this.logSystemHealth(health);
    }
    
    // Detect potential system issues
    this.detectSystemIssues(health);
  }
  
  /**
   * Log operation details to file
   */
  private logOperation(
    type: OperationType, 
    id: string, 
    executionTime: number, 
    memoryUsed: number, 
    error: boolean
  ): void {
    try {
      const logFile = path.join(this.logPath, `operations-${new Date().toISOString().split('T')[0]}.log`);
      const logEntry = {
        timestamp: new Date().toISOString(),
        type,
        id,
        executionTime,
        memoryUsed,
        error,
        profile: resourceManager.getProfile()
      };
      
      fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');
    } catch (error) {
      console.error(`[PerformanceMonitor] Error logging operation: ${error}`);
    }
  }
  
  /**
   * Log system health to file
   */
  private logSystemHealth(health: SystemHealth): void {
    try {
      const logFile = path.join(this.logPath, `health-${new Date().toISOString().split('T')[0]}.log`);
      fs.appendFileSync(logFile, JSON.stringify(health) + '\n');
    } catch (error) {
      console.error(`[PerformanceMonitor] Error logging system health: ${error}`);
    }
  }
  
  /**
   * Detect potential system issues based on health metrics
   */
  private detectSystemIssues(health: SystemHealth): void {
    // CPU usage > 90%
    if (health.cpuUsage > 90) {
      console.warn(`[PerformanceMonitor] High CPU usage detected: ${health.cpuUsage.toFixed(2)}%`);
    }
    
    // Memory usage > 90%
    if (health.memoryUsage > 90) {
      console.warn(`[PerformanceMonitor] High memory usage detected: ${health.memoryUsage.toFixed(2)}%`);
    }
    
    // Request rate > 1000 per minute
    if (health.requestsPerMinute > 1000) {
      console.warn(`[PerformanceMonitor] High request rate detected: ${health.requestsPerMinute} requests/minute`);
    }
    
    // Error rate > 5%
    if (health.errorRate > 5) {
      console.warn(`[PerformanceMonitor] High error rate detected: ${health.errorRate.toFixed(2)}%`);
    }
    
    // Average response time > 2000ms
    if (health.avgResponseTime > 2000) {
      console.warn(`[PerformanceMonitor] Slow average response time: ${health.avgResponseTime.toFixed(2)}ms`);
    }
  }
  
  /**
   * Get metrics for a specific operation type
   */
  public getMetrics(type: OperationType): OperationMetrics | undefined {
    return this.metricsMap.get(type);
  }
  
  /**
   * Get all operation metrics
   */
  public getAllMetrics(): Record<string, OperationMetrics> {
    const result: Record<string, OperationMetrics> = {};
    this.metricsMap.forEach((metrics, type) => {
      result[type] = metrics;
    });
    return result;
  }
  
  /**
   * Get recent system health history
   */
  public getHealthHistory(): SystemHealth[] {
    return [...this.healthHistory];
  }
  
  /**
   * Get latest system health snapshot
   */
  public getLatestHealth(): SystemHealth | undefined {
    if (this.healthHistory.length === 0) {
      return undefined;
    }
    return this.healthHistory[this.healthHistory.length - 1];
  }
  
  /**
   * Get system uptime in seconds
   */
  public getUptime(): number {
    return Math.floor((Date.now() - this.startTime) / 1000);
  }
  
  /**
   * Reset all metrics (for testing or maintenance)
   */
  public resetMetrics(): void {
    this.metricsMap.forEach((_, type) => {
      this.metricsMap.set(type, this.createEmptyMetrics());
    });
    this.healthHistory = [];
    this.requestsInLastMinute = [];
    this.lastMinuteTimestamp = Date.now();
    console.log('[PerformanceMonitor] Metrics reset');
  }
  
  /**
   * Express middleware for automatically tracking API requests
   */
  public trackApiRequest() {
    return (req: Request, res: Response, next: NextFunction) => {
      // Skip tracking for static assets
      if (req.path.includes('.') || req.path.startsWith('/static')) {
        return next();
      }
      
      const requestId = this.startOperation('api_request');
      
      // Track response time
      const end = res.end;
      res.end = (...args: any[]) => {
        const error = res.statusCode >= 400;
        this.endOperation(requestId, error);
        
        return end.apply(res, args);
      };
      
      next();
    };
  }
}

export const performanceMonitor = PerformanceMonitor.getInstance();