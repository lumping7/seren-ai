/**
 * Performance Monitoring System
 * 
 * Provides real-time monitoring and analysis of system performance metrics,
 * including latency tracking, throughput measurement, error rate monitoring,
 * and resource utilization tracking.
 */

import fs from 'fs';
import path from 'path';
import { Transform } from 'stream';
import { v4 as uuidv4 } from 'uuid';
import os from 'os';

// Performance metric types
type MetricType = 'counter' | 'gauge' | 'histogram' | 'summary';

// Operation status types 
type OperationStatus = 'pending' | 'completed' | 'failed';

// Metric data structure
interface Metric {
  name: string;
  type: MetricType;
  value: number;
  timestamp: number;
  tags?: Record<string, string>;
}

// Operation tracking
interface Operation {
  id: string;
  name: string;
  startTime: number;
  endTime?: number;
  status: OperationStatus;
  duration?: number;
  parentId?: string;
  metadata?: Record<string, any>;
}

// Time window for metrics aggregation
interface TimeWindow {
  start: number;
  end: number;
  metrics: Metric[];
  operations: Operation[];
}

// Percentile calculation
interface Percentile {
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

/**
 * Performance Monitor Class
 * 
 * Provides comprehensive performance monitoring capabilities
 */
class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  
  // Metrics storage
  private metrics: Metric[] = [];
  private operations: Map<string, Operation> = new Map();
  private timeWindows: TimeWindow[] = [];
  private activeWindows: Map<string, TimeWindow> = new Map();
  
  // Configuration
  private metricsRetentionTime: number = 24 * 60 * 60 * 1000; // 24 hours
  private windowSize: number = 60 * 1000; // 1 minute
  private logMetrics: boolean = true;
  private logOperations: boolean = true;
  private logFilePath: string;
  private maxWindows: number = 1440; // 24 hours of 1-minute windows
  
  // Runtime stats
  private startTime: number;
  private lastCleanupTime: number;
  private cleanupInterval: number = 5 * 60 * 1000; // 5 minutes
  
  // Alert thresholds
  private thresholds: Record<string, number> = {
    error_rate: 0.05, // 5% error rate
    latency_p95: 1000, // 1000ms
    memory_usage: 0.85, // 85% memory usage
    cpu_usage: 0.9 // 90% CPU usage
  };
  
  /**
   * Private constructor to enforce singleton pattern
   */
  private constructor() {
    this.startTime = Date.now();
    this.lastCleanupTime = this.startTime;
    this.logFilePath = path.join(process.cwd(), 'logs', 'performance.log');
    
    // Ensure log directory exists
    const logDir = path.dirname(this.logFilePath);
    if (!fs.existsSync(logDir)) {
      try {
        fs.mkdirSync(logDir, { recursive: true });
      } catch (error) {
        console.error(`[PerformanceMonitor] Failed to create log directory: ${error}`);
        // Fallback to a directory we know exists
        this.logFilePath = path.join(os.tmpdir(), 'performance.log');
      }
    }
    
    // Schedule metrics cleanup
    setInterval(() => this.cleanupStaleMetrics(), this.cleanupInterval);
    
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
   * Record a single metric
   */
  public recordMetric(
    name: string,
    value: number,
    type: MetricType = 'gauge',
    tags?: Record<string, string>
  ): void {
    const metric: Metric = {
      name,
      type,
      value,
      timestamp: Date.now(),
      tags
    };
    
    this.metrics.push(metric);
    this.addMetricToCurrentWindow(metric);
    
    if (this.logMetrics) {
      this.logMetricToFile(metric);
    }
  }
  
  /**
   * Increment a counter metric
   */
  public incrementCounter(name: string, value: number = 1, tags?: Record<string, string>): void {
    this.recordMetric(name, value, 'counter', tags);
  }
  
  /**
   * Update a gauge metric
   */
  public updateGauge(name: string, value: number, tags?: Record<string, string>): void {
    this.recordMetric(name, value, 'gauge', tags);
  }
  
  /**
   * Record a histogram value
   */
  public recordHistogram(name: string, value: number, tags?: Record<string, string>): void {
    this.recordMetric(name, value, 'histogram', tags);
  }
  
  /**
   * Start tracking an operation
   */
  public startOperation(name: string, id: string = uuidv4(), parentId?: string): string {
    const operation: Operation = {
      id,
      name,
      startTime: Date.now(),
      status: 'pending',
      parentId
    };
    
    this.operations.set(id, operation);
    
    if (this.logOperations) {
      this.logOperationToFile('start', operation);
    }
    
    return id;
  }
  
  /**
   * End tracking an operation
   */
  public endOperation(
    id: string, 
    failed: boolean = false, 
    metadata?: Record<string, any>
  ): void {
    const operation = this.operations.get(id);
    if (!operation) {
      console.warn(`[PerformanceMonitor] Attempted to end unknown operation: ${id}`);
      return;
    }
    
    const endTime = Date.now();
    operation.endTime = endTime;
    operation.status = failed ? 'failed' : 'completed';
    operation.duration = endTime - operation.startTime;
    
    if (metadata) {
      operation.metadata = { ...operation.metadata, ...metadata };
    }
    
    // Update metrics
    this.recordMetric(`operation.${operation.name}.duration`, operation.duration, 'histogram');
    this.recordMetric(`operation.${operation.name}.${operation.status}`, 1, 'counter');
    
    this.addOperationToCurrentWindow(operation);
    
    if (this.logOperations) {
      this.logOperationToFile('end', operation);
    }
    
    // Check for alerts
    this.checkOperationAlerts(operation);
  }
  
  /**
   * Get metrics for a specific time range
   */
  public getMetrics(
    startTime: number, 
    endTime: number = Date.now(), 
    metricName?: string,
    metricType?: MetricType
  ): Metric[] {
    return this.metrics.filter(metric => {
      const timeMatch = metric.timestamp >= startTime && metric.timestamp <= endTime;
      const nameMatch = !metricName || metric.name === metricName;
      const typeMatch = !metricType || metric.type === metricType;
      return timeMatch && nameMatch && typeMatch;
    });
  }
  
  /**
   * Get operations for a specific time range
   */
  public getOperations(
    startTime: number, 
    endTime: number = Date.now(), 
    operationName?: string, 
    status?: OperationStatus
  ): Operation[] {
    const operations: Operation[] = [];
    
    for (const op of this.operations.values()) {
      const timeMatch = op.startTime >= startTime && 
                       (op.endTime ? op.endTime <= endTime : Date.now() <= endTime);
      const nameMatch = !operationName || op.name === operationName;
      const statusMatch = !status || op.status === status;
      
      if (timeMatch && nameMatch && statusMatch) {
        operations.push(op);
      }
    }
    
    return operations;
  }
  
  /**
   * Calculate percentiles for a metric
   */
  public calculatePercentiles(
    metricName: string, 
    startTime: number, 
    endTime: number = Date.now()
  ): Percentile {
    const values = this.getMetrics(startTime, endTime, metricName)
      .map(m => m.value)
      .sort((a, b) => a - b);
    
    if (values.length === 0) {
      return { p50: 0, p90: 0, p95: 0, p99: 0 };
    }
    
    return {
      p50: this.getPercentile(values, 50),
      p90: this.getPercentile(values, 90),
      p95: this.getPercentile(values, 95),
      p99: this.getPercentile(values, 99)
    };
  }
  
  /**
   * Get error rate for an operation
   */
  public getErrorRate(
    operationName: string,
    startTime: number,
    endTime: number = Date.now()
  ): number {
    const operations = this.getOperations(startTime, endTime, operationName);
    
    if (operations.length === 0) {
      return 0;
    }
    
    const failedCount = operations.filter(op => op.status === 'failed').length;
    return failedCount / operations.length;
  }
  
  /**
   * Get summary statistics
   */
  public getSummaryStats(): Record<string, any> {
    const now = Date.now();
    const oneHourAgo = now - (60 * 60 * 1000);
    
    // Calculate error rates for the last hour
    const errorRates: Record<string, number> = {};
    const operationNames = new Set<string>();
    
    for (const op of this.operations.values()) {
      if (op.startTime >= oneHourAgo) {
        operationNames.add(op.name);
      }
    }
    
    for (const name of operationNames) {
      errorRates[name] = this.getErrorRate(name, oneHourAgo);
    }
    
    // Get overall error rate
    const totalOps = this.getOperations(oneHourAgo).length;
    const failedOps = this.getOperations(oneHourAgo, now, undefined, 'failed').length;
    const overallErrorRate = totalOps > 0 ? failedOps / totalOps : 0;
    
    // Check if any error rates exceed thresholds
    if (overallErrorRate > this.thresholds.error_rate) {
      console.warn(`[PerformanceMonitor] High error rate detected: ${(overallErrorRate * 100).toFixed(2)}%`);
    }
    
    // Calculate system metrics
    const memUsage = process.memoryUsage();
    const memoryMetrics = {
      rss: memUsage.rss / (1024 * 1024),
      heapTotal: memUsage.heapTotal / (1024 * 1024),
      heapUsed: memUsage.heapUsed / (1024 * 1024),
      external: memUsage.external / (1024 * 1024)
    };
    
    return {
      uptime: (now - this.startTime) / 1000,
      totalMetricsCount: this.metrics.length,
      totalOperationsCount: this.operations.size,
      activeOperationsCount: Array.from(this.operations.values())
        .filter(op => !op.endTime).length,
      errorRates,
      overallErrorRate,
      memory: memoryMetrics,
      timeWindowsCount: this.timeWindows.length
    };
  }
  
  /**
   * Create a metrics stream transformer
   */
  public createMetricsTransform(): Transform {
    return new Transform({
      objectMode: true,
      transform: (chunk, encoding, callback) => {
        try {
          // Add metrics from the incoming chunk
          if (chunk && typeof chunk === 'object') {
            if (Array.isArray(chunk)) {
              for (const item of chunk) {
                if (item && typeof item === 'object' && 'name' in item && 'value' in item) {
                  this.recordMetric(
                    item.name as string,
                    item.value as number,
                    (item.type as MetricType) || 'gauge',
                    item.tags as Record<string, string>
                  );
                }
              }
            } else if ('name' in chunk && 'value' in chunk) {
              this.recordMetric(
                chunk.name as string,
                chunk.value as number,
                (chunk.type as MetricType) || 'gauge',
                chunk.tags as Record<string, string>
              );
            }
          }
          callback(null, chunk);
        } catch (error) {
          callback(error);
        }
      }
    });
  }
  
  /**
   * Add a metric to the current time window
   */
  private addMetricToCurrentWindow(metric: Metric): void {
    const windowKey = this.getWindowKey(metric.timestamp);
    let window = this.activeWindows.get(windowKey);
    
    if (!window) {
      const windowStart = Math.floor(metric.timestamp / this.windowSize) * this.windowSize;
      window = {
        start: windowStart,
        end: windowStart + this.windowSize,
        metrics: [],
        operations: []
      };
      
      this.activeWindows.set(windowKey, window);
      this.timeWindows.push(window);
      
      // Limit the number of windows we keep
      if (this.timeWindows.length > this.maxWindows) {
        this.timeWindows.shift();
      }
    }
    
    window.metrics.push(metric);
  }
  
  /**
   * Add an operation to the current time window
   */
  private addOperationToCurrentWindow(operation: Operation): void {
    if (!operation.endTime) return;
    
    const windowKey = this.getWindowKey(operation.endTime);
    let window = this.activeWindows.get(windowKey);
    
    if (!window) {
      const windowStart = Math.floor(operation.endTime / this.windowSize) * this.windowSize;
      window = {
        start: windowStart,
        end: windowStart + this.windowSize,
        metrics: [],
        operations: []
      };
      
      this.activeWindows.set(windowKey, window);
      this.timeWindows.push(window);
      
      // Limit the number of windows we keep
      if (this.timeWindows.length > this.maxWindows) {
        this.timeWindows.shift();
      }
    }
    
    window.operations.push(operation);
  }
  
  /**
   * Get window key from timestamp
   */
  private getWindowKey(timestamp: number): string {
    return Math.floor(timestamp / this.windowSize).toString();
  }
  
  /**
   * Calculate a percentile value
   */
  private getPercentile(sortedValues: number[], percentile: number): number {
    if (sortedValues.length === 0) return 0;
    if (sortedValues.length === 1) return sortedValues[0];
    
    const index = (percentile / 100) * (sortedValues.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) return sortedValues[lower];
    
    const weight = index - lower;
    return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
  }
  
  /**
   * Clean up stale metrics
   */
  private cleanupStaleMetrics(): void {
    const now = Date.now();
    const cutoffTime = now - this.metricsRetentionTime;
    
    // Clean up old metrics
    this.metrics = this.metrics.filter(m => m.timestamp >= cutoffTime);
    
    // Clean up old operations
    for (const [id, op] of this.operations.entries()) {
      if (op.endTime && op.endTime < cutoffTime) {
        this.operations.delete(id);
      }
    }
    
    // Clean up old time windows
    this.timeWindows = this.timeWindows.filter(w => w.end >= cutoffTime);
    
    // Reset active windows map
    this.activeWindows.clear();
    for (const window of this.timeWindows) {
      const key = this.getWindowKey(window.start);
      this.activeWindows.set(key, window);
    }
    
    this.lastCleanupTime = now;
  }
  
  /**
   * Log metric to file
   */
  private logMetricToFile(metric: Metric): void {
    try {
      const logLine = JSON.stringify({
        type: 'metric',
        timestamp: new Date(metric.timestamp).toISOString(),
        ...metric
      });
      
      fs.appendFileSync(this.logFilePath, logLine + '\n');
    } catch (error) {
      console.error(`[PerformanceMonitor] Failed to log metric: ${error}`);
    }
  }
  
  /**
   * Log operation to file
   */
  private logOperationToFile(event: 'start' | 'end', operation: Operation): void {
    try {
      const logLine = JSON.stringify({
        type: 'operation',
        event,
        timestamp: new Date(event === 'start' ? operation.startTime : (operation.endTime || Date.now())).toISOString(),
        ...operation
      });
      
      fs.appendFileSync(this.logFilePath, logLine + '\n');
    } catch (error) {
      console.error(`[PerformanceMonitor] Failed to log operation: ${error}`);
    }
  }
  
  /**
   * Check for operation alerts
   */
  private checkOperationAlerts(operation: Operation): void {
    if (!operation.duration) return;
    
    // Check for slow operations
    if (operation.duration > 10000) { // 10 seconds
      console.warn(`[PerformanceMonitor] Slow operation detected: ${operation.name} took ${operation.duration}ms`);
    }
    
    // Check for failed operations
    if (operation.status === 'failed') {
      // Calculate recent error rate for this operation type
      const oneMinuteAgo = Date.now() - (60 * 1000);
      const errorRate = this.getErrorRate(operation.name, oneMinuteAgo);
      
      if (errorRate > this.thresholds.error_rate) {
        console.warn(`[PerformanceMonitor] High error rate for ${operation.name}: ${(errorRate * 100).toFixed(2)}%`);
      }
    }
  }
}

export const performanceMonitor = PerformanceMonitor.getInstance();