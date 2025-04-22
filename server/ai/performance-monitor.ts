/**
 * Performance Monitor for AI Operations
 * 
 * This module tracks the performance of various AI operations,
 * providing insights into execution time, resource usage,
 * and success/failure rates.
 */

interface OperationRecord {
  id: string;
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  success: boolean;
  metadata?: Record<string, any>;
}

class PerformanceMonitor {
  private operations: Map<string, OperationRecord> = new Map();
  private historicalOperations: OperationRecord[] = [];
  private maxHistoricalRecords: number = 1000;
  private initialized: boolean = false;

  /**
   * Initialize the performance monitor
   */
  initialize(): void {
    if (this.initialized) {
      return;
    }

    this.initialized = true;
    console.log('[PerformanceMonitor] Initialized');
  }

  /**
   * Start tracking an operation
   */
  startOperation(name: string, id: string = Math.random().toString(36).substring(2, 9)): string {
    const operationId = `${name}_${id}`;
    
    this.operations.set(operationId, {
      id: operationId,
      name,
      startTime: Date.now(),
      success: false
    });
    
    return operationId;
  }

  /**
   * End tracking an operation
   */
  endOperation(
    operationId: string, 
    failed: boolean = false, 
    metadata?: Record<string, any>
  ): OperationRecord | null {
    const operation = this.operations.get(operationId);
    
    if (!operation) {
      console.warn(`[PerformanceMonitor] Operation ${operationId} not found`);
      return null;
    }
    
    const now = Date.now();
    
    operation.endTime = now;
    operation.duration = now - operation.startTime;
    operation.success = !failed;
    
    if (metadata) {
      operation.metadata = { ...operation.metadata, ...metadata };
    }
    
    // Move to historical records
    this.operations.delete(operationId);
    this.historicalOperations.unshift(operation);
    
    // Trim historical records if needed
    if (this.historicalOperations.length > this.maxHistoricalRecords) {
      this.historicalOperations = this.historicalOperations.slice(0, this.maxHistoricalRecords);
    }
    
    // Log slow operations
    if (operation.duration > 1000) {
      console.warn(`[PerformanceMonitor] Slow operation: ${operation.name} took ${operation.duration}ms`);
    }
    
    return operation;
  }

  /**
   * Get stats for a specific operation type
   */
  getOperationStats(name: string): {
    count: number;
    successCount: number;
    failureCount: number;
    avgDuration: number;
    maxDuration: number;
    minDuration: number;
  } {
    const operations = this.historicalOperations.filter(op => op.name === name);
    
    if (operations.length === 0) {
      return {
        count: 0,
        successCount: 0,
        failureCount: 0,
        avgDuration: 0,
        maxDuration: 0,
        minDuration: 0
      };
    }
    
    const successCount = operations.filter(op => op.success).length;
    const durations = operations.map(op => op.duration || 0);
    
    return {
      count: operations.length,
      successCount,
      failureCount: operations.length - successCount,
      avgDuration: durations.reduce((sum, d) => sum + d, 0) / durations.length,
      maxDuration: Math.max(...durations),
      minDuration: Math.min(...durations)
    };
  }

  /**
   * Get overall performance stats
   */
  getOverallStats(): Record<string, any> {
    const operationNames = new Set(this.historicalOperations.map(op => op.name));
    const stats: Record<string, any> = {};
    
    operationNames.forEach(name => {
      stats[name] = this.getOperationStats(name);
    });
    
    // Add global stats
    stats.global = {
      totalOperations: this.historicalOperations.length,
      activeOperations: this.operations.size,
      successRate: this.historicalOperations.length > 0
        ? this.historicalOperations.filter(op => op.success).length / this.historicalOperations.length
        : 0
    };
    
    return stats;
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();
performanceMonitor.initialize();