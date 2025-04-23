/**
 * Resource Manager
 * 
 * Manages and optimizes system resources for AI operations, ensuring
 * efficient utilization of compute resources and preventing overload.
 */

import os from 'os';
import { errorHandler, ErrorCategory } from './error-handler';

// Resource types
export type ResourceType = 'qwen_inference' | 'olympic_inference' | 'hybrid_inference' | 'reasoning' | 'system';

// Request priority levels
export type PriorityLevel = 'high' | 'normal' | 'low';

// Resource allocation request
export interface ResourceAllocationRequest {
  estimated_tokens?: number;
  estimated_duration_ms?: number;
  priority?: PriorityLevel;
  metadata?: Record<string, any>;
}

// Resource thresholds
interface ResourceThresholds {
  max_concurrent_operations: number;
  memory_usage_percentage: number;
  cpu_usage_percentage: number;
  gpu_usage_percentage: number;
  max_pending_requests: number;
  token_rate_limit: number;
}

// Resource status
export interface ResourceStatus {
  system_load: {
    cpu_usage: number;
    memory_usage: number;
    memory_available_gb: number;
    total_memory_gb: number;
  };
  current_allocations: Record<ResourceType, number>;
  pending_requests: number;
  queue_length: number;
  available_capacity: number;
  gpu_available: boolean;
  gpu_usage?: number;
}

// Resource allocation
interface ResourceAllocation {
  id: string;
  type: ResourceType;
  allocatedAt: number;
  estimatedDuration?: number;
  estimatedTokens?: number;
  priority: PriorityLevel;
  metadata?: Record<string, any>;
}

// Profile for resource allocation based on system capacity
interface ResourceProfile {
  name: string;
  description: string;
  thresholds: Record<ResourceType, ResourceThresholds>;
  queue_capacity: number;
  max_token_rate_per_minute: number;
}

/**
 * Resource Manager Class
 * 
 * Singleton class for managing and allocating system resources for AI operations.
 * Ensures that system resources are used efficiently and prevents overload.
 */
class ResourceManager {
  private static instance: ResourceManager;
  
  // Current resource allocations
  private allocations: Map<string, ResourceAllocation> = new Map();
  
  // Request queue
  private queue: Array<{
    type: ResourceType;
    request: ResourceAllocationRequest;
    timestamp: number;
    id: string;
  }> = [];
  
  // Resource profiles for different system capacities
  private profiles: Record<string, ResourceProfile> = {
    'standard': {
      name: 'Standard',
      description: 'Default profile for systems with 16GB RAM and 4-8 CPU cores',
      thresholds: {
        'system': {
          max_concurrent_operations: 50,
          memory_usage_percentage: 80,
          cpu_usage_percentage: 80,
          gpu_usage_percentage: 90,
          max_pending_requests: 100,
          token_rate_limit: 50000
        },
        'qwen_inference': {
          max_concurrent_operations: 8,
          memory_usage_percentage: 80,
          cpu_usage_percentage: 80,
          gpu_usage_percentage: 90,
          max_pending_requests: 20,
          token_rate_limit: 20000
        },
        'olympic_inference': {
          max_concurrent_operations: 10,
          memory_usage_percentage: 80,
          cpu_usage_percentage: 80,
          gpu_usage_percentage: 90,
          max_pending_requests: 25,
          token_rate_limit: 25000
        },
        'hybrid_inference': {
          max_concurrent_operations: 5,
          memory_usage_percentage: 80,
          cpu_usage_percentage: 80,
          gpu_usage_percentage: 90,
          max_pending_requests: 15,
          token_rate_limit: 15000
        },
        'reasoning': {
          max_concurrent_operations: 15,
          memory_usage_percentage: 80,
          cpu_usage_percentage: 80,
          gpu_usage_percentage: 90,
          max_pending_requests: 30,
          token_rate_limit: 30000
        }
      },
      queue_capacity: 200,
      max_token_rate_per_minute: 100000
    },
    'enhanced': {
      name: 'Enhanced',
      description: 'Profile for high-performance systems with 32GB+ RAM and 8+ CPU cores',
      thresholds: {
        'system': {
          max_concurrent_operations: 100,
          memory_usage_percentage: 85,
          cpu_usage_percentage: 85,
          gpu_usage_percentage: 95,
          max_pending_requests: 200,
          token_rate_limit: 100000
        },
        'qwen_inference': {
          max_concurrent_operations: 20,
          memory_usage_percentage: 85,
          cpu_usage_percentage: 85,
          gpu_usage_percentage: 95,
          max_pending_requests: 40,
          token_rate_limit: 40000
        },
        'olympic_inference': {
          max_concurrent_operations: 25,
          memory_usage_percentage: 85,
          cpu_usage_percentage: 85,
          gpu_usage_percentage: 95,
          max_pending_requests: 50,
          token_rate_limit: 50000
        },
        'hybrid_inference': {
          max_concurrent_operations: 15,
          memory_usage_percentage: 85,
          cpu_usage_percentage: 85,
          gpu_usage_percentage: 95,
          max_pending_requests: 30,
          token_rate_limit: 30000
        },
        'reasoning': {
          max_concurrent_operations: 30,
          memory_usage_percentage: 85,
          cpu_usage_percentage: 85,
          gpu_usage_percentage: 95,
          max_pending_requests: 60,
          token_rate_limit: 60000
        }
      },
      queue_capacity: 400,
      max_token_rate_per_minute: 200000
    },
    'minimal': {
      name: 'Minimal',
      description: 'Profile for limited systems with <16GB RAM',
      thresholds: {
        'system': {
          max_concurrent_operations: 20,
          memory_usage_percentage: 75,
          cpu_usage_percentage: 75,
          gpu_usage_percentage: 85,
          max_pending_requests: 50,
          token_rate_limit: 25000
        },
        'qwen_inference': {
          max_concurrent_operations: 3,
          memory_usage_percentage: 75,
          cpu_usage_percentage: 75,
          gpu_usage_percentage: 85,
          max_pending_requests: 10,
          token_rate_limit: 10000
        },
        'olympic_inference': {
          max_concurrent_operations: 4,
          memory_usage_percentage: 75,
          cpu_usage_percentage: 75,
          gpu_usage_percentage: 85,
          max_pending_requests: 12,
          token_rate_limit: 12000
        },
        'hybrid_inference': {
          max_concurrent_operations: 2,
          memory_usage_percentage: 75,
          cpu_usage_percentage: 75,
          gpu_usage_percentage: 85,
          max_pending_requests: 6,
          token_rate_limit: 6000
        },
        'reasoning': {
          max_concurrent_operations: 6,
          memory_usage_percentage: 75,
          cpu_usage_percentage: 75,
          gpu_usage_percentage: 85,
          max_pending_requests: 15,
          token_rate_limit: 15000
        }
      },
      queue_capacity: 100,
      max_token_rate_per_minute: 50000
    }
  };
  
  // Current profile
  private currentProfile: ResourceProfile;
  
  // Token usage tracking
  private tokenUsage: {
    lastMinute: number;
    lastReset: number;
  } = {
    lastMinute: 0,
    lastReset: Date.now()
  };
  
  // System metrics
  private systemMetrics: {
    cpuCores: number;
    totalMemoryGB: number;
    availableMemoryGB: number;
    cpuUsage: number;
    memoryUsagePercent: number;
    lastChecked: number;
  };
  
  /**
   * Private constructor to prevent direct instantiation
   */
  private constructor() {
    // Initialize system metrics
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const cpuCores = os.cpus().length;
    
    this.systemMetrics = {
      cpuCores,
      totalMemoryGB: totalMemory / (1024 * 1024 * 1024),
      availableMemoryGB: freeMemory / (1024 * 1024 * 1024),
      cpuUsage: 0,
      memoryUsagePercent: ((totalMemory - freeMemory) / totalMemory) * 100,
      lastChecked: Date.now()
    };
    
    // Set appropriate profile based on system metrics
    this.currentProfile = this.selectProfile();
    
    // Log initialization
    console.log('[ResourceManager] Initialized with enhanced profile');
    console.log(`[ResourceManager] System has ${Math.round(this.systemMetrics.totalMemoryGB)}GB RAM and ${this.systemMetrics.cpuCores} CPU cores`);
    
    // Start periodic tasks
    this.startPeriodicTasks();
  }
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): ResourceManager {
    if (!ResourceManager.instance) {
      ResourceManager.instance = new ResourceManager();
    }
    return ResourceManager.instance;
  }
  
  /**
   * Select appropriate resource profile based on system capacity
   */
  private selectProfile(): ResourceProfile {
    const { totalMemoryGB, cpuCores } = this.systemMetrics;
    
    if (totalMemoryGB >= 32 && cpuCores >= 8) {
      return this.profiles['enhanced'];
    } else if (totalMemoryGB >= 16 && cpuCores >= 4) {
      return this.profiles['standard'];
    } else {
      return this.profiles['minimal'];
    }
  }
  
  /**
   * Start periodic tasks for monitoring and maintenance
   */
  private startPeriodicTasks(): void {
    // Update system metrics every 30 seconds
    setInterval(() => this.updateSystemMetrics(), 30000);
    
    // Clean up stale allocations every minute
    setInterval(() => this.cleanupStaleAllocations(), 60000);
    
    // Process queue every 5 seconds
    setInterval(() => this.processQueue(), 5000);
    
    // Reset token usage counter every minute
    setInterval(() => this.resetTokenUsage(), 60000);
  }
  
  /**
   * Update system metrics
   */
  private updateSystemMetrics(): void {
    try {
      const totalMemory = os.totalmem();
      const freeMemory = os.freemem();
      
      // Calculate memory usage percentage
      const memoryUsagePercent = ((totalMemory - freeMemory) / totalMemory) * 100;
      
      // Get CPU usage (average across cores)
      const cpuInfo = os.cpus();
      let totalIdle = 0;
      let totalTick = 0;
      
      for (const cpu of cpuInfo) {
        for (const type in cpu.times) {
          totalTick += (cpu.times as any)[type];
        }
        totalIdle += cpu.times.idle;
      }
      
      const idlePercent = totalIdle / totalTick;
      const cpuUsage = 100 - (idlePercent * 100);
      
      // Update system metrics
      this.systemMetrics = {
        ...this.systemMetrics,
        availableMemoryGB: freeMemory / (1024 * 1024 * 1024),
        cpuUsage,
        memoryUsagePercent,
        lastChecked: Date.now()
      };
      
      // Check if we need to adjust the profile based on current usage
      const currentProfile = this.currentProfile;
      const newProfile = this.selectProfile();
      
      if (newProfile.name !== currentProfile.name) {
        console.log(`[ResourceManager] Switching from ${currentProfile.name} to ${newProfile.name} profile based on system metrics`);
        this.currentProfile = newProfile;
      }
    } catch (error) {
      errorHandler.handleError(
        errorHandler.createError(
          'Failed to update system metrics',
          ErrorCategory.SYSTEM_ERROR,
          { error }
        )
      );
    }
  }
  
  /**
   * Clean up stale allocations
   */
  private cleanupStaleAllocations(): void {
    const now = Date.now();
    let cleanedCount = 0;
    
    for (const [id, allocation] of this.allocations.entries()) {
      // If an allocation is older than 5 minutes, consider it stale
      if (now - allocation.allocatedAt > 5 * 60 * 1000) {
        this.allocations.delete(id);
        cleanedCount++;
        
        console.warn(`[ResourceManager] Cleaned up stale allocation ${id} of type ${allocation.type}`);
      }
    }
    
    if (cleanedCount > 0) {
      console.log(`[ResourceManager] Cleaned up ${cleanedCount} stale allocations`);
    }
  }
  
  /**
   * Process the request queue
   */
  private processQueue(): void {
    if (this.queue.length === 0) return;
    
    const now = Date.now();
    let processedCount = 0;
    
    // Process up to 10 requests at a time
    for (let i = 0; i < Math.min(10, this.queue.length); i++) {
      const request = this.queue[0];
      
      // Check if we can allocate resources for this request
      if (this.canAllocateResources(request.type)) {
        // Remove from queue
        this.queue.shift();
        
        // Allocate resources
        this.allocations.set(request.id, {
          id: request.id,
          type: request.type,
          allocatedAt: now,
          estimatedDuration: request.request.estimated_duration_ms,
          estimatedTokens: request.request.estimated_tokens,
          priority: request.request.priority || 'normal',
          metadata: request.request.metadata
        });
        
        processedCount++;
      } else {
        // If we can't allocate for this request, we probably can't allocate for others either
        break;
      }
    }
    
    if (processedCount > 0) {
      console.log(`[ResourceManager] Processed ${processedCount} queued requests`);
    }
    
    // Check for old requests in the queue
    const oldRequestThreshold = now - 60000; // 1 minute
    const oldRequests = this.queue.filter(r => r.timestamp < oldRequestThreshold);
    
    if (oldRequests.length > 0) {
      console.warn(`[ResourceManager] ${oldRequests.length} requests in queue are older than 1 minute`);
    }
  }
  
  /**
   * Reset token usage counter
   */
  private resetTokenUsage(): void {
    this.tokenUsage = {
      lastMinute: 0,
      lastReset: Date.now()
    };
  }
  
  /**
   * Check if resources are available for a specific type
   */
  public checkAvailableResources(type: ResourceType): boolean {
    // Check if the system is overloaded
    if (this.isSystemOverloaded()) {
      return false;
    }
    
    // Get current allocations for this type
    const currentAllocations = Array.from(this.allocations.values())
      .filter(a => a.type === type)
      .length;
    
    // Get thresholds for this type
    const thresholds = this.currentProfile.thresholds[type];
    
    // Check if we're at the concurrent operation limit
    if (currentAllocations >= thresholds.max_concurrent_operations) {
      return false;
    }
    
    // Check token rate limit
    if (this.tokenUsage.lastMinute >= this.currentProfile.max_token_rate_per_minute) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Check if the system is overloaded
   */
  private isSystemOverloaded(): boolean {
    const { cpuUsage, memoryUsagePercent } = this.systemMetrics;
    const systemThresholds = this.currentProfile.thresholds['system'];
    
    // Check CPU usage
    if (cpuUsage > systemThresholds.cpu_usage_percentage) {
      return true;
    }
    
    // Check memory usage
    if (memoryUsagePercent > systemThresholds.memory_usage_percentage) {
      return true;
    }
    
    return false;
  }
  
  /**
   * Allocate resources for a specific type
   */
  public allocateResources(
    type: ResourceType,
    request: ResourceAllocationRequest = {}
  ): string {
    const id = `${type}-${Date.now()}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`;
    
    // Check if we can allocate resources immediately
    if (this.canAllocateResources(type)) {
      // Allocate resources
      this.allocations.set(id, {
        id,
        type,
        allocatedAt: Date.now(),
        estimatedDuration: request.estimated_duration_ms,
        estimatedTokens: request.estimated_tokens,
        priority: request.priority || 'normal',
        metadata: request.metadata
      });
      
      // If tokens are estimated, increment token usage
      if (request.estimated_tokens) {
        this.tokenUsage.lastMinute += request.estimated_tokens;
      }
      
      return id;
    }
    
    // If resources aren't available, add to queue (if space allows)
    if (this.queue.length < this.currentProfile.queue_capacity) {
      this.queue.push({
        type,
        request,
        timestamp: Date.now(),
        id
      });
      
      // Sort queue by priority (high > normal > low)
      this.queue.sort((a, b) => {
        const priorityOrder = { high: 0, normal: 1, low: 2 };
        return (priorityOrder[a.request.priority || 'normal'] - priorityOrder[b.request.priority || 'normal']);
      });
      
      console.log(`[ResourceManager] Request ${id} queued for ${type} (queue length: ${this.queue.length})`);
      
      return id;
    }
    
    // If queue is full, throw an error
    throw errorHandler.createError(
      'Resource queue is full',
      ErrorCategory.RESOURCE_LIMIT,
      { type, queueLength: this.queue.length }
    );
  }
  
  /**
   * Check if resources can be allocated for a specific type
   */
  private canAllocateResources(type: ResourceType): boolean {
    // Get current allocations for this type
    const currentAllocations = Array.from(this.allocations.values())
      .filter(a => a.type === type)
      .length;
    
    // Get thresholds for this type
    const thresholds = this.currentProfile.thresholds[type];
    
    // Check if we're at the concurrent operation limit
    if (currentAllocations >= thresholds.max_concurrent_operations) {
      return false;
    }
    
    // Check if the system is overloaded
    if (this.isSystemOverloaded()) {
      return false;
    }
    
    // Check token rate limit
    if (this.tokenUsage.lastMinute >= thresholds.token_rate_limit) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Release allocated resources
   */
  public releaseResources(idOrType: string | ResourceType): void {
    // If it's a resource type, release the oldest allocation of that type
    if (['llama3_inference', 'gemma3_inference', 'hybrid_inference', 'reasoning', 'system'].includes(idOrType)) {
      const type = idOrType as ResourceType;
      const allocationsOfType = Array.from(this.allocations.entries())
        .filter(([_, a]) => a.type === type)
        .sort(([_, a], [__, b]) => a.allocatedAt - b.allocatedAt);
      
      if (allocationsOfType.length > 0) {
        const [id] = allocationsOfType[0];
        this.allocations.delete(id);
      }
    } else {
      // Otherwise, release a specific allocation by ID
      this.allocations.delete(idOrType);
    }
  }
  
  /**
   * Get current resource status
   */
  public getResourceStatus(): ResourceStatus {
    // Count allocations by type
    const allocationsByType = Array.from(this.allocations.values()).reduce((acc, allocation) => {
      acc[allocation.type] = (acc[allocation.type] || 0) + 1;
      return acc;
    }, {} as Record<ResourceType, number>);
    
    // Ensure all resource types are represented
    const currentAllocations: Record<ResourceType, number> = {
      'llama3_inference': 0,
      'gemma3_inference': 0,
      'hybrid_inference': 0,
      'reasoning': 0,
      'system': 0,
      ...allocationsByType
    };
    
    // Calculate available capacity as a percentage
    const systemThresholds = this.currentProfile.thresholds['system'];
    const cpuCapacity = Math.max(0, (systemThresholds.cpu_usage_percentage - this.systemMetrics.cpuUsage) / systemThresholds.cpu_usage_percentage);
    const memoryCapacity = Math.max(0, (systemThresholds.memory_usage_percentage - this.systemMetrics.memoryUsagePercent) / systemThresholds.memory_usage_percentage);
    const availableCapacity = Math.min(cpuCapacity, memoryCapacity) * 100;
    
    return {
      system_load: {
        cpu_usage: this.systemMetrics.cpuUsage,
        memory_usage: this.systemMetrics.memoryUsagePercent,
        memory_available_gb: this.systemMetrics.availableMemoryGB,
        total_memory_gb: this.systemMetrics.totalMemoryGB
      },
      current_allocations: currentAllocations,
      pending_requests: this.queue.length,
      queue_length: this.queue.length,
      available_capacity: Math.round(availableCapacity),
      gpu_available: false // Update if GPU detection is implemented
    };
  }
}

// Export singleton instance
export const resourceManager = ResourceManager.getInstance();