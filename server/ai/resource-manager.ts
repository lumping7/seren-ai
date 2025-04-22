/**
 * Adaptive Resource Management System
 * 
 * Provides dynamic allocation and optimization of system resources
 * based on workload, available hardware, and request priority.
 * Implements adaptive throttling and graceful degradation under load.
 */

import os from 'os';

// Resource types that can be managed
type ResourceType = 'memory' | 'cpu' | 'gpu' | 'tokens' | 'requests';

// Operation types that require resource allocation
type OperationType = 
  | 'llama3_inference' 
  | 'gemma3_inference'
  | 'hybrid_inference'
  | 'reasoning_operation'
  | 'knowledge_retrieval'
  | 'database_operation';

// Priority levels for resource allocation
type PriorityLevel = 'low' | 'normal' | 'high' | 'critical';

// Resource profiles for different hardware environments
type ResourceProfile = 'minimal' | 'standard' | 'enhanced' | 'dedicated';

// Resource allocation options
interface AllocationOptions {
  estimated_tokens?: number;
  expected_duration_ms?: number;
  priority?: PriorityLevel;
  timeout_ms?: number;
  metadata?: Record<string, any>;
}

// Resource status for monitoring
interface ResourceStatus {
  total: number;
  used: number;
  available: number;
  reserved: number;
  utilization: number;
}

// Active allocation tracking
interface ActiveAllocation {
  operationType: OperationType;
  startTime: number;
  resources: Record<ResourceType, number>;
  priority: PriorityLevel;
  metadata?: Record<string, any>;
}

/**
 * Resource Manager Class
 * 
 * Handles dynamic allocation and optimization of system resources
 */
class ResourceManager {
  private static instance: ResourceManager;
  
  // System resources
  private totalMemory: number; // In MB
  private availableMemory: number; // In MB
  private cpuCores: number;
  private gpuAvailable: boolean;
  
  // Active allocations and thresholds
  private activeAllocations: Map<string, ActiveAllocation> = new Map();
  private resourceProfile: ResourceProfile;
  private utilizationThresholds: Record<ResourceType, number>;
  private priorityWeights: Record<PriorityLevel, number>;
  
  // Operation-specific resource requirements
  private operationRequirements: Record<OperationType, Record<ResourceType, number>> = {
    llama3_inference: {
      memory: 1024, // Base memory requirement in MB
      cpu: 2,       // CPU cores
      gpu: 0,       // GPU memory in MB (if available)
      tokens: 1,    // Token multiplier
      requests: 1   // Concurrent request count
    },
    gemma3_inference: {
      memory: 1024,
      cpu: 2,
      gpu: 0,
      tokens: 1,
      requests: 1
    },
    hybrid_inference: {
      memory: 2048,
      cpu: 4,
      gpu: 0,
      tokens: 2,
      requests: 1
    },
    reasoning_operation: {
      memory: 512,
      cpu: 1,
      gpu: 0,
      tokens: 0.5,
      requests: 1
    },
    knowledge_retrieval: {
      memory: 256,
      cpu: 1,
      gpu: 0,
      tokens: 0.2,
      requests: 1
    },
    database_operation: {
      memory: 128,
      cpu: 0.5,
      gpu: 0,
      tokens: 0.1,
      requests: 1
    }
  };
  
  /**
   * Private constructor to enforce singleton pattern
   */
  private constructor() {
    // Get system information
    this.totalMemory = Math.floor(os.totalmem() / (1024 * 1024)); // Convert to MB
    this.availableMemory = Math.floor(os.freemem() / (1024 * 1024)); // Convert to MB
    this.cpuCores = os.cpus().length;
    
    // Detect if GPU is present (simplified simulation)
    this.gpuAvailable = false; // In a real system, use proper GPU detection
    
    // Set resource profile based on available hardware
    this.resourceProfile = this.determineResourceProfile();
    
    // Configure thresholds based on profile
    this.utilizationThresholds = this.configureThresholds();
    
    // Configure priority weights
    this.priorityWeights = {
      low: 0.5,
      normal: 1.0,
      high: 2.0,
      critical: 4.0
    };
    
    console.log(`[ResourceManager] Initialized with ${this.resourceProfile} profile`);
    console.log(`[ResourceManager] System has ${Math.round(this.totalMemory / 1024)}GB RAM and ${this.cpuCores} CPU cores`);
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
   * Determine the resource profile based on hardware
   */
  private determineResourceProfile(): ResourceProfile {
    // Determine profile based on available hardware
    if (this.totalMemory < 4 * 1024) { // Less than 4GB RAM
      return 'minimal';
    } else if (this.totalMemory < 16 * 1024) { // Less than 16GB RAM
      return 'standard';
    } else if (this.totalMemory < 64 * 1024) { // Less than 64GB RAM
      return 'enhanced';
    } else {
      return 'dedicated';
    }
  }
  
  /**
   * Configure utilization thresholds based on profile
   */
  private configureThresholds(): Record<ResourceType, number> {
    // Default thresholds
    const defaults = {
      memory: 0.85, // 85% memory utilization
      cpu: 0.9,     // 90% CPU utilization
      gpu: 0.95,    // 95% GPU utilization
      tokens: 0.8,  // 80% token rate limit
      requests: 0.9 // 90% request rate limit
    };
    
    // Adjust based on profile
    switch (this.resourceProfile) {
      case 'minimal':
        return {
          memory: 0.7,
          cpu: 0.8,
          gpu: 0.9,
          tokens: 0.6,
          requests: 0.7
        };
      case 'standard':
        return defaults;
      case 'enhanced':
        return {
          memory: 0.9,
          cpu: 0.95,
          gpu: 0.98,
          tokens: 0.85,
          requests: 0.95
        };
      case 'dedicated':
        return {
          memory: 0.95,
          cpu: 0.98,
          gpu: 0.99,
          tokens: 0.9,
          requests: 0.98
        };
    }
  }
  
  /**
   * Check if sufficient resources are available for an operation
   */
  public checkAvailableResources(operationType: OperationType, options?: AllocationOptions): boolean {
    const requirements = this.calculateResourceRequirements(operationType, options);
    const currentUtilization = this.getCurrentUtilization();
    
    // Check each resource type
    for (const [resource, required] of Object.entries(requirements) as [ResourceType, number][]) {
      const available = this.getAvailableResource(resource);
      if (required > available) {
        console.log(`[ResourceManager] Insufficient ${resource} for ${operationType}: requires ${required}, available ${available}`);
        return false;
      }
      
      // Check if allocation would exceed threshold
      const currentUsed = this.getUsedResource(resource);
      const potentialUtilization = (currentUsed + required) / this.getTotalResource(resource);
      if (potentialUtilization > this.utilizationThresholds[resource]) {
        console.log(`[ResourceManager] ${resource} threshold would be exceeded for ${operationType}: ${potentialUtilization.toFixed(2)} > ${this.utilizationThresholds[resource]}`);
        return false;
      }
    }
    
    return true;
  }
  
  /**
   * Allocate resources for an operation
   */
  public allocateResources(operationType: OperationType, options?: AllocationOptions): string {
    const allocationId = `${operationType}-${Date.now()}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`;
    const requirements = this.calculateResourceRequirements(operationType, options);
    const priority = options?.priority || 'normal';
    
    // Create allocation record
    this.activeAllocations.set(allocationId, {
      operationType,
      startTime: Date.now(),
      resources: requirements,
      priority,
      metadata: options?.metadata
    });
    
    // Set timeout to auto-release if specified
    if (options?.timeout_ms) {
      setTimeout(() => {
        if (this.activeAllocations.has(allocationId)) {
          console.log(`[ResourceManager] Auto-releasing timed out allocation: ${allocationId}`);
          this.releaseResources(allocationId);
        }
      }, options.timeout_ms);
    }
    
    return allocationId;
  }
  
  /**
   * Release resources for an operation
   */
  public releaseResources(allocationIdOrType: string): boolean {
    // Check if it's an allocation ID
    if (this.activeAllocations.has(allocationIdOrType)) {
      this.activeAllocations.delete(allocationIdOrType);
      return true;
    }
    
    // Check if it's an operation type and release all of that type
    let found = false;
    for (const [id, allocation] of this.activeAllocations.entries()) {
      if (allocation.operationType === allocationIdOrType) {
        this.activeAllocations.delete(id);
        found = true;
      }
    }
    
    return found;
  }
  
  /**
   * Calculate resource requirements for an operation
   */
  private calculateResourceRequirements(
    operationType: OperationType, 
    options?: AllocationOptions
  ): Record<ResourceType, number> {
    const baseRequirements = this.operationRequirements[operationType];
    const result: Record<ResourceType, number> = { ...baseRequirements };
    
    // Scale memory based on estimated tokens
    if (options?.estimated_tokens) {
      // Assumption: 1KB of memory per token as a rough estimate
      const tokenMemory = options.estimated_tokens * 1; // 1KB per token
      result.memory += Math.ceil(tokenMemory / 1024); // Convert to MB
    }
    
    // Scale based on priority
    if (options?.priority) {
      const priorityMultiplier = this.priorityWeights[options.priority];
      // Priority affects CPU allocation more than memory
      result.cpu *= priorityMultiplier;
    }
    
    // If in minimal profile, reduce all resource requirements
    if (this.resourceProfile === 'minimal') {
      result.memory = Math.ceil(result.memory * 0.7);
      result.cpu = Math.ceil(result.cpu * 0.7);
    }
    
    return result;
  }
  
  /**
   * Get current utilization of resources
   */
  public getCurrentUtilization(): Record<ResourceType, number> {
    return {
      memory: this.getUsedResource('memory') / this.getTotalResource('memory'),
      cpu: this.getUsedResource('cpu') / this.getTotalResource('cpu'),
      gpu: this.getUsedResource('gpu') / this.getTotalResource('gpu'),
      tokens: this.getUsedResource('tokens') / this.getTotalResource('tokens'),
      requests: this.getUsedResource('requests') / this.getTotalResource('requests')
    };
  }
  
  /**
   * Get total amount of a resource
   */
  private getTotalResource(resource: ResourceType): number {
    switch (resource) {
      case 'memory':
        return this.totalMemory;
      case 'cpu':
        return this.cpuCores;
      case 'gpu':
        return this.gpuAvailable ? 1024 : 0; // Simplified value
      case 'tokens':
        // Token rate limit depends on profile
        switch (this.resourceProfile) {
          case 'minimal': return 10000;
          case 'standard': return 50000;
          case 'enhanced': return 200000;
          case 'dedicated': return 1000000;
        }
      case 'requests':
        // Request concurrency depends on profile
        switch (this.resourceProfile) {
          case 'minimal': return 10;
          case 'standard': return 50;
          case 'enhanced': return 200;
          case 'dedicated': return 500;
        }
    }
  }
  
  /**
   * Get used amount of a resource
   */
  private getUsedResource(resource: ResourceType): number {
    // Sum all active allocations
    let used = 0;
    for (const allocation of this.activeAllocations.values()) {
      used += allocation.resources[resource] || 0;
    }
    
    // For memory, also factor in system usage
    if (resource === 'memory') {
      const systemUsed = this.totalMemory - this.availableMemory;
      used = Math.max(used, systemUsed); // Take the larger value
    }
    
    return used;
  }
  
  /**
   * Get available amount of a resource
   */
  private getAvailableResource(resource: ResourceType): number {
    const total = this.getTotalResource(resource);
    const used = this.getUsedResource(resource);
    return Math.max(0, total - used);
  }
  
  /**
   * Get resource status for monitoring
   */
  public getResourceStatus(): Record<ResourceType, ResourceStatus> {
    const result: Record<ResourceType, ResourceStatus> = {} as any;
    
    for (const resource of ['memory', 'cpu', 'gpu', 'tokens', 'requests'] as ResourceType[]) {
      const total = this.getTotalResource(resource);
      const used = this.getUsedResource(resource);
      const available = this.getAvailableResource(resource);
      
      // Calculate reserved (allocated but not necessarily used)
      let reserved = 0;
      for (const allocation of this.activeAllocations.values()) {
        reserved += allocation.resources[resource] || 0;
      }
      
      result[resource] = {
        total,
        used,
        available,
        reserved,
        utilization: total > 0 ? used / total : 0
      };
    }
    
    return result;
  }
  
  /**
   * Optimize resource allocation based on load
   */
  public optimizeResourceAllocation(): void {
    const currentUtilization = this.getCurrentUtilization();
    
    // Check if we're approaching resource limits
    const approachingLimit = Object.entries(currentUtilization).some(
      ([resource, utilization]) => utilization > this.utilizationThresholds[resource as ResourceType] * 0.9
    );
    
    if (approachingLimit) {
      console.log('[ResourceManager] Approaching resource limits, optimizing allocations...');
      
      // Implement priority-based scaling - reduce resources for low-priority tasks
      for (const [id, allocation] of this.activeAllocations.entries()) {
        if (allocation.priority === 'low') {
          // Reduce CPU allocation for low-priority tasks
          allocation.resources.cpu = Math.max(0.5, allocation.resources.cpu * 0.8);
        }
      }
    }
  }
  
  /**
   * Get active allocations count by operation type
   */
  public getActiveAllocationsByType(): Record<OperationType, number> {
    const result: Partial<Record<OperationType, number>> = {};
    
    for (const allocation of this.activeAllocations.values()) {
      const type = allocation.operationType;
      result[type] = (result[type] || 0) + 1;
    }
    
    return result as Record<OperationType, number>;
  }
}

export const resourceManager = ResourceManager.getInstance();