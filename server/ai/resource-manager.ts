/**
 * Resource Management System
 * 
 * Enables the AI system to adapt to different server environments
 * from 16GB RAM servers to high-end infrastructure by dynamically
 * adjusting parameters based on available resources.
 */

import os from 'os';

// Resource profile types
export type ResourceProfile = 'minimal' | 'standard' | 'enhanced' | 'unlimited';

// System resource limits based on profile
export interface ResourceLimits {
  maxContextSize: number;
  maxTokens: number;
  maxConcurrentRequests: number;
  maxProcessingTime: number;
  enableParallelInference: boolean;
  useQuantization: boolean;
}

// Current system resource usage
export interface SystemResources {
  totalMemory: number;
  availableMemory: number;
  cpuCount: number;
  cpuUsage: number;
  activeRequests: number;
}

/**
 * ResourceManager class
 * 
 * Handles dynamic resource allocation and adaptation
 * based on the server environment.
 */
export class ResourceManager {
  private static instance: ResourceManager;
  private activeRequests: number = 0;
  private lastCpuUsage: { user: number; system: number; idle: number } = { user: 0, system: 0, idle: 0 };
  private cpuUsagePercentage: number = 0;
  private profile: ResourceProfile;
  
  // Resource limit profiles
  private readonly resourceProfiles: Record<ResourceProfile, ResourceLimits> = {
    minimal: {
      maxContextSize: 4096,
      maxTokens: 1024,
      maxConcurrentRequests: 2,
      maxProcessingTime: 10000, // 10 seconds
      enableParallelInference: false,
      useQuantization: true
    },
    standard: {
      maxContextSize: 8192,
      maxTokens: 2048,
      maxConcurrentRequests: 5,
      maxProcessingTime: 15000, // 15 seconds
      enableParallelInference: true,
      useQuantization: true
    },
    enhanced: {
      maxContextSize: 16384,
      maxTokens: 4096,
      maxConcurrentRequests: 10,
      maxProcessingTime: 20000, // 20 seconds
      enableParallelInference: true,
      useQuantization: false
    },
    unlimited: {
      maxContextSize: 32768,
      maxTokens: 8192,
      maxConcurrentRequests: 30,
      maxProcessingTime: 30000, // 30 seconds
      enableParallelInference: true,
      useQuantization: false
    }
  };
  
  /**
   * Private constructor - use getInstance()
   */
  private constructor() {
    // Determine the appropriate resource profile based on system capabilities
    this.profile = this.determineResourceProfile();
    
    // Initialize CPU usage tracking
    this.updateCpuUsage();
    
    // Set up CPU usage monitoring interval
    setInterval(() => this.updateCpuUsage(), 5000);
    
    console.log(`[ResourceManager] Initialized with ${this.profile} profile`);
    console.log(`[ResourceManager] System has ${Math.round(os.totalmem() / (1024 * 1024 * 1024))}GB RAM and ${os.cpus().length} CPU cores`);
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
   * Determine the appropriate resource profile based on system capabilities
   */
  private determineResourceProfile(): ResourceProfile {
    const totalMemoryGB = os.totalmem() / (1024 * 1024 * 1024);
    const cpuCount = os.cpus().length;
    
    // Determine profile based on available RAM and CPU
    if (totalMemoryGB >= 64) {
      return cpuCount >= 8 ? 'unlimited' : 'enhanced';
    } else if (totalMemoryGB >= 32) {
      return 'enhanced';
    } else if (totalMemoryGB >= 16) {
      return 'standard';
    } else {
      return 'minimal';
    }
  }
  
  /**
   * Update CPU usage statistics
   */
  private updateCpuUsage(): void {
    const cpuInfo = os.cpus();
    
    // Calculate CPU times
    let user = 0, system = 0, idle = 0;
    
    for (const cpu of cpuInfo) {
      user += cpu.times.user;
      system += cpu.times.sys;
      idle += cpu.times.idle;
    }
    
    // Calculate CPU usage percentage based on delta since last check
    if (this.lastCpuUsage.user > 0) {
      const userDelta = user - this.lastCpuUsage.user;
      const systemDelta = system - this.lastCpuUsage.system;
      const idleDelta = idle - this.lastCpuUsage.idle;
      const totalDelta = userDelta + systemDelta + idleDelta;
      
      this.cpuUsagePercentage = totalDelta > 0 ? 
        ((userDelta + systemDelta) / totalDelta) * 100 : 0;
    }
    
    // Update last CPU usage
    this.lastCpuUsage = { user, system, idle };
  }
  
  /**
   * Get the current resource profile
   */
  public getProfile(): ResourceProfile {
    return this.profile;
  }
  
  /**
   * Get the current resource limits
   */
  public getLimits(): ResourceLimits {
    return this.resourceProfiles[this.profile];
  }
  
  /**
   * Get current system resource usage
   */
  public getSystemResources(): SystemResources {
    return {
      totalMemory: os.totalmem(),
      availableMemory: os.freemem(),
      cpuCount: os.cpus().length,
      cpuUsage: this.cpuUsagePercentage,
      activeRequests: this.activeRequests
    };
  }
  
  /**
   * Register the start of a new request
   */
  public startRequest(): void {
    this.activeRequests++;
    
    // Dynamically downgrade profile if system is under heavy load
    this.checkForProfileDowngrade();
    
    // Log system load
    if (this.activeRequests % 5 === 0) {
      const resources = this.getSystemResources();
      console.log(`[ResourceManager] System load: ${this.activeRequests} active requests, ${Math.round(this.cpuUsagePercentage)}% CPU usage, ${Math.round(resources.availableMemory / (1024 * 1024 * 1024))}GB free memory`);
    }
  }
  
  /**
   * Register the end of a request
   */
  public endRequest(): void {
    this.activeRequests = Math.max(0, this.activeRequests - 1);
    
    // Check if we can upgrade profile again
    this.checkForProfileUpgrade();
  }
  
  /**
   * Check if the system can process a new request
   */
  public canProcessRequest(): boolean {
    const limits = this.getLimits();
    return this.activeRequests < limits.maxConcurrentRequests;
  }
  
  /**
   * Get the maximum context size for the current profile
   */
  public getMaxContextSize(): number {
    return this.getLimits().maxContextSize;
  }
  
  /**
   * Get the maximum tokens for the current profile
   */
  public getMaxTokens(): number {
    return this.getLimits().maxTokens;
  }
  
  /**
   * Get whether parallel inference is enabled for the current profile
   */
  public isParallelInferenceEnabled(): boolean {
    return this.getLimits().enableParallelInference;
  }
  
  /**
   * Get whether model quantization is enabled for the current profile
   */
  public isQuantizationEnabled(): boolean {
    return this.getLimits().useQuantization;
  }
  
  /**
   * Check if we need to downgrade the profile due to resource constraints
   */
  private checkForProfileDowngrade(): void {
    const resources = this.getSystemResources();
    const freeMemoryGB = resources.availableMemory / (1024 * 1024 * 1024);
    
    // Check if we need to downgrade based on memory pressure
    if (this.profile === 'unlimited' && (freeMemoryGB < 16 || this.cpuUsagePercentage > 85)) {
      this.profile = 'enhanced';
      console.log('[ResourceManager] Downgraded to enhanced profile due to resource constraints');
    } else if (this.profile === 'enhanced' && (freeMemoryGB < 8 || this.cpuUsagePercentage > 90)) {
      this.profile = 'standard';
      console.log('[ResourceManager] Downgraded to standard profile due to resource constraints');
    } else if (this.profile === 'standard' && (freeMemoryGB < 4 || this.cpuUsagePercentage > 95)) {
      this.profile = 'minimal';
      console.log('[ResourceManager] Downgraded to minimal profile due to resource constraints');
    }
  }
  
  /**
   * Check if we can upgrade the profile due to available resources
   */
  private checkForProfileUpgrade(): void {
    // Only check for upgrade if active requests are low
    if (this.activeRequests > 2) return;
    
    const resources = this.getSystemResources();
    const freeMemoryGB = resources.availableMemory / (1024 * 1024 * 1024);
    const totalMemoryGB = resources.totalMemory / (1024 * 1024 * 1024);
    
    // Check if we can upgrade based on memory availability
    if (this.profile === 'minimal' && freeMemoryGB > 8 && this.cpuUsagePercentage < 70) {
      this.profile = 'standard';
      console.log('[ResourceManager] Upgraded to standard profile due to available resources');
    } else if (this.profile === 'standard' && freeMemoryGB > 16 && this.cpuUsagePercentage < 60) {
      this.profile = 'enhanced';
      console.log('[ResourceManager] Upgraded to enhanced profile due to available resources');
    } else if (this.profile === 'enhanced' && freeMemoryGB > 32 && totalMemoryGB >= 64 && this.cpuUsagePercentage < 50) {
      this.profile = 'unlimited';
      console.log('[ResourceManager] Upgraded to unlimited profile due to available resources');
    }
  }
  
  /**
   * Calculate appropriate temperature based on profile and request importance
   */
  public calculateTemperature(baseTemperature: number, requestImportance: number): number {
    const profile = this.getProfile();
    
    // Adjust temperature based on profile and request importance
    // Higher importance requests get closer to the requested temperature
    switch (profile) {
      case 'minimal':
        // Minimal profile uses lower temperatures for stability
        return 0.5 + (baseTemperature - 0.5) * 0.5 * requestImportance;
      case 'standard':
        // Standard profile slightly reduces temperature
        return 0.5 + (baseTemperature - 0.5) * 0.7 * requestImportance;
      case 'enhanced':
        // Enhanced profile uses closer to requested temperature
        return 0.5 + (baseTemperature - 0.5) * 0.9 * requestImportance;
      case 'unlimited':
        // Unlimited uses exactly requested temperature
        return baseTemperature;
      default:
        return baseTemperature;
    }
  }
  
  /**
   * Estimate token count for a given input
   * This is a simple approximation
   */
  public estimateTokenCount(text: string): number {
    // Simple approximation: ~4 characters per token for English text
    return Math.ceil(text.length / 4);
  }
  
  /**
   * Get appropriate batch size for parallel operations
   */
  public getBatchSize(): number {
    switch (this.profile) {
      case 'minimal': return 1; // No batching
      case 'standard': return 2;
      case 'enhanced': return 4;
      case 'unlimited': return 8;
      default: return 2;
    }
  }
  
  /**
   * Reset the resource manager (for testing)
   */
  public reset(): void {
    this.activeRequests = 0;
    this.profile = this.determineResourceProfile();
  }
}

export const resourceManager = ResourceManager.getInstance();