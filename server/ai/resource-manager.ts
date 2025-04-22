/**
 * Resource Management System
 * 
 * Provides adaptive resource allocation based on system capabilities
 * and current workload. This allows the AI system to scale from
 * modest servers (16GB RAM) to high-end infrastructure.
 */

import os from 'os';

// Resource profile types
export type ResourceProfile = 'minimal' | 'standard' | 'enhanced' | 'unlimited';

interface SystemResources {
  totalMemory: number;
  availableMemory: number;
  cpuCount: number;
  cpuUsage: number;
  activeRequests: number;
}

interface ResourceLimits {
  maxContextSize: number;
  maxTokens: number;
  maxConcurrentRequests: number;
  maxProcessingTime: number;
  enableParallelInference: boolean;
  useQuantization: boolean;
}

/**
 * Resource Manager Class
 * 
 * Handles adaptive scaling based on available system resources
 */
export class ResourceManager {
  private static instance: ResourceManager;
  private activeRequests: number = 0;
  private lastCpuUsageCheck: number = 0;
  private cpuUsage: number = 0;
  private limits: ResourceLimits;
  private profile: ResourceProfile;
  
  private constructor() {
    // Set initial resource profile based on system capabilities
    this.profile = this.determineResourceProfile();
    this.limits = this.getResourceLimits(this.profile);
    
    // Update CPU usage periodically
    setInterval(() => this.updateCpuUsage(), 60000);
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
   * Get current resource profile
   */
  public getProfile(): ResourceProfile {
    return this.profile;
  }
  
  /**
   * Get current resource limits
   */
  public getLimits(): ResourceLimits {
    return { ...this.limits };
  }
  
  /**
   * Track the start of a new request
   */
  public startRequest(): void {
    this.activeRequests++;
    
    // Re-evaluate resource profile if necessary based on load
    if (this.activeRequests % 5 === 0) {
      this.reevaluateResources();
    }
  }
  
  /**
   * Track the completion of a request
   */
  public endRequest(): void {
    this.activeRequests = Math.max(0, this.activeRequests - 1);
  }
  
  /**
   * Check if a new request can be processed based on current load
   */
  public canProcessRequest(): boolean {
    return this.activeRequests < this.limits.maxConcurrentRequests;
  }
  
  /**
   * Get the maximum context size based on current profile
   */
  public getMaxContextSize(): number {
    return this.limits.maxContextSize;
  }
  
  /**
   * Get the current system resources
   */
  public getSystemResources(): SystemResources {
    return {
      totalMemory: os.totalmem(),
      availableMemory: os.freemem(),
      cpuCount: os.cpus().length,
      cpuUsage: this.cpuUsage,
      activeRequests: this.activeRequests
    };
  }
  
  /**
   * Determine the appropriate resource profile based on system capabilities
   */
  private determineResourceProfile(): ResourceProfile {
    const totalMemGB = os.totalmem() / (1024 * 1024 * 1024);
    const cpuCount = os.cpus().length;
    
    if (totalMemGB <= 16) {
      return 'minimal';
    } else if (totalMemGB <= 32) {
      return 'standard';
    } else if (totalMemGB <= 64) {
      return 'enhanced';
    } else {
      return 'unlimited';
    }
  }
  
  /**
   * Get resource limits based on profile
   */
  private getResourceLimits(profile: ResourceProfile): ResourceLimits {
    switch (profile) {
      case 'minimal':
        return {
          maxContextSize: 4096,
          maxTokens: 1024,
          maxConcurrentRequests: 2,
          maxProcessingTime: 10000,
          enableParallelInference: false,
          useQuantization: true
        };
      case 'standard':
        return {
          maxContextSize: 8192,
          maxTokens: 2048,
          maxConcurrentRequests: 5,
          maxProcessingTime: 15000,
          enableParallelInference: true,
          useQuantization: true
        };
      case 'enhanced':
        return {
          maxContextSize: 16384,
          maxTokens: 4096,
          maxConcurrentRequests: 10,
          maxProcessingTime: 20000,
          enableParallelInference: true,
          useQuantization: false
        };
      case 'unlimited':
        return {
          maxContextSize: 32768,
          maxTokens: 8192,
          maxConcurrentRequests: 30,
          maxProcessingTime: 30000,
          enableParallelInference: true,
          useQuantization: false
        };
    }
  }
  
  /**
   * Update CPU usage measurement
   */
  private async updateCpuUsage(): Promise<void> {
    // Simple CPU usage estimation
    const cpus = os.cpus();
    let idle = 0;
    let total = 0;
    
    for (const cpu of cpus) {
      idle += cpu.times.idle;
      total += Object.values(cpu.times).reduce((a, b) => a + b, 0);
    }
    
    // Calculate CPU usage as percentage
    if (this.lastCpuUsageCheck > 0) {
      this.cpuUsage = 100 - ((idle / total) * 100);
    }
    
    this.lastCpuUsageCheck = Date.now();
  }
  
  /**
   * Re-evaluate resource profile based on current system load
   */
  private reevaluateResources(): void {
    const current = this.getSystemResources();
    const memUsagePercent = 100 - ((current.availableMemory / current.totalMemory) * 100);
    
    // Adjust profile down if system is under heavy load
    if (memUsagePercent > 90 || current.cpuUsage > 90) {
      this.downgradeProfile();
    } 
    // Adjust profile up if system has available resources
    else if (memUsagePercent < 50 && current.cpuUsage < 50) {
      this.upgradeProfile();
    }
  }
  
  /**
   * Downgrade resource profile during high load
   */
  private downgradeProfile(): void {
    const profiles: ResourceProfile[] = ['unlimited', 'enhanced', 'standard', 'minimal'];
    const currentIndex = profiles.indexOf(this.profile);
    
    if (currentIndex < profiles.length - 1) {
      this.profile = profiles[currentIndex + 1];
      this.limits = this.getResourceLimits(this.profile);
      console.log(`[ResourceManager] Downgraded to ${this.profile} profile due to high system load`);
    }
  }
  
  /**
   * Upgrade resource profile when resources are available
   */
  private upgradeProfile(): void {
    const profiles: ResourceProfile[] = ['minimal', 'standard', 'enhanced', 'unlimited'];
    const currentIndex = profiles.indexOf(this.profile);
    
    // Only upgrade if not at the highest profile already and the system has enough resources
    if (currentIndex < profiles.length - 1) {
      const totalMemGB = os.totalmem() / (1024 * 1024 * 1024);
      const nextProfile = profiles[currentIndex + 1];
      
      // Check if system has resources to support next profile
      if ((nextProfile === 'standard' && totalMemGB > 16) ||
          (nextProfile === 'enhanced' && totalMemGB > 32) ||
          (nextProfile === 'unlimited' && totalMemGB > 64)) {
        this.profile = nextProfile;
        this.limits = this.getResourceLimits(this.profile);
        console.log(`[ResourceManager] Upgraded to ${this.profile} profile due to available system resources`);
      }
    }
  }
}

// Singleton instance
export const resourceManager = ResourceManager.getInstance();