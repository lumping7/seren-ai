/**
 * AI Model Integration System
 * 
 * This module provides the integration layer between the NodeJS backend and the Python-based
 * AI models (Qwen2.5-7b-omni and OlympicCoder-7B). It handles communication, process management,
 * and failover between the models to create a true autonomous dev team.
 */

import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs/promises';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';
import { errorHandler } from './error-handler';
import { resourceManager } from './resource-manager';
import { performanceMonitor } from './performance-monitor';
import { EventEmitter } from 'events';

// ----------------------------------------------------------------------
// Configuration & Types
// ----------------------------------------------------------------------

// Model configuration
export enum ModelType {
  QWEN_OMNI = 'qwen2.5-7b-omni',
  OLYMPIC_CODER = 'olympiccoder-7b',
  HYBRID = 'hybrid'
}

// Roles in the autonomous dev team
export enum DevTeamRole {
  ARCHITECT = 'architect',  // System design and architecture planning
  BUILDER = 'builder',      // Implementation and coding
  TESTER = 'tester',        // Testing and quality assurance
  REVIEWER = 'reviewer'     // Code review and enhancement
}

// Messages between TypeScript and Python processes
export interface ModelMessage {
  id: string;
  type: 'request' | 'response' | 'error' | 'status' | 'heartbeat';
  model: ModelType;
  role: DevTeamRole;
  content: any;
  timestamp: number;
}

// Model process status
export enum ModelStatus {
  INITIALIZING = 'initializing',
  READY = 'ready',
  BUSY = 'busy',
  FAILING = 'failing',
  OFFLINE = 'offline'
}

// Model process info
interface ModelProcess {
  model: ModelType;
  process: ChildProcess | null;
  status: ModelStatus;
  lastHeartbeat: number;
  pendingRequests: Map<string, {
    resolve: (value: any) => void;
    reject: (reason: any) => void;
    timer: NodeJS.Timeout;
    startTime: number;
  }>;
  stats: {
    requestsHandled: number;
    failedRequests: number;
    averageResponseTime: number;
    lastError: string | null;
    memoryUsage: number;
    startTime: number;
  };
}

// Internal configuration
const CONFIG = {
  modelBasePath: path.join(process.cwd(), 'ai_core'),
  pythonPath: 'python3',
  launchScript: 'model_server.py',
  heartbeatInterval: 10000, // 10 seconds
  requestTimeout: 300000, // 5 minutes
  restartThreshold: 3, // Number of failures before restart
  maxRestarts: 5, // Maximum number of restarts before switching to backup strategy
  logModelOutput: process.env.NODE_ENV !== 'production' // Log model output in dev mode
};

// ----------------------------------------------------------------------
// Model Process Management
// ----------------------------------------------------------------------

// Event emitter for model events
export const modelEvents = new EventEmitter();

// Map of active model processes
const modelProcesses = new Map<ModelType, ModelProcess>();

// Initialize model processes
export async function initializeModelProcesses(): Promise<boolean> {
  performanceMonitor.startOperation('model_initialization', 'init');
  console.log('[ModelIntegration] Initializing model processes...');
  
  try {
    // Check if model directory exists
    await checkModelEnvironment();
    
    // Start Qwen model process
    await startModelProcess(ModelType.QWEN_OMNI);
    
    // Start Olympic model process
    await startModelProcess(ModelType.OLYMPIC_CODER);
    
    // Register cleanup handlers
    process.on('SIGTERM', cleanupModelProcesses);
    process.on('SIGINT', cleanupModelProcesses);
    process.on('exit', cleanupModelProcesses);
    
    // Schedule heartbeat checks
    setInterval(checkModelHeartbeats, CONFIG.heartbeatInterval);
    
    console.log('[ModelIntegration] Model processes initialized successfully');
    performanceMonitor.endOperation('model_initialization', false);
    
    return true;
  } catch (error) {
    console.error('[ModelIntegration] Failed to initialize model processes:', error);
    performanceMonitor.endOperation('model_initialization', true);
    errorHandler.handleError(error);
    
    return false;
  }
}

/**
 * Check if the model environment is properly set up
 */
async function checkModelEnvironment(): Promise<void> {
  try {
    // Check if model base path exists
    const baseDirStat = await fs.stat(CONFIG.modelBasePath).catch(() => null);
    
    if (!baseDirStat || !baseDirStat.isDirectory()) {
      throw new Error(`Model base path does not exist: ${CONFIG.modelBasePath}`);
    }
    
    // Check if model files exist
    const requiredFiles = [
      'model_communication.py',
      'neurosymbolic_reasoning.py',
      'liquid_neural_network.py',
      'model_server.py',
      'metacognition.py'
    ];
    
    for (const file of requiredFiles) {
      const filePath = path.join(CONFIG.modelBasePath, file);
      const fileStat = await fs.stat(filePath).catch(() => null);
      
      if (!fileStat || !fileStat.isFile()) {
        throw new Error(`Required model file does not exist: ${filePath}`);
      }
    }
    
    console.log('[ModelIntegration] Model environment check passed');
  } catch (error) {
    console.error('[ModelIntegration] Model environment check failed:', error);
    throw new Error(`Model environment check failed: ${error.message}`);
  }
}

/**
 * Start a model process
 */
async function startModelProcess(model: ModelType): Promise<void> {
  // Clean up any existing process for this model
  if (modelProcesses.has(model)) {
    await stopModelProcess(model);
  }
  
  console.log(`[ModelIntegration] Starting ${model} process...`);
  
  // Create process tracking object
  const modelProcess: ModelProcess = {
    model,
    process: null,
    status: ModelStatus.INITIALIZING,
    lastHeartbeat: Date.now(),
    pendingRequests: new Map(),
    stats: {
      requestsHandled: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      lastError: null,
      memoryUsage: 0,
      startTime: Date.now()
    }
  };
  
  // Configure environment variables
  const env = {
    ...process.env,
    MODEL_TYPE: model,
    MODEL_BASE_PATH: CONFIG.modelBasePath,
    LOG_LEVEL: process.env.NODE_ENV === 'production' ? 'INFO' : 'DEBUG',
    PORT: model === ModelType.QWEN_OMNI ? '50051' : '50052' // Use different ports for each model
  };
  
  // Spawn the Python process
  const scriptPath = path.join(CONFIG.modelBasePath, CONFIG.launchScript);
  
  try {
    const process = spawn(CONFIG.pythonPath, [scriptPath], {
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: false
    });
    
    modelProcess.process = process;
    
    // Handle process exit
    process.on('exit', (code, signal) => {
      console.error(`[ModelIntegration] ${model} process exited with code ${code} and signal ${signal}`);
      
      // Update status
      modelProcess.status = ModelStatus.OFFLINE;
      
      // Reject any pending requests
      for (const [id, request] of modelProcess.pendingRequests.entries()) {
        clearTimeout(request.timer);
        request.reject(new Error(`Model process exited unexpectedly (code: ${code}, signal: ${signal})`));
        modelProcess.pendingRequests.delete(id);
      }
      
      // Attempt restart if not shutting down
      const globalRegistry = global as any;
      if (!globalRegistry.__SERVER_SHUTDOWN) {
        handleModelFailure(model);
      }
    });
    
    // Handle process errors
    process.on('error', (error) => {
      console.error(`[ModelIntegration] ${model} process error:`, error);
      modelProcess.status = ModelStatus.FAILING;
      modelProcess.stats.lastError = error.message;
      
      handleModelFailure(model);
    });
    
    // Process stdout (for parsing structured output)
    process.stdout.on('data', (data) => {
      const output = data.toString().trim();
      
      if (CONFIG.logModelOutput) {
        console.log(`[${model}] ${output}`);
      }
      
      // Try to parse as JSON
      try {
        if (output.startsWith('{') && output.endsWith('}')) {
          const message = JSON.parse(output) as ModelMessage;
          
          // Handle different message types
          handleModelMessage(model, message);
        }
      } catch (error) {
        // Non-JSON output, likely just logging
      }
    });
    
    // Process stderr (for errors and warnings)
    process.stderr.on('data', (data) => {
      const output = data.toString().trim();
      
      if (CONFIG.logModelOutput || output.toLowerCase().includes('error')) {
        console.error(`[${model}] ERROR: ${output}`);
      }
      
      modelProcess.stats.lastError = output;
    });
    
    // Store the process in the registry
    modelProcesses.set(model, modelProcess);
    
    // Wait for the model to be ready
    await waitForModelReady(model);
    
    console.log(`[ModelIntegration] ${model} process started successfully`);
  } catch (error) {
    console.error(`[ModelIntegration] Failed to start ${model} process:`, error);
    
    // Clean up
    if (modelProcess.process) {
      modelProcess.process.kill();
      modelProcess.process = null;
    }
    
    // Update status
    modelProcess.status = ModelStatus.OFFLINE;
    
    throw error;
  }
}

/**
 * Wait for a model to be ready
 */
async function waitForModelReady(model: ModelType, timeout = 120000): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const startTime = Date.now();
    
    // Set timeout
    const timeoutId = setTimeout(() => {
      reject(new Error(`Timed out waiting for ${model} to be ready`));
    }, timeout);
    
    // Check status periodically
    const checkInterval = setInterval(() => {
      const modelProcess = modelProcesses.get(model);
      
      if (!modelProcess) {
        clearInterval(checkInterval);
        clearTimeout(timeoutId);
        reject(new Error(`${model} process not found`));
        return;
      }
      
      if (modelProcess.status === ModelStatus.READY) {
        clearInterval(checkInterval);
        clearTimeout(timeoutId);
        resolve();
        return;
      }
      
      // Send a heartbeat to check status
      if (modelProcess.process && modelProcess.process.stdin.writable) {
        const heartbeat: ModelMessage = {
          id: uuidv4(),
          type: 'heartbeat',
          model,
          role: DevTeamRole.ARCHITECT, // Doesn't matter for heartbeat
          content: { action: 'status_check' },
          timestamp: Date.now()
        };
        
        modelProcess.process.stdin.write(JSON.stringify(heartbeat) + '\n');
      }
      
      // Log progress
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      console.log(`[ModelIntegration] Waiting for ${model} to be ready (${elapsed}s)...`);
      
    }, 3000); // Check every 3 seconds
  });
}

/**
 * Handle a failure in a model process
 */
async function handleModelFailure(model: ModelType): Promise<void> {
  const modelProcess = modelProcesses.get(model);
  
  if (!modelProcess) {
    console.error(`[ModelIntegration] Cannot handle failure for unknown model: ${model}`);
    return;
  }
  
  // Update status
  modelProcess.status = ModelStatus.FAILING;
  modelProcess.stats.failedRequests++;
  
  // Emit event
  modelEvents.emit('model:failure', {
    model,
    error: modelProcess.stats.lastError,
    timestamp: Date.now()
  });
  
  // Determine if we should restart
  const failureCount = modelProcess.stats.failedRequests;
  const uptime = Date.now() - modelProcess.stats.startTime;
  const uptimeHours = uptime / (1000 * 60 * 60);
  
  // Reset failure count if the model has been running for a while
  const effectiveFailureCount = uptimeHours > 1 ? 1 : failureCount;
  
  if (effectiveFailureCount <= CONFIG.maxRestarts) {
    console.log(`[ModelIntegration] Attempting to restart ${model} (attempt ${effectiveFailureCount}/${CONFIG.maxRestarts})...`);
    
    // Exponential backoff for restart delay
    const delay = Math.min(Math.pow(2, effectiveFailureCount - 1) * 1000, 30000);
    await new Promise(resolve => setTimeout(resolve, delay));
    
    try {
      await startModelProcess(model);
      console.log(`[ModelIntegration] Successfully restarted ${model}`);
    } catch (error) {
      console.error(`[ModelIntegration] Failed to restart ${model}:`, error);
      
      // If we've exhausted restarts, switch to backup strategy
      if (effectiveFailureCount >= CONFIG.maxRestarts) {
        console.error(`[ModelIntegration] Exhausted restart attempts for ${model}, switching to backup strategy`);
        
        // Update status
        modelProcess.status = ModelStatus.OFFLINE;
        
        // Emit event
        modelEvents.emit('model:offline', {
          model,
          error: modelProcess.stats.lastError,
          timestamp: Date.now()
        });
      }
    }
  } else {
    console.error(`[ModelIntegration] Exhausted restart attempts for ${model}, switching to backup strategy`);
    
    // Update status
    modelProcess.status = ModelStatus.OFFLINE;
    
    // Emit event
    modelEvents.emit('model:offline', {
      model,
      error: modelProcess.stats.lastError,
      timestamp: Date.now()
    });
  }
}

/**
 * Stop a model process
 */
async function stopModelProcess(model: ModelType): Promise<void> {
  const modelProcess = modelProcesses.get(model);
  
  if (!modelProcess) {
    return;
  }
  
  console.log(`[ModelIntegration] Stopping ${model} process...`);
  
  // Reject any pending requests
  for (const [id, request] of modelProcess.pendingRequests.entries()) {
    clearTimeout(request.timer);
    request.reject(new Error('Model process is shutting down'));
    modelProcess.pendingRequests.delete(id);
  }
  
  // Kill the process
  if (modelProcess.process) {
    modelProcess.process.removeAllListeners();
    
    // Send SIGTERM and wait for graceful shutdown
    modelProcess.process.kill('SIGTERM');
    
    // Wait for the process to exit (with timeout)
    try {
      await Promise.race([
        new Promise<void>((resolve) => {
          if (modelProcess.process) {
            modelProcess.process.once('exit', () => resolve());
          } else {
            resolve();
          }
        }),
        new Promise<void>((_, reject) => {
          setTimeout(() => reject(new Error('Process termination timed out')), 5000);
        })
      ]);
    } catch (error) {
      // Force kill if timeout
      if (modelProcess.process) {
        modelProcess.process.kill('SIGKILL');
      }
    }
    
    modelProcess.process = null;
  }
  
  // Update status
  modelProcess.status = ModelStatus.OFFLINE;
  
  console.log(`[ModelIntegration] ${model} process stopped`);
}

/**
 * Cleanup all model processes on shutdown
 */
async function cleanupModelProcesses(): Promise<void> {
  console.log('[ModelIntegration] Cleaning up model processes...');
  
  // Mark as shutting down
  const globalRegistry = global as any;
  globalRegistry.__SERVER_SHUTDOWN = true;
  
  // Stop all model processes
  const stopPromises = Array.from(modelProcesses.keys()).map(model => 
    stopModelProcess(model).catch(error => 
      console.error(`[ModelIntegration] Error stopping ${model}:`, error)
    )
  );
  
  // Wait for all processes to stop
  await Promise.all(stopPromises);
  
  console.log('[ModelIntegration] All model processes stopped');
}

/**
 * Check model heartbeats
 */
function checkModelHeartbeats(): void {
  const now = Date.now();
  
  for (const [model, modelProcess] of modelProcesses.entries()) {
    // Skip if offline
    if (modelProcess.status === ModelStatus.OFFLINE) {
      continue;
    }
    
    // Check heartbeat
    const heartbeatAge = now - modelProcess.lastHeartbeat;
    
    if (heartbeatAge > CONFIG.heartbeatInterval * 2) {
      console.warn(`[ModelIntegration] ${model} heartbeat is stale (${heartbeatAge}ms), checking status...`);
      
      // Send a heartbeat to check status
      if (modelProcess.process && modelProcess.process.stdin.writable) {
        const heartbeat: ModelMessage = {
          id: uuidv4(),
          type: 'heartbeat',
          model,
          role: DevTeamRole.ARCHITECT, // Doesn't matter for heartbeat
          content: { action: 'status_check' },
          timestamp: now
        };
        
        modelProcess.process.stdin.write(JSON.stringify(heartbeat) + '\n');
      } else {
        console.error(`[ModelIntegration] ${model} process is not responsive, handling failure...`);
        handleModelFailure(model);
      }
    }
  }
}

/**
 * Handle a message from a model process
 */
function handleModelMessage(model: ModelType, message: ModelMessage): void {
  const modelProcess = modelProcesses.get(model);
  
  if (!modelProcess) {
    console.error(`[ModelIntegration] Received message for unknown model: ${model}`);
    return;
  }
  
  // Update heartbeat
  modelProcess.lastHeartbeat = Date.now();
  
  // Handle message based on type
  switch (message.type) {
    case 'response':
      // Handle response to a request
      const requestId = message.id;
      const pendingRequest = modelProcess.pendingRequests.get(requestId);
      
      if (pendingRequest) {
        clearTimeout(pendingRequest.timer);
        
        // Calculate response time
        const responseTime = Date.now() - pendingRequest.startTime;
        
        // Update stats
        modelProcess.stats.requestsHandled++;
        modelProcess.stats.averageResponseTime = 
          (modelProcess.stats.averageResponseTime * (modelProcess.stats.requestsHandled - 1) + responseTime) / 
          modelProcess.stats.requestsHandled;
        
        // Update status
        modelProcess.status = ModelStatus.READY;
        
        // Resolve the promise
        pendingRequest.resolve(message.content);
        modelProcess.pendingRequests.delete(requestId);
      } else {
        console.warn(`[ModelIntegration] Received response for unknown request: ${requestId}`);
      }
      break;
      
    case 'error':
      // Handle error response
      const errorRequestId = message.id;
      const errorPendingRequest = modelProcess.pendingRequests.get(errorRequestId);
      
      if (errorPendingRequest) {
        clearTimeout(errorPendingRequest.timer);
        
        // Update stats
        modelProcess.stats.failedRequests++;
        modelProcess.stats.lastError = message.content.error;
        
        // Reject the promise
        errorPendingRequest.reject(new Error(message.content.error));
        modelProcess.pendingRequests.delete(errorRequestId);
      } else {
        console.error(`[ModelIntegration] Received error for unknown request: ${errorRequestId}`, message.content.error);
      }
      break;
      
    case 'status':
      // Handle status update
      if (message.content.status === 'ready') {
        modelProcess.status = ModelStatus.READY;
        console.log(`[ModelIntegration] ${model} is ready`);
        
        // Emit event
        modelEvents.emit('model:ready', {
          model,
          timestamp: Date.now()
        });
      } else if (message.content.status === 'busy') {
        modelProcess.status = ModelStatus.BUSY;
      }
      
      // Update memory usage if reported
      if (message.content.memory_usage) {
        modelProcess.stats.memoryUsage = message.content.memory_usage;
      }
      break;
      
    case 'heartbeat':
      // Nothing special to do for heartbeat responses
      break;
      
    default:
      console.warn(`[ModelIntegration] Received unknown message type: ${message.type}`);
  }
}

// ----------------------------------------------------------------------
// Model Communication Interface
// ----------------------------------------------------------------------

/**
 * Send a request to a model and get the response
 */
export async function sendModelRequest(
  model: ModelType,
  role: DevTeamRole,
  content: any,
  timeout = CONFIG.requestTimeout
): Promise<any> {
  const requestId = uuidv4();
  const operationId = `model_request_${requestId}`;
  
  performanceMonitor.startOperation(operationId, model);
  
  try {
    const modelProcess = modelProcesses.get(model);
    
    if (!modelProcess) {
      throw new Error(`Model ${model} not found`);
    }
    
    if (modelProcess.status === ModelStatus.OFFLINE) {
      throw new Error(`Model ${model} is offline`);
    }
    
    if (modelProcess.status === ModelStatus.INITIALIZING) {
      throw new Error(`Model ${model} is still initializing`);
    }
    
    // Create request message
    const message: ModelMessage = {
      id: requestId,
      type: 'request',
      model,
      role,
      content,
      timestamp: Date.now()
    };
    
    // Send request to model
    return await new Promise<any>((resolve, reject) => {
      // Set timeout
      const timer = setTimeout(() => {
        modelProcess.pendingRequests.delete(requestId);
        performanceMonitor.endOperation(operationId, true, { error: 'timeout' });
        reject(new Error(`Request to ${model} timed out after ${timeout}ms`));
      }, timeout);
      
      // Store pending request
      modelProcess.pendingRequests.set(requestId, {
        resolve: (value) => {
          performanceMonitor.endOperation(operationId, false);
          resolve(value);
        },
        reject: (reason) => {
          performanceMonitor.endOperation(operationId, true, { error: reason.message });
          reject(reason);
        },
        timer,
        startTime: Date.now()
      });
      
      // Update status
      modelProcess.status = ModelStatus.BUSY;
      
      // Send message to model process
      if (modelProcess.process && modelProcess.process.stdin.writable) {
        modelProcess.process.stdin.write(JSON.stringify(message) + '\n');
      } else {
        // Process is not available
        modelProcess.pendingRequests.delete(requestId);
        clearTimeout(timer);
        performanceMonitor.endOperation(operationId, true, { error: 'process_unavailable' });
        reject(new Error(`Model ${model} process is not available`));
      }
    });
  } catch (error) {
    performanceMonitor.endOperation(operationId, true, { error: error.message });
    throw error;
  }
}

/**
 * Send a hybrid request to both models and combine results
 */
export async function sendHybridRequest(
  role: DevTeamRole,
  content: any,
  options: {
    primaryModel?: ModelType;
    fallbackOnly?: boolean;
    timeout?: number;
  } = {}
): Promise<any> {
  const requestId = uuidv4();
  const operationId = `hybrid_request_${requestId}`;
  
  performanceMonitor.startOperation(operationId);
  
  try {
    // Determine primary model
    const primaryModel = options.primaryModel || ModelType.QWEN_OMNI;
    const secondaryModel = primaryModel === ModelType.QWEN_OMNI ? ModelType.OLYMPIC_CODER : ModelType.QWEN_OMNI;
    
    // Check if primary model is available
    const primaryModelProcess = modelProcesses.get(primaryModel);
    const primaryAvailable = primaryModelProcess && primaryModelProcess.status !== ModelStatus.OFFLINE;
    
    // Check if secondary model is available
    const secondaryModelProcess = modelProcesses.get(secondaryModel);
    const secondaryAvailable = secondaryModelProcess && secondaryModelProcess.status !== ModelStatus.OFFLINE;
    
    // Handle fallback logic if primary is not available
    if (!primaryAvailable) {
      console.log(`[ModelIntegration] Primary model ${primaryModel} not available, fallback to ${secondaryModel}`);
      
      if (!secondaryAvailable) {
        throw new Error('All models are offline, cannot process request');
      }
      
      // Use secondary model as fallback
      const result = await sendModelRequest(
        secondaryModel,
        role,
        { ...content, is_fallback: true },
        options.timeout
      );
      
      performanceMonitor.endOperation(operationId, false, { fallback: true });
      
      return {
        result,
        model: secondaryModel,
        is_fallback: true
      };
    }
    
    // For fallback-only requests, just use the primary model
    if (options.fallbackOnly) {
      const result = await sendModelRequest(
        primaryModel,
        role,
        content,
        options.timeout
      );
      
      performanceMonitor.endOperation(operationId, false, { single_model: true });
      
      return {
        result,
        model: primaryModel,
        is_fallback: false
      };
    }
    
    // For hybrid requests, use both models and combine results
    // Start both requests in parallel
    const primaryPromise = primaryAvailable ? 
      sendModelRequest(primaryModel, role, content, options.timeout) : 
      Promise.reject(new Error(`Primary model ${primaryModel} not available`));
    
    const secondaryPromise = secondaryAvailable ? 
      sendModelRequest(secondaryModel, role, content, options.timeout) : 
      Promise.reject(new Error(`Secondary model ${secondaryModel} not available`));
    
    // Wait for both with individual error handling
    const [primaryResult, secondaryResult] = await Promise.allSettled([primaryPromise, secondaryPromise]);
    
    // Check results and combine
    if (primaryResult.status === 'fulfilled' && secondaryResult.status === 'fulfilled') {
      // Both succeeded, combine results
      const combined = combineModelResults(primaryResult.value, secondaryResult.value, role);
      
      performanceMonitor.endOperation(operationId, false, { combined: true });
      
      return {
        result: combined,
        models: [primaryModel, secondaryModel],
        contributions: {
          [primaryModel]: 0.6,
          [secondaryModel]: 0.4
        }
      };
    } else if (primaryResult.status === 'fulfilled') {
      // Only primary succeeded
      performanceMonitor.endOperation(operationId, false, { primary_only: true });
      
      return {
        result: primaryResult.value,
        model: primaryModel,
        is_fallback: false
      };
    } else if (secondaryResult.status === 'fulfilled') {
      // Only secondary succeeded
      performanceMonitor.endOperation(operationId, false, { secondary_only: true });
      
      return {
        result: secondaryResult.value,
        model: secondaryModel,
        is_fallback: true
      };
    } else {
      // Both failed
      throw new Error(`Both models failed: ${primaryResult.reason.message} and ${secondaryResult.reason.message}`);
    }
  } catch (error) {
    performanceMonitor.endOperation(operationId, true, { error: error.message });
    throw error;
  }
}

/**
 * Combine results from multiple models
 */
function combineModelResults(primaryResult: any, secondaryResult: any, role: DevTeamRole): any {
  // The combination strategy depends on the role
  switch (role) {
    case DevTeamRole.ARCHITECT:
      // For architecture, take the more comprehensive design
      if (typeof primaryResult === 'string' && typeof secondaryResult === 'string') {
        // Simple strategy: take the longer one as it likely has more detail
        return primaryResult.length >= secondaryResult.length ? primaryResult : secondaryResult;
      }
      
      // For structured data, merge the results
      return {
        ...secondaryResult,
        ...primaryResult,
        combined: true
      };
      
    case DevTeamRole.BUILDER:
      // For code generation, take the primary model's code but consider improvements from secondary
      if (typeof primaryResult === 'string' && typeof secondaryResult === 'string') {
        // If they're both strings (code), prefer the primary
        return primaryResult;
      }
      
      // For structured outputs, merge with primary taking precedence
      return {
        ...secondaryResult,
        ...primaryResult,
        combined: true
      };
      
    case DevTeamRole.TESTER:
      // For testing, combine the test cases from both models
      if (Array.isArray(primaryResult) && Array.isArray(secondaryResult)) {
        // Deduplicate test cases
        const combined = [...primaryResult];
        
        for (const testCase of secondaryResult) {
          if (!combined.some(tc => JSON.stringify(tc) === JSON.stringify(testCase))) {
            combined.push(testCase);
          }
        }
        
        return combined;
      }
      
      // If not arrays, merge objects with primary taking precedence
      return {
        ...secondaryResult,
        ...primaryResult,
        combined: true
      };
      
    case DevTeamRole.REVIEWER:
      // For review, combine feedback from both models
      if (Array.isArray(primaryResult) && Array.isArray(secondaryResult)) {
        // Combine feedback items
        return [...primaryResult, ...secondaryResult];
      }
      
      // For text feedback, concatenate
      if (typeof primaryResult === 'string' && typeof secondaryResult === 'string') {
        return `${primaryResult}\n\nAdditional insights:\n${secondaryResult}`;
      }
      
      // Default: merge objects
      return {
        ...secondaryResult,
        ...primaryResult,
        combined: true
      };
  }
}

// ----------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------

/**
 * Get the status of all model processes
 */
export function getModelStatus(): Record<string, any> {
  const status: Record<string, any> = {};
  
  for (const [model, modelProcess] of modelProcesses.entries()) {
    status[model] = {
      status: modelProcess.status,
      stats: {
        ...modelProcess.stats,
        pendingRequests: modelProcess.pendingRequests.size,
        uptimeSeconds: Math.round((Date.now() - modelProcess.stats.startTime) / 1000)
      }
    };
  }
  
  return status;
}

/**
 * Generate code using the AI dev team
 */
export async function generateCode(
  requirements: string,
  options: {
    language?: string;
    framework?: string;
    architecture?: string;
    primaryModel?: ModelType;
    timeout?: number;
  } = {}
): Promise<any> {
  try {
    // 1. Plan architecture using the ARCHITECT role
    console.log('[ModelIntegration] Step 1: Planning architecture...');
    const architectureRequest = {
      task: 'architecture_planning',
      requirements,
      options: {
        language: options.language,
        framework: options.framework,
        architecture: options.architecture
      }
    };
    
    const architectureResult = await sendHybridRequest(
      DevTeamRole.ARCHITECT,
      architectureRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    // 2. Generate code using the BUILDER role
    console.log('[ModelIntegration] Step 2: Generating code...');
    const codeRequest = {
      task: 'code_generation',
      requirements,
      architecture: architectureResult.result,
      options: {
        language: options.language,
        framework: options.framework
      }
    };
    
    const codeResult = await sendHybridRequest(
      DevTeamRole.BUILDER,
      codeRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    // 3. Generate tests using the TESTER role
    console.log('[ModelIntegration] Step 3: Generating tests...');
    const testRequest = {
      task: 'test_generation',
      requirements,
      code: codeResult.result,
      options: {
        language: options.language,
        framework: options.framework
      }
    };
    
    const testResult = await sendHybridRequest(
      DevTeamRole.TESTER,
      testRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    // 4. Review and optimize using the REVIEWER role
    console.log('[ModelIntegration] Step 4: Reviewing and optimizing...');
    const reviewRequest = {
      task: 'code_review',
      requirements,
      code: codeResult.result,
      tests: testResult.result,
      options: {
        language: options.language,
        framework: options.framework
      }
    };
    
    const reviewResult = await sendHybridRequest(
      DevTeamRole.REVIEWER,
      reviewRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    // Return the final result
    return {
      architecture: architectureResult.result,
      code: codeResult.result,
      tests: testResult.result,
      review: reviewResult.result,
      models_used: {
        architect: architectureResult.model || architectureResult.models,
        builder: codeResult.model || codeResult.models,
        tester: testResult.model || testResult.models,
        reviewer: reviewResult.model || reviewResult.models
      }
    };
  } catch (error) {
    console.error('[ModelIntegration] Error in generateCode:', error);
    throw error;
  }
}

/**
 * Analyze and enhance existing code
 */
export async function enhanceCode(
  code: string,
  options: {
    requirements?: string;
    language?: string;
    enhancement?: 'optimize' | 'refactor' | 'document' | 'test' | 'fix';
    primaryModel?: ModelType;
    timeout?: number;
  } = {}
): Promise<any> {
  try {
    const enhanceRequest = {
      task: 'code_enhancement',
      code,
      requirements: options.requirements,
      enhancement: options.enhancement || 'optimize',
      language: options.language
    };
    
    const result = await sendHybridRequest(
      DevTeamRole.BUILDER,
      enhanceRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    return {
      enhanced_code: result.result,
      model: result.model || result.models
    };
  } catch (error) {
    console.error('[ModelIntegration] Error in enhanceCode:', error);
    throw error;
  }
}

/**
 * Debug and fix code
 */
export async function debugCode(
  code: string,
  error: string,
  options: {
    language?: string;
    primaryModel?: ModelType;
    timeout?: number;
  } = {}
): Promise<any> {
  try {
    const debugRequest = {
      task: 'code_debugging',
      code,
      error,
      language: options.language
    };
    
    const result = await sendHybridRequest(
      DevTeamRole.TESTER,
      debugRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    return {
      fixed_code: result.result,
      model: result.model || result.models
    };
  } catch (error) {
    console.error('[ModelIntegration] Error in debugCode:', error);
    throw error;
  }
}

/**
 * Analyze and explain code
 */
export async function explainCode(
  code: string,
  options: {
    language?: string;
    detail_level?: 'simple' | 'detailed' | 'comprehensive';
    primaryModel?: ModelType;
    timeout?: number;
  } = {}
): Promise<any> {
  try {
    const explainRequest = {
      task: 'code_explanation',
      code,
      language: options.language,
      detail_level: options.detail_level || 'detailed'
    };
    
    const result = await sendHybridRequest(
      DevTeamRole.REVIEWER,
      explainRequest,
      { primaryModel: options.primaryModel, timeout: options.timeout }
    );
    
    return {
      explanation: result.result,
      model: result.model || result.models
    };
  } catch (error) {
    console.error('[ModelIntegration] Error in explainCode:', error);
    throw error;
  }
}