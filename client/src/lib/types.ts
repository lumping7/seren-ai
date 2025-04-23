export type AIMessage = {
  id?: number;
  conversationId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  model?: 'qwen' | 'olympic' | 'hybrid';
  userId?: number;
  timestamp?: Date;
  metadata?: Record<string, any>;
};

export type AIMemory = {
  id?: number;
  title: string;
  content: string;
  type: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
};

export type AIStatus = {
  qwen: boolean;
  olympic: boolean;
  neuroSymbolic: boolean;
  memory: boolean;
  systemLoad: number;
};

export type FeatureFlags = {
  selfUpgradeEnabled: boolean;
  codeExecutionEnabled: boolean;
  autonomousActionsEnabled: boolean;
  internetAccessEnabled: boolean;
};

export type ModelConfig = {
  defaultModel: 'qwen' | 'olympic' | 'hybrid';
  temperature: number;
  maxTokens: number;
};

export type AISettings = {
  modelConfig: ModelConfig;
  featureFlags: FeatureFlags;
};

export type AIResponse = {
  model: string;
  generated_text: string;
  metadata: {
    processing_time: number;
    tokens_used: number;
    model_version: string;
  };
};

export type CodeExecutionResult = {
  success: boolean;
  stdout: string;
  stderr: string;
  executionTime: number;
  executionId: string;
  exitCode?: number;
};

export type WebSocketMessage = {
  type: 'auth' | 'chat-message' | 'new-message' | 'project-update' | 'task-complete' | 'status-update' | 'error';
  message?: AIMessage;
  data?: Record<string, any>;
  projectId?: string;
  update?: Partial<Project>;
  taskId?: string;
  error?: string;
  statusCode?: number;
};

export type Project = {
  id: string;
  name: string;
  description: string;
  status: string;
  progress: number;
  createdAt: string;
  lastUpdated: string;
};

export type BackgroundTask = {
  id: string;
  type: string;
  name: string;
  status: string;
  startTime: string;
};
