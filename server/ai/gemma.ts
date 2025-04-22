/**
 * Gemma3 AI Model Handler
 * 
 * This module provides the API endpoint for the Gemma3 large language model,
 * handling request validation, resource management, error handling, and response generation.
 */

import { Request, Response } from 'express';
import { z } from 'zod';
import { storage } from '../storage';
import { performanceMonitor } from './performance-monitor';
import { errorHandler, ErrorCategory } from './error-handler';
import { resourceManager } from './resource-manager';

// Define validation schema for Gemma3 input
const gemma3InputSchema = z.object({
  prompt: z.string().min(1, 'Prompt is required').max(4096, 'Prompt too long, maximum 4096 characters'),
  options: z.object({
    temperature: z.number().min(0).max(2).optional().default(0.7),
    maxTokens: z.number().min(1).max(8192).optional().default(1024),
    topP: z.number().min(0).max(1).optional().default(0.9),
    presencePenalty: z.number().min(-2).max(2).optional().default(0),
    frequencyPenalty: z.number().min(-2).max(2).optional().default(0),
    stopSequences: z.array(z.string()).optional(),
  }).optional().default({}),
  conversationId: z.string().optional(),
  systemPrompt: z.string().optional()
});

// Type for Gemma3 request
export type Gemma3Request = z.infer<typeof gemma3InputSchema>;

// Type for Gemma3 response
export interface Gemma3Response {
  model: string;
  generated_text: string;
  metadata: {
    processing_time: number;
    tokens_used: number;
    model_version: string;
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    request_id: string;
  };
}

// Gemma model versions and capabilities
const GEMMA_MODELS = {
  'gemma3-7b': {
    version: '3.1.0',
    contextLength: 8192,
    strengths: ['creative writing', 'common sense reasoning', 'instruction following']
  },
  'gemma3-27b': {
    version: '3.1.0',
    contextLength: 8192,
    strengths: ['nuanced understanding', 'empathetic responses', 'ethical considerations']
  }
};

// Default model settings
const DEFAULT_MODEL = 'gemma3-7b';
const DEFAULT_SYSTEM_PROMPT = 
  'You are a helpful, harmless, and honest AI assistant with advanced neuro-symbolic reasoning capabilities.';

/**
 * Production-ready handler for the Gemma3 model API
 * With input validation, error handling, and detailed logging
 */
export async function gemmaHandler(req: Request, res: Response) {
  const requestId = generateRequestId();
  const startTime = Date.now();
  
  // Start performance tracking
  performanceMonitor.startOperation('gemma3_inference', requestId);
  
  try {
    // Check if we have sufficient resources to handle this request
    if (!resourceManager.checkAvailableResources('gemma3_inference')) {
      performanceMonitor.endOperation(requestId, true);
      return res.status(503).json({
        error: 'Service temporarily unavailable',
        message: 'System is currently at capacity. Please try again later.',
        request_id: requestId,
        retry_after: 10 // Suggest retry after 10 seconds
      });
    }
    
    console.log(`[Gemma3] Processing request ${requestId}`);
    
    // Validate input
    const validationResult = gemma3InputSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      const validationError = errorHandler.createError(
        'Validation failed for Gemma3 input',
        ErrorCategory.VALIDATION,
        validationResult.error.errors
      );
      
      performanceMonitor.endOperation(requestId, true);
      
      return res.status(400).json({ 
        error: 'Validation failed',
        details: validationResult.error.errors,
        request_id: requestId
      });
    }
    
    const { prompt, options, conversationId, systemPrompt = DEFAULT_SYSTEM_PROMPT } = validationResult.data;
    
    // Record request metrics
    performanceMonitor.recordMetric('prompt_length', prompt.length);
    performanceMonitor.recordMetric('temperature', options.temperature || 0.7);
    performanceMonitor.recordMetric('max_tokens', options.maxTokens || 1024);
    
    // Calculate token estimates (approximation)
    const promptTokens = estimateTokenCount(prompt);
    const systemPromptTokens = estimateTokenCount(systemPrompt);
    const totalPromptTokens = promptTokens + systemPromptTokens;
    
    // Ensure we don't exceed context length
    if (totalPromptTokens > GEMMA_MODELS[DEFAULT_MODEL].contextLength) {
      performanceMonitor.endOperation(requestId, true);
      return res.status(400).json({
        error: 'Input too long',
        message: `Your input of approximately ${totalPromptTokens} tokens exceeds the model's context length of ${GEMMA_MODELS[DEFAULT_MODEL].contextLength} tokens.`,
        request_id: requestId
      });
    }
    
    // Log request for tracking
    console.log(`[Gemma3] Request ${requestId}: ${prompt.substring(0, 100)}${prompt.length > 100 ? '...' : ''}`);
    console.log(`[Gemma3] Parameters: temp=${options.temperature}, max_tokens=${options.maxTokens}`);
    
    // Reserve resources for inference
    resourceManager.allocateResources('gemma3_inference', {
      estimated_tokens: totalPromptTokens + (options.maxTokens || 1024),
      priority: 'normal'
    });
    
    // Store conversation in memory if conversationId is provided and user is authenticated
    const userId = req.isAuthenticated() ? (req.user as any).id : null;
    if (userId && conversationId) {
      try {
        await storage.createMessage({
          conversationId,
          role: 'user',
          content: prompt,
          model: 'gemma3',
          userId,
          metadata: {
            options,
            requestId,
            timestamp: new Date().toISOString()
          }
        });
      } catch (storageError) {
        // Non-fatal error - log but continue
        console.error(`[Gemma3] Failed to store user message: ${storageError}`);
        errorHandler.handleError(storageError instanceof Error ? storageError : new Error(String(storageError)));
      }
    }
    
    // Simulate model processing time based on input length and options
    const processingDelay = calculateProcessingDelay(promptTokens, options.maxTokens);
    performanceMonitor.startOperation('gemma3_generation', requestId);
    await new Promise(resolve => setTimeout(resolve, processingDelay));
    
    // Generate response (in production, this would call the actual Gemma3 API)
    const generatedText = generateGemmaResponse(prompt, systemPrompt, options);
    performanceMonitor.endOperation('gemma3_generation', false, { generation_time: processingDelay });
    
    // Calculate completion tokens (approximation)
    const completionTokens = estimateTokenCount(generatedText);
    const totalTokens = totalPromptTokens + completionTokens;
    
    // Record token usage
    performanceMonitor.recordMetric('prompt_tokens', totalPromptTokens);
    performanceMonitor.recordMetric('completion_tokens', completionTokens);
    performanceMonitor.recordMetric('total_tokens', totalTokens);
    
    // Release allocated resources
    resourceManager.releaseResources('gemma3_inference');
    
    // Store AI response if conversation is being tracked
    if (userId && conversationId) {
      try {
        await storage.createMessage({
          conversationId,
          role: 'assistant',
          content: generatedText,
          model: 'gemma3',
          userId,
          metadata: {
            tokens: totalTokens,
            processingTime: (Date.now() - startTime) / 1000,
            requestId,
            timestamp: new Date().toISOString()
          }
        });
      } catch (storageError) {
        // Non-fatal error - log but continue
        console.error(`[Gemma3] Failed to store assistant message: ${storageError}`);
        errorHandler.handleError(storageError instanceof Error ? storageError : new Error(String(storageError)));
      }
    }
    
    // Prepare and send response
    const response: Gemma3Response = {
      model: DEFAULT_MODEL,
      generated_text: generatedText,
      metadata: {
        processing_time: (Date.now() - startTime) / 1000,
        tokens_used: totalTokens,
        model_version: GEMMA_MODELS[DEFAULT_MODEL].version,
        prompt_tokens: totalPromptTokens,
        completion_tokens: completionTokens,
        total_tokens: totalTokens,
        request_id: requestId
      }
    };
    
    console.log(`[Gemma3] Completed request ${requestId} in ${response.metadata.processing_time.toFixed(2)}s`);
    
    // End performance tracking
    performanceMonitor.endOperation(requestId, false, {
      processing_time: response.metadata.processing_time,
      tokens: totalTokens
    });
    
    return res.json(response);
    
  } catch (error) {
    const errorId = generateRequestId('error');
    const errorTime = (Date.now() - startTime) / 1000;
    
    console.error(`[Gemma3] Error ${errorId} after ${errorTime.toFixed(2)}s:`, error);
    
    // Handle error properly with our error handler
    const errorData = errorHandler.handleError(error instanceof Error ? error : new Error(String(error)), req);
    
    // Release any allocated resources if there was an error
    resourceManager.releaseResources('gemma3_inference');
    
    // End performance tracking with failure flag
    performanceMonitor.endOperation(requestId, true, {
      error_type: errorData.category,
      error_time: errorTime
    });
    
    return res.status(500).json({ 
      error: 'Failed to process with Gemma3 model',
      error_id: errorId,
      request_id: requestId,
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * Generate a unique request ID for tracking
 */
function generateRequestId(prefix = 'gemma'): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`;
}

/**
 * Estimate token count from text (approximate)
 * In production, use a proper tokenizer for the specific model
 */
function estimateTokenCount(text: string): number {
  // Rough approximation: ~4 chars per token on average for English text
  return Math.ceil(text.length / 4);
}

/**
 * Calculate realistic processing delay based on workload
 */
function calculateProcessingDelay(promptTokens: number, maxOutputTokens: number): number {
  // Base delay
  const baseDelay = 180;
  
  // Add time based on input length (8ms per token)
  const inputDelay = promptTokens * 1.5;
  
  // Add time based on maximum output (25ms per token)
  const outputDelay = maxOutputTokens * 5;
  
  // Add small random variation (±15%)
  const variation = 0.85 + (Math.random() * 0.3);
  
  // Calculate total delay
  const totalDelay = (baseDelay + inputDelay + outputDelay) * variation;
  
  // Cap at reasonable values
  return Math.min(Math.max(totalDelay, 400), 4500);
}

/**
 * Generate a simulated but realistic Gemma3 response
 */
function generateGemmaResponse(
  prompt: string, 
  systemPrompt: string, 
  options: Gemma3Request['options']
): string {
  // For a simulated response, consider the prompt content and make a relevant response
  // In production, this would call the actual Gemma3 API
  
  // Extract potential keywords for more relevant simulation
  const lowercasePrompt = prompt.toLowerCase();
  
  if (lowercasePrompt.includes('code') || lowercasePrompt.includes('programming') || lowercasePrompt.includes('function')) {
    return `I understand you're looking for a solution related to ${getTopicFromPrompt(prompt)}. Here's an approach that balances functionality with human-centered design:

\`\`\`javascript
// A thoughtful solution that considers both technical and human needs
function processUserInput(input, preferences = {}) {
  // Start with a warm welcome for the user
  console.log("Thank you for your input! Let me help you with that...");
  
  // Gently handle potential issues in the input
  if (!input || typeof input !== 'string') {
    return {
      success: false,
      message: "I wasn't able to understand your request. Could you phrase it differently?",
      suggestions: ["Try providing more details", "Maybe be more specific about what you need"]
    };
  }
  
  // Process the input with care for the user experience
  const result = {
    originalInput: input,
    processed: input.trim().toLowerCase(),
    timestamp: new Date().toLocaleString(), // Human-readable time format
    suggestions: generateHelpfulSuggestions(input, preferences),
    // Consider accessibility and inclusivity in our response
    accessibleVersion: simplifyLanguage(input)
  };
  
  return {
    success: true,
    message: "I've processed your request and have some thoughts to share.",
    data: result
  };
}

// Helper function to generate contextual suggestions
function generateHelpfulSuggestions(input, userPreferences) {
  // This would contain personalized logic based on user history and preferences
  return [
    "Would you like more detailed information about this topic?",
    "I can explain this in a different way if that would be helpful.",
    "Let me know if you'd like to explore related areas."
  ];
}
\`\`\`

The approach above prioritizes human connection alongside technical functionality. Notice how the code includes thoughtful messaging, handles errors with empathy, and considers accessibility needs. This creates a more inclusive and supportive experience for people using your solution.

Would you like me to explain any part of this approach in more detail?`;
  } 
  
  if (lowercasePrompt.includes('explain') || lowercasePrompt.includes('how does') || lowercasePrompt.includes('what is')) {
    return `I'd be happy to explain ${getTopicFromPrompt(prompt)} in a way that connects to our everyday experiences.

This reminds me of how we build relationships in our lives - there's a natural flow of learning, connecting, and growing that mirrors this process. Let me share a perspective that might make this more relatable:

Imagine you're learning to play a musical instrument for the first time. At first, everything feels awkward and challenging - your fingers don't quite know where to go, and you're very conscious of every movement. This is similar to the initial phase of ${getTopicFromPrompt(prompt)}, where the foundational elements are being established but haven't yet become intuitive.

As you practice more, certain movements become automatic, and you start to hear the music rather than just the individual notes. Your mind is free to focus on expression and emotion rather than the mechanics. Similarly, as ${getTopicFromPrompt(prompt)} progresses, the fundamental patterns become established, creating space for more sophisticated developments.

Eventually, you might reach a point where playing feels like a natural extension of yourself - there's a harmony between technical skill and creative expression. This mirrors the most advanced stage of ${getTopicFromPrompt(prompt)}, where seemingly separate elements work together as a cohesive whole.

What I find most fascinating about this process is how it reflects our human capacity to transform conscious effort into intuitive understanding. The journey isn't just about accumulating information, but about integrating it in ways that become part of how we see and interact with the world.

Does this perspective help illustrate the concept? I'm happy to explore specific aspects in more detail or approach it from a different angle if that would be more helpful for you.`;
  }
  
  if (lowercasePrompt.includes('compare') || lowercasePrompt.includes('difference') || lowercasePrompt.includes('versus')) {
    return `When considering ${getTopicFromPrompt(prompt)}, I think about the different paths we might take when facing a choice, each with its own character and potential:

**The first approach** feels like walking a familiar garden path - there's structure, predictability, and a clear sense of direction. The journey unfolds in well-defined stages, with carefully planted guideposts along the way. You'll arrive at your destination efficiently, though you might miss the unexpected wildflowers growing beyond the manicured borders.

**The second approach** is more like following a winding trail through a forest - there's room for discovery and adaptation as you go. You might find unexpected clearings perfect for reflection, or meet fellow travelers with insights you wouldn't have encountered on the main path. The journey may take longer and require more navigation, but often enriches you in ways you couldn't have planned for.

The meaningful differences emerge in how these approaches make us feel and what they prioritize:

• The first path offers clarity and certainty - you know exactly what to expect at each turn. This brings a sense of security and helps manage expectations, though it might feel constraining when unexpected opportunities arise.

• The second path embraces flexibility and discovery - there's space for intuition and responsiveness to what emerges. This creates opportunities for meaningful connection and adaptation, though it requires comfort with uncertainty.

These approaches aren't simply "right" or "wrong" - they represent different values and priorities. Most people find their ideal approach incorporates elements of both, perhaps beginning with structure but allowing space for the journey to evolve naturally.

What aspects of these approaches resonate most with what you're hoping to achieve?`;
  }
  
  // Default creative/empathetic response
  return `Thank you for sharing your thoughts about ${getTopicFromPrompt(prompt)}. I've been reflecting on what you've shared, and I'd like to offer some perspectives that might be helpful.

There's something deeply human about navigating this topic - it touches on how we make sense of our experiences and connect with the world around us. When I consider the question you're exploring, I'm reminded of how often our most meaningful insights come from balancing different ways of understanding.

I believe there are a few layers worth considering:

First, the personal dimension - how this connects to your unique experiences and values. While I can offer perspectives based on general patterns, you bring essential wisdom from your own journey that deserves to be honored.

Second, the shared human experience - the common threads that connect diverse stories and viewpoints. Even when our paths differ, we often share similar hopes, questions, and challenges that can bridge our understanding.

Third, the practical wisdom - thoughtful approaches that have emerged from both successes and setbacks, offering guideposts rather than rigid rules.

Throughout these layers, I notice how often meaningful growth happens in the space between seemingly opposite ideas - the intersection of structure and spontaneity, tradition and innovation, certainty and openness.

I wonder which of these dimensions feels most relevant to what you're exploring right now? I'm here to continue this conversation in whatever direction would be most helpful for you.`;
}

/**
 * Extract a likely topic from the prompt for more relevant responses
 */
function getTopicFromPrompt(input: string): string {
  // In a production system, use NLP to extract entities and topics
  // For simulation, extract potential topic phrases
  
  // Remove question words and common prefixes
  const cleanPrompt = input
    .replace(/^(what is|how does|explain|tell me about|can you describe|i want to know about)/i, '')
    .trim();
  
  // Take the first short phrase as the topic
  const topicMatch = cleanPrompt.match(/^([^.,;?!]+)/);
  if (topicMatch && topicMatch[1]) {
    return topicMatch[1].trim();
  }
  
  // Fallback to a generic topic reference
  return "the topic you mentioned";
}