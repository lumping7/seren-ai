import { Request, Response } from 'express';
import { z } from 'zod';
import { storage } from '../storage';
import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';

// Define validation schema for hybrid model input
const hybridInputSchema = z.object({
  prompt: z.string().min(1, 'Prompt is required').max(8192, 'Prompt too long, maximum 8192 characters'),
  options: z.object({
    mode: z.enum(['collaborative', 'specialized', 'competitive']).optional().default('collaborative'),
    temperature: z.number().min(0).max(2).optional().default(0.75),
    maxTokens: z.number().min(1).max(8192).optional().default(2048),
    topP: z.number().min(0).max(1).optional().default(0.95),
    presencePenalty: z.number().min(-2).max(2).optional().default(0),
    frequencyPenalty: z.number().min(-2).max(2).optional().default(0),
    stopSequences: z.array(z.string()).optional(),
  }).optional().default({}),
  conversationId: z.string().optional(),
  systemPrompt: z.string().optional(),
  includeModelDetails: z.boolean().optional().default(true)
});

// Type for Hybrid request
type HybridRequest = z.infer<typeof hybridInputSchema>;

// Type for Hybrid response
interface HybridResponse {
  model: string;
  generated_text: string;
  metadata: {
    processing_time: number;
    tokens_used: number;
    model_version: string;
    collaborative_data?: {
      llama3_contribution: number;
      gemma3_contribution: number;
      reasoning_steps: string[];
    };
    request_id: string;
  };
}

/**
 * Production-ready Hybrid Model Handler
 * This orchestrates collaboration between Llama3 and Gemma3 models, leveraging 
 * their complementary strengths to generate higher quality outputs
 */
export async function hybridHandler(req: Request, res: Response) {
  const requestId = generateRequestId();
  const startTime = Date.now();
  
  try {
    console.log(`[Hybrid] Processing request ${requestId}`);
    
    // Validate input
    const validationResult = hybridInputSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      return res.status(400).json({
        error: 'Validation failed',
        details: validationResult.error.errors,
        request_id: requestId
      });
    }
    
    const { prompt, options, conversationId, systemPrompt, includeModelDetails } = validationResult.data;
    
    // Store user message in memory if conversationId is provided and user is authenticated
    const userId = req.isAuthenticated() ? (req.user as any).id : null;
    if (userId && conversationId) {
      try {
        await storage.createMessage({
          conversationId,
          role: 'user',
          content: prompt,
          model: 'hybrid',
          userId,
          metadata: {
            options,
            requestId,
            timestamp: new Date().toISOString()
          }
        });
      } catch (storageError: any) {
        // Non-fatal error - log but continue
        console.error(`[Hybrid] Failed to store user message: ${storageError.message || 'Unknown error'}`);
      }
    }
    
    // Setup prompts and options for each model
    // These specialized prompts give each model a distinct role
    const llamaPrompt = `[SYSTEM: You are the architecture expert in a collaborative AI team. Your role is to design systems, plan implementation details, and provide technical specifications. Focus on architecture, design patterns, and technical accuracy.]\n\n${prompt}`;
    
    const gemmaPrompt = `[SYSTEM: You are the creative implementation expert in a collaborative AI team. Your role is to implement systems designed by the architecture expert, add creative solutions, and ensure the final product is user-friendly. Focus on practical implementation, edge cases, and user experience.]\n\n${prompt}`;
    
    // Adjust options for each model based on their strengths
    const llamaOptions = {
      ...options,
      temperature: Math.min(options.temperature * 0.9, 1.0) // Slightly lower temperature for more precise logic
    };
    
    const gemmaOptions = {
      ...options,
      temperature: Math.min(options.temperature * 1.1, 1.5) // Slightly higher temperature for more creative outputs
    };
    
    // Determine processing mode based on options
    const processingMode = options.mode || 'collaborative';
    let result: string;
    let collaborativeData: any = {
      llama3_contribution: 0.5,
      gemma3_contribution: 0.5,
      reasoning_steps: []
    };
    
    console.log(`[Hybrid] Using ${processingMode} mode for request ${requestId}`);
    
    switch (processingMode) {
      case 'collaborative':
        // Make parallel calls to both models
        const [llamaResult, gemmaResult] = await Promise.all([
          callModelHandler(llamaHandler, 'llama3', llamaPrompt, llamaOptions),
          callModelHandler(gemmaHandler, 'gemma3', gemmaPrompt, gemmaOptions)
        ]);
        
        // Combine the responses using the collaborative approach
        result = await combineCollaborativeResponses(
          llamaResult.generated_text, 
          gemmaResult.generated_text,
          prompt,
          collaborativeData
        );
        break;
        
      case 'specialized':
        // Choose model based on prompt content
        if (isArchitecturalTask(prompt)) {
          // For architecture/logical tasks, prefer Llama3
          const llamaResult = await callModelHandler(llamaHandler, 'llama3', llamaPrompt, llamaOptions);
          result = llamaResult.generated_text;
          collaborativeData.llama3_contribution = 0.9;
          collaborativeData.gemma3_contribution = 0.1;
          collaborativeData.reasoning_steps.push('Detected architectural task, prioritized Llama3');
        } else {
          // For creative/implementation tasks, prefer Gemma3
          const gemmaResult = await callModelHandler(gemmaHandler, 'gemma3', gemmaPrompt, gemmaOptions);
          result = gemmaResult.generated_text;
          collaborativeData.llama3_contribution = 0.1;
          collaborativeData.gemma3_contribution = 0.9;
          collaborativeData.reasoning_steps.push('Detected implementation task, prioritized Gemma3');
        }
        break;
        
      case 'competitive':
        // Make parallel calls to both models
        const [llamaCompResult, gemmaCompResult] = await Promise.all([
          callModelHandler(llamaHandler, 'llama3', llamaPrompt, llamaOptions),
          callModelHandler(gemmaHandler, 'gemma3', gemmaPrompt, gemmaOptions)
        ]);
        
        // Choose the best response based on quality metrics
        const [selectedResponse, metrics] = selectBestResponse(
          llamaCompResult.generated_text, 
          gemmaCompResult.generated_text,
          prompt
        );
        
        result = selectedResponse;
        collaborativeData = metrics;
        break;
        
      default:
        throw new Error(`Invalid processing mode: ${processingMode}`);
    }
    
    // Store AI response if conversation is being tracked
    if (userId && conversationId) {
      try {
        await storage.createMessage({
          conversationId,
          role: 'assistant',
          content: result,
          model: 'hybrid',
          userId,
          metadata: {
            processingMode,
            collaborativeData,
            processingTime: (Date.now() - startTime) / 1000,
            requestId,
            timestamp: new Date().toISOString()
          }
        });
      } catch (storageError: any) {
        // Non-fatal error - log but continue
        console.error(`[Hybrid] Failed to store assistant message: ${storageError.message || 'Unknown error'}`);
      }
    }
    
    // Prepare and send response
    const response: HybridResponse = {
      model: 'hybrid',
      generated_text: includeModelDetails 
        ? addModelAttribution(result, collaborativeData) 
        : result,
      metadata: {
        processing_time: (Date.now() - startTime) / 1000,
        tokens_used: estimateTokenCount(result) + estimateTokenCount(prompt),
        model_version: '1.0.0',
        collaborative_data: collaborativeData,
        request_id: requestId
      }
    };
    
    console.log(`[Hybrid] Completed request ${requestId} in ${response.metadata.processing_time.toFixed(2)}s`);
    return res.json(response);
    
  } catch (error: any) {
    const errorId = generateRequestId('error');
    const errorTime = (Date.now() - startTime) / 1000;
    
    console.error(`[Hybrid] Error ${errorId} after ${errorTime.toFixed(2)}s:`, error);
    
    return res.status(500).json({
      error: 'Failed to process with Hybrid model',
      error_id: errorId,
      request_id: requestId,
      timestamp: new Date().toISOString(),
      details: error.message || 'Unknown error'
    });
  }
}

// Function removed - no longer needed

/**
 * Call a model without passing the actual req/res objects to avoid circular JSON
 */
async function callModelHandler(
  handler: Function, 
  modelType: 'llama3' | 'gemma3',
  prompt: string,
  options: any = {}
): Promise<any> {
  // Create mock req/res objects to avoid circular references
  const mockReq = {
    body: {
      prompt,
      options,
      includeModelDetails: false
    },
    isAuthenticated: () => false // No need to store for internal calls
  };
  
  // Create a synthetic response object
  const mockRes = {
    _status: 200,
    _data: null,
    status: function(code: number) {
      this._status = code;
      return this;
    },
    json: function(data: any) {
      this._data = data;
      return this;
    },
    getData: function() {
      return this._data;
    }
  };
  
  try {
    // Call the handler with our mock objects
    await handler(mockReq, mockRes);
    
    // Return the captured data
    return mockRes.getData();
  } catch (error: any) {
    console.error(`[Hybrid] Error calling ${modelType} model:`, error.message || 'Unknown error');
    return {
      model: modelType,
      generated_text: `Error: Could not get response from ${modelType} model.`,
      metadata: {
        error: true,
        error_message: error.message || 'Unknown error'
      }
    };
  }
}

/**
 * Combine responses from Llama3 and Gemma3 in a collaborative way
 */
async function combineCollaborativeResponses(
  llamaResponse: string,
  gemmaResponse: string,
  originalPrompt: string,
  collaborativeData: any
): Promise<string> {
  // Analyze which parts of the response should come from which model
  const llamaStrengths = detectModelStrengths(llamaResponse, 'llama3');
  const gemmaStrengths = detectModelStrengths(gemmaResponse, 'gemma3');
  
  // Determine the contribution ratios based on strengths
  const totalStrengthScore = llamaStrengths.score + gemmaStrengths.score;
  collaborativeData.llama3_contribution = llamaStrengths.score / totalStrengthScore;
  collaborativeData.gemma3_contribution = gemmaStrengths.score / totalStrengthScore;
  
  // Log the contribution reasoning
  collaborativeData.reasoning_steps.push(
    `Llama3 strength areas: ${llamaStrengths.areas.join(', ')}`
  );
  collaborativeData.reasoning_steps.push(
    `Gemma3 strength areas: ${gemmaStrengths.areas.join(', ')}`
  );
  collaborativeData.reasoning_steps.push(
    `Contribution ratio - Llama3: ${(collaborativeData.llama3_contribution * 100).toFixed(1)}%, Gemma3: ${(collaborativeData.gemma3_contribution * 100).toFixed(1)}%`
  );
  
  // Parse responses into sections
  const llamaSections = parseResponseIntoSections(llamaResponse);
  const gemmaSections = parseResponseIntoSections(gemmaResponse);
  
  // Construct final combined response
  let combinedResponse = '';
  
  // Always use Llama3 for architectural decisions and technical specifications
  if (llamaSections.architecture) {
    combinedResponse += llamaSections.architecture + '\n\n';
    collaborativeData.reasoning_steps.push('Used Llama3 for architectural design');
  }
  
  // Always use Gemma3 for implementation details and creative solutions
  if (gemmaSections.implementation) {
    combinedResponse += gemmaSections.implementation + '\n\n';
    collaborativeData.reasoning_steps.push('Used Gemma3 for implementation details');
  }
  
  // For other sections, choose based on quality
  if (llamaSections.introduction && gemmaSections.introduction) {
    combinedResponse = selectBetterSection(
      llamaSections.introduction, 
      gemmaSections.introduction,
      'introduction'
    ) + '\n\n' + combinedResponse;
    
    collaborativeData.reasoning_steps.push(
      `Selected ${combinedResponse.startsWith(llamaSections.introduction) ? 'Llama3' : 'Gemma3'} introduction`
    );
  }
  
  if (llamaSections.conclusion && gemmaSections.conclusion) {
    combinedResponse += selectBetterSection(
      llamaSections.conclusion, 
      gemmaSections.conclusion,
      'conclusion'
    );
    
    collaborativeData.reasoning_steps.push(
      `Selected ${combinedResponse.endsWith(llamaSections.conclusion) ? 'Llama3' : 'Gemma3'} conclusion`
    );
  }
  
  // If we couldn't parse sections properly, do a simpler content integration
  if (!combinedResponse.trim()) {
    combinedResponse = integrateContent(llamaResponse, gemmaResponse, llamaStrengths.areas, gemmaStrengths.areas);
    collaborativeData.reasoning_steps.push('Used content integration for combining responses');
  }
  
  return combinedResponse;
}

/**
 * Detect the strengths of a model's response
 */
function detectModelStrengths(response: string, model: 'llama3' | 'gemma3'): { score: number; areas: string[] } {
  const strengths: string[] = [];
  let score = 1.0; // Base score
  
  // Analyze response for different strengths
  if (model === 'llama3') {
    // Check for Llama3 strengths
    if (response.includes('architecture') || response.includes('design pattern') || 
        response.includes('system design') || response.includes('components')) {
      strengths.push('architecture');
      score += 0.5;
    }
    
    if (response.includes('database') || response.includes('schema') || 
        response.includes('data model') || response.includes('SQL')) {
      strengths.push('data modeling');
      score += 0.4;
    }
    
    if (response.includes('algorithm') || response.includes('complexity') || 
        response.includes('efficiency') || response.includes('optimization')) {
      strengths.push('algorithmic thinking');
      score += 0.6;
    }
    
    if (response.includes('security') || response.includes('authentication') || 
        response.includes('authorization') || response.includes('validation')) {
      strengths.push('security considerations');
      score += 0.5;
    }
    
    if (response.includes('API') || response.includes('interface') || 
        response.includes('endpoint') || response.includes('REST')) {
      strengths.push('API design');
      score += 0.4;
    }
  } else {
    // Check for Gemma3 strengths
    if (response.includes('user experience') || response.includes('UI') || 
        response.includes('interface design') || response.includes('usability')) {
      strengths.push('UI/UX design');
      score += 0.6;
    }
    
    if (response.includes('implementation') || response.includes('code example') || 
        response.includes('function') || response.includes('method')) {
      strengths.push('implementation details');
      score += 0.5;
    }
    
    if (response.includes('test') || response.includes('edge case') || 
        response.includes('validation') || response.includes('error handling')) {
      strengths.push('testing & validation');
      score += 0.4;
    }
    
    if (response.includes('creative') || response.includes('innovative') || 
        response.includes('unique approach') || response.includes('solution')) {
      strengths.push('creative solutions');
      score += 0.5;
    }
    
    if (response.includes('deploy') || response.includes('CI/CD') || 
        response.includes('containerization') || response.includes('cloud')) {
      strengths.push('deployment considerations');
      score += 0.3;
    }
  }
  
  // Add base strength if nothing specific was detected
  if (strengths.length === 0) {
    strengths.push(model === 'llama3' ? 'general reasoning' : 'general creativity');
  }
  
  return { score, areas: strengths };
}

/**
 * Parse a response into logical sections
 */
function parseResponseIntoSections(response: string): Record<string, string> {
  const sections: Record<string, string> = {};
  
  // Try to identify introduction (first paragraph)
  const paragraphs = response.split('\n\n');
  if (paragraphs.length > 0) {
    sections.introduction = paragraphs[0];
  }
  
  // Try to identify conclusion (last paragraph)
  if (paragraphs.length > 1) {
    sections.conclusion = paragraphs[paragraphs.length - 1];
  }
  
  // Look for architecture section
  const architectureMatch = response.match(/(?:Architecture|System Design|Component Structure)[\s\S]*?(?=\n\n\w|$)/i);
  if (architectureMatch) {
    sections.architecture = architectureMatch[0];
  }
  
  // Look for implementation section
  const implementationMatch = response.match(/(?:Implementation|Code Example|Development Process)[\s\S]*?(?=\n\n\w|$)/i);
  if (implementationMatch) {
    sections.implementation = implementationMatch[0];
  }
  
  return sections;
}

/**
 * Select the better section from two options
 */
function selectBetterSection(section1: string, section2: string, sectionType: string): string {
  // For introduction, prefer concise but informative
  if (sectionType === 'introduction') {
    return section1.length < section2.length ? section1 : section2;
  }
  
  // For conclusion, prefer the more comprehensive one
  if (sectionType === 'conclusion') {
    return section1.length > section2.length ? section1 : section2;
  }
  
  // Default to section1 (Llama3)
  return section1;
}

/**
 * Integrate content from both responses when section parsing fails
 */
function integrateContent(response1: string, response2: string, strengths1: string[], strengths2: string[]): string {
  // Simple approach: alternate paragraphs while giving preference to each model's strengths
  const paragraphs1 = response1.split('\n\n');
  const paragraphs2 = response2.split('\n\n');
  
  const result: string[] = [];
  const maxLength = Math.max(paragraphs1.length, paragraphs2.length);
  
  for (let i = 0; i < maxLength; i++) {
    if (i < paragraphs1.length) {
      // Check if this paragraph aligns with Llama3's strengths
      const shouldAdd1 = strengths1.some(strength => 
        paragraphs1[i].toLowerCase().includes(strength.toLowerCase()));
      
      if (shouldAdd1 || i === 0) { // Always add introduction from Llama3
        result.push(paragraphs1[i]);
      }
    }
    
    if (i < paragraphs2.length) {
      // Check if this paragraph aligns with Gemma3's strengths
      const shouldAdd2 = strengths2.some(strength => 
        paragraphs2[i].toLowerCase().includes(strength.toLowerCase()));
      
      if (shouldAdd2 || i === paragraphs2.length - 1) { // Always add conclusion from Gemma3
        result.push(paragraphs2[i]);
      }
    }
  }
  
  return result.join('\n\n');
}

/**
 * Select the best response when using competitive mode
 */
function selectBestResponse(
  response1: string, 
  response2: string, 
  prompt: string
): [string, any] {
  // Simple metrics: length, specificity, relevance
  const metrics = {
    llama3_contribution: 0,
    gemma3_contribution: 0,
    reasoning_steps: []
  };
  
  // Calculate scores
  const length1 = response1.length;
  const length2 = response2.length;
  
  const specificityScore1 = calculateSpecificityScore(response1);
  const specificityScore2 = calculateSpecificityScore(response2);
  
  const relevanceScore1 = calculateRelevanceScore(response1, prompt);
  const relevanceScore2 = calculateRelevanceScore(response2, prompt);
  
  // Weighted total score
  const totalScore1 = (0.2 * length1/1000) + (0.4 * specificityScore1) + (0.4 * relevanceScore1);
  const totalScore2 = (0.2 * length2/1000) + (0.4 * specificityScore2) + (0.4 * relevanceScore2);
  
  metrics.reasoning_steps.push(
    `Llama3 scores - Length: ${(length1/1000).toFixed(2)}, Specificity: ${specificityScore1.toFixed(2)}, Relevance: ${relevanceScore1.toFixed(2)}`
  );
  metrics.reasoning_steps.push(
    `Gemma3 scores - Length: ${(length2/1000).toFixed(2)}, Specificity: ${specificityScore2.toFixed(2)}, Relevance: ${relevanceScore2.toFixed(2)}`
  );
  metrics.reasoning_steps.push(
    `Llama3 total score: ${totalScore1.toFixed(2)}`
  );
  metrics.reasoning_steps.push(
    `Gemma3 total score: ${totalScore2.toFixed(2)}`
  );
  
  if (totalScore1 > totalScore2) {
    metrics.llama3_contribution = 1;
    metrics.gemma3_contribution = 0;
    metrics.reasoning_steps.push('Selected Llama3 response based on higher total score');
    return [response1, metrics];
  } else {
    metrics.llama3_contribution = 0;
    metrics.gemma3_contribution = 1;
    metrics.reasoning_steps.push('Selected Gemma3 response based on higher total score');
    return [response2, metrics];
  }
}

/**
 * Calculate specificity score of a response
 */
function calculateSpecificityScore(response: string): number {
  // Count technical terms, code snippets, specific numbers
  const technicalTermsCount = (response.match(/API|database|schema|component|function|class|interface|module|system|architecture/gi) || []).length;
  const codeSnippetsCount = (response.match(/```[\s\S]*?```/g) || []).length;
  const numbersCount = (response.match(/\d+(\.\d+)?/g) || []).length;
  
  // Calculate normalized score (0-1)
  return Math.min((technicalTermsCount * 0.1) + (codeSnippetsCount * 0.5) + (numbersCount * 0.05), 1);
}

/**
 * Calculate relevance score of a response to the prompt
 */
function calculateRelevanceScore(response: string, prompt: string): number {
  // Extract keywords from prompt
  const promptKeywords = prompt.toLowerCase().split(/\W+/).filter(word => 
    word.length > 3 && !['this', 'that', 'with', 'from', 'about', 'would', 'should', 'could'].includes(word)
  );
  
  // Count keyword occurrences in response
  let keywordMatches = 0;
  for (const keyword of promptKeywords) {
    const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
    const matches = (response.match(regex) || []).length;
    keywordMatches += matches;
  }
  
  // Calculate normalized score (0-1)
  return Math.min(keywordMatches / (promptKeywords.length || 1), 1);
}

/**
 * Determine if a task is primarily architectural or implementation-focused
 */
function isArchitecturalTask(prompt: string): boolean {
  const architecturalKeywords = [
    'architecture', 'design', 'structure', 'system', 'database schema', 'model',
    'pattern', 'abstract', 'concept', 'diagram', 'organize', 'components',
    'service', 'infrastructure', 'scalability', 'security'
  ];
  
  const implementationKeywords = [
    'implement', 'code', 'build', 'develop', 'create', 'make', 'write',
    'function', 'class', 'method', 'interface', 'programming', 'testing',
    'debugging', 'execution', 'deployment', 'UI', 'frontend', 'styling'
  ];
  
  // Count occurrences of each type of keyword
  const archCount = architecturalKeywords.reduce((count, keyword) => 
    count + (prompt.toLowerCase().includes(keyword) ? 1 : 0), 0);
  
  const implCount = implementationKeywords.reduce((count, keyword) => 
    count + (prompt.toLowerCase().includes(keyword) ? 1 : 0), 0);
  
  // Compare counts to determine focus
  return archCount >= implCount;
}

/**
 * Add attribution about which models contributed to the response
 */
function addModelAttribution(response: string, collaborativeData: any): string {
  const llama3Percentage = Math.round(collaborativeData.llama3_contribution * 100);
  const gemma3Percentage = Math.round(collaborativeData.gemma3_contribution * 100);
  
  const attribution = `\n\n---\n*This response was generated by a hybrid AI system combining Llama3 (${llama3Percentage}%) and Gemma3 (${gemma3Percentage}%) models.*`;
  
  return response + attribution;
}

/**
 * Generate a unique request ID for tracking
 */
function generateRequestId(prefix = 'hybrid'): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`;
}

/**
 * Estimate token count from text (approximate)
 */
function estimateTokenCount(text: string): number {
  // Rough approximation: ~4 chars per token on average for English text
  return Math.ceil(text.length / 4);
}