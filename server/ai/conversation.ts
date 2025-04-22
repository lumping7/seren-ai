/**
 * AI-to-AI Conversational Framework
 * 
 * Enables human-like conversational interactions between the Llama3 and Gemma3 models
 * where they can question, challenge, and build upon each other's ideas.
 */

import { llamaHandler } from './llama';
import { gemmaHandler } from './gemma';
import { storage } from '../storage';
import { v4 as uuidv4 } from 'uuid';
import { resourceManager } from './resource-manager';

// Types of inter-model conversations
type ConversationMode = 'debate' | 'collaborative' | 'critical' | 'brainstorming';

// Structure for a single turn in the conversation
interface ConversationTurn {
  model: 'llama3' | 'gemma3';
  content: string;
  timestamp: Date;
  metadata?: {
    confidence?: number;
    questionCount?: number;
    criticalPoints?: number;
    suggestions?: number;
  };
}

// Full conversation between models
interface ModelConversation {
  id: string;
  topic: string;
  mode: ConversationMode;
  turns: ConversationTurn[];
  startedAt: Date;
  endedAt?: Date;
  userPrompt: string;
  conclusion?: string;
}

/**
 * Conversation Manager Class
 * 
 * Handles the conversation flow between AI models, ensuring they interact
 * in a human-like manner, questioning assumptions and building on ideas.
 */
export class ConversationManager {
  private static instance: ConversationManager;
  private activeConversations: Map<string, ModelConversation> = new Map();
  
  private constructor() {}
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): ConversationManager {
    if (!ConversationManager.instance) {
      ConversationManager.instance = new ConversationManager();
    }
    return ConversationManager.instance;
  }
  
  /**
   * Start a new conversation between models on a given topic
   */
  public async startConversation(
    topic: string,
    userPrompt: string,
    mode: ConversationMode = 'collaborative',
    userId?: number
  ): Promise<string> {
    const conversationId = uuidv4();
    
    const conversation: ModelConversation = {
      id: conversationId,
      topic,
      mode,
      turns: [],
      startedAt: new Date(),
      userPrompt
    };
    
    this.activeConversations.set(conversationId, conversation);
    
    // Start the conversation with the Llama3 model (architect) 
    await this.addTurn(
      conversationId, 
      'llama3',
      this.createModelPrompt('llama3', topic, userPrompt, mode, []),
      userId
    );
    
    return conversationId;
  }
  
  /**
   * Add a turn to the conversation
   */
  private async addTurn(
    conversationId: string,
    model: 'llama3' | 'gemma3',
    prompt: string,
    userId?: number
  ): Promise<void> {
    const conversation = this.activeConversations.get(conversationId);
    if (!conversation) {
      throw new Error(`Conversation with ID ${conversationId} not found`);
    }
    
    // Create mock objects to avoid circular references
    const mockReq = {
      body: {
        prompt,
        options: {
          temperature: this.getTemperatureForMode(conversation.mode),
          max_tokens: resourceManager.getLimits().maxTokens,
        },
        includeModelDetails: false
      },
      isAuthenticated: () => userId !== undefined,
      user: userId ? { id: userId } : undefined
    };
    
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
      // Call the appropriate model handler
      if (model === 'llama3') {
        await llamaHandler(mockReq as any, mockRes as any);
      } else {
        await gemmaHandler(mockReq as any, mockRes as any);
      }
      
      const response = mockRes.getData();
      
      // Add turn to conversation
      const turn: ConversationTurn = {
        model,
        content: response.generated_text,
        timestamp: new Date(),
        metadata: {
          questionCount: this.countQuestions(response.generated_text),
          criticalPoints: this.countCriticalPoints(response.generated_text),
          suggestions: this.countSuggestions(response.generated_text)
        }
      };
      
      conversation.turns.push(turn);
      
      // Store in database if user is authenticated
      if (userId) {
        try {
          await storage.createMessage({
            conversationId,
            role: model === 'llama3' ? 'architect' : 'builder',
            content: response.generated_text,
            model,
            userId,
            metadata: {
              ...turn.metadata,
              timestamp: turn.timestamp.toISOString()
            }
          });
        } catch (error: any) {
          console.error(`[Conversation] Failed to store message: ${error.message || 'Unknown error'}`);
        }
      }
      
      // Continue conversation by alternating models if we haven't reached the end
      if (conversation.turns.length < this.getMaxTurnsForMode(conversation.mode)) {
        const nextModel = model === 'llama3' ? 'gemma3' : 'llama3';
        const nextPrompt = this.createModelPrompt(
          nextModel,
          conversation.topic,
          conversation.userPrompt,
          conversation.mode,
          conversation.turns
        );
        
        // Add a small delay to make the conversation feel more natural
        setTimeout(() => {
          this.addTurn(conversationId, nextModel, nextPrompt, userId);
        }, 1000);
      } else {
        // Finalize conversation with a conclusion
        conversation.endedAt = new Date();
        conversation.conclusion = this.generateConclusion(conversation);
        
        // Store conclusion if user is authenticated
        if (userId) {
          try {
            await storage.createMessage({
              conversationId,
              role: 'conclusion',
              content: conversation.conclusion,
              model: 'hybrid',
              userId,
              metadata: {
                timestamp: new Date().toISOString()
              }
            });
          } catch (error: any) {
            console.error(`[Conversation] Failed to store conclusion: ${error.message || 'Unknown error'}`);
          }
        }
      }
    } catch (error: any) {
      console.error(`[Conversation] Error in turn: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Get a conversation by ID
   */
  public getConversation(conversationId: string): ModelConversation | undefined {
    return this.activeConversations.get(conversationId);
  }
  
  /**
   * Check if conversation is complete
   */
  public isConversationComplete(conversationId: string): boolean {
    const conversation = this.activeConversations.get(conversationId);
    return conversation ? conversation.endedAt !== undefined : false;
  }
  
  /**
   * Create a tailored prompt for the next model in the conversation
   */
  private createModelPrompt(
    model: 'llama3' | 'gemma3',
    topic: string,
    userPrompt: string,
    mode: ConversationMode,
    previousTurns: ConversationTurn[]
  ): string {
    // Base context for each model
    const baseContext = model === 'llama3' 
      ? "You are the architecture expert (Llama3) in a collaborative AI team. Your role is to design systems, plan implementation details, and provide technical specifications."
      : "You are the creative implementation expert (Gemma3) in a collaborative AI team. Your role is to implement systems designed by the architecture expert, add creative solutions, and ensure the final product is user-friendly.";
    
    // Define conversation context based on mode
    let conversationContext = "";
    
    switch (mode) {
      case 'debate':
        conversationContext = "You are engaged in a constructive debate. Challenge assumptions, ask critical questions, and provide counterarguments supported by evidence. Be respectful but firm in your position.";
        break;
      case 'collaborative':
        conversationContext = "You are collaboratively solving a problem. Build upon previous points, ask clarifying questions, and suggest improvements or extensions to ideas. Be supportive while adding value.";
        break;
      case 'critical':
        conversationContext = "You are conducting a critical analysis. Identify potential flaws, edge cases, security concerns, or scaling issues. Suggest rigorous improvements based on best practices.";
        break;
      case 'brainstorming':
        conversationContext = "You are in a creative brainstorming session. Generate innovative ideas, explore unconventional approaches, and make unexpected connections. Quantity of ideas is valued alongside quality.";
        break;
    }
    
    // Start with system context
    let prompt = `[SYSTEM: ${baseContext} ${conversationContext}]\n\n`;
    
    // Add the original user query
    prompt += `Original request: ${userPrompt}\n\n`;
    
    // Add conversation history if any
    if (previousTurns.length > 0) {
      prompt += "Conversation so far:\n";
      
      for (const turn of previousTurns) {
        const roleName = turn.model === 'llama3' ? "Architecture Expert (Llama3)" : "Implementation Expert (Gemma3)";
        prompt += `${roleName}: ${turn.content}\n\n`;
      }
    }
    
    // Add specific instructions for this turn
    if (previousTurns.length === 0) {
      // First turn
      prompt += model === 'llama3'
        ? "As the Architecture Expert, start by outlining a high-level approach to this problem. Include key components, data structures, and architectural considerations."
        : "As the Implementation Expert, start by suggesting practical implementation details, coding patterns, and user experience considerations.";
    } else if (model === 'llama3') {
      // Llama3 responding
      prompt += "As the Architecture Expert, respond to the Implementation Expert's points. Address any technical questions, clarify architectural decisions, and suggest improvements. Be sure to ask at least 2 challenging questions about the implementation approach.";
    } else {
      // Gemma3 responding
      prompt += "As the Implementation Expert, respond to the Architecture Expert's points. Suggest concrete implementation details, highlight practical concerns, and ask questions about any unclear architectural aspects. Be sure to propose at least 2 creative solutions to the problems identified.";
    }
    
    // Add specific behavior guidance based on conversation mode
    switch (mode) {
      case 'debate':
        prompt += " Find at least one point to respectfully disagree with and explain why.";
        break;
      case 'collaborative':
        prompt += " Focus on finding synergies between architectural vision and implementation reality.";
        break;
      case 'critical':
        prompt += " Highlight at least two potential weaknesses in the current approach and suggest mitigations.";
        break;
      case 'brainstorming':
        prompt += " Introduce at least one unconventional idea that might lead to a breakthrough solution.";
        break;
    }
    
    return prompt;
  }
  
  /**
   * Generate a conclusion for the conversation
   */
  private generateConclusion(conversation: ModelConversation): string {
    // Extract key points from the conversation
    const keyPoints = this.extractKeyPoints(conversation);
    
    // Create conclusion based on conversation mode
    switch (conversation.mode) {
      case 'debate':
        return `After a thoughtful debate between the Architecture Expert and Implementation Expert, several key perspectives emerged:\n\n${keyPoints}\n\nWhile some differences in approach remain, this exchange has illuminated important considerations from both architectural and implementation perspectives.`;
      
      case 'collaborative':
        return `Through collaborative discussion between the Architecture Expert and Implementation Expert, a comprehensive solution has emerged:\n\n${keyPoints}\n\nBy combining architectural vision with implementation expertise, this approach addresses both theoretical and practical concerns.`;
      
      case 'critical':
        return `After critical analysis by both the Architecture Expert and Implementation Expert, the following insights and concerns have been identified:\n\n${keyPoints}\n\nAddressing these critical points will be essential for creating a robust, production-ready solution.`;
      
      case 'brainstorming':
        return `The brainstorming session between the Architecture Expert and Implementation Expert has generated several innovative ideas:\n\n${keyPoints}\n\nThese creative approaches offer multiple pathways forward, each with unique advantages and considerations.`;
    }
  }
  
  /**
   * Extract key points from the conversation
   */
  private extractKeyPoints(conversation: ModelConversation): string {
    let points = "";
    
    // Extract statements that appear to be key points or conclusions
    for (const turn of conversation.turns) {
      const roleName = turn.model === 'llama3' ? "Architecture Expert" : "Implementation Expert";
      
      // Look for statements that summarize or highlight important considerations
      const content = turn.content;
      const lines = content.split('\n');
      
      for (const line of lines) {
        // Identify likely key points using linguistic patterns
        if (line.includes('important') || 
            line.includes('key') || 
            line.includes('critical') || 
            line.includes('essential') ||
            line.includes('must') ||
            line.includes('should consider') ||
            line.match(/^[\d\*\-\•]+\s+[A-Z]/) || // Bullet points starting with capital letter
            line.match(/^In summary/) ||
            line.match(/^To conclude/)) {
          points += `• ${roleName}: "${line.trim()}"\n\n`;
        }
      }
    }
    
    // If we didn't find enough key points, extract some generic ones
    if (points.split('\n').length < 4) {
      for (const turn of conversation.turns) {
        const roleName = turn.model === 'llama3' ? "Architecture Expert" : "Implementation Expert";
        const content = turn.content;
        
        // Take first sentence of each paragraph that's long enough to be substantive
        const paragraphs = content.split('\n\n');
        for (const paragraph of paragraphs) {
          if (paragraph.length > 100) {
            const firstSentence = paragraph.split('.')[0];
            if (firstSentence.length > 50) {
              points += `• ${roleName}: "${firstSentence.trim()}"\n\n`;
              break;
            }
          }
        }
      }
    }
    
    return points || "Both experts shared valuable insights throughout the conversation.";
  }
  
  /**
   * Get appropriate temperature based on conversation mode
   */
  private getTemperatureForMode(mode: ConversationMode): number {
    switch (mode) {
      case 'debate': return 0.7;  // Slightly higher for diverse opinions
      case 'collaborative': return 0.6;  // Moderate for balanced responses
      case 'critical': return 0.4;  // Lower for more focused/analytical responses
      case 'brainstorming': return 0.8;  // Higher for more creative responses
    }
  }
  
  /**
   * Get maximum conversation turns based on mode
   */
  private getMaxTurnsForMode(mode: ConversationMode): number {
    switch (mode) {
      case 'debate': return 6;  // Three turns each for thorough debate
      case 'collaborative': return 4;  // Two turns each for concise collaboration
      case 'critical': return 4;  // Two turns each for focused analysis
      case 'brainstorming': return 6;  // Three turns each for idea generation
    }
  }
  
  /**
   * Count questions in a response to evaluate engagement
   */
  private countQuestions(text: string): number {
    return (text.match(/\?/g) || []).length;
  }
  
  /**
   * Count critical points in a response
   */
  private countCriticalPoints(text: string): number {
    const criticalPhrases = [
      'however', 'but', 'although', 'challenge', 'problem', 'issue',
      'concern', 'risk', 'weakness', 'drawback', 'limitation'
    ];
    
    return criticalPhrases.reduce((count, phrase) => {
      const regex = new RegExp(`\\b${phrase}\\b`, 'gi');
      return count + (text.match(regex) || []).length;
    }, 0);
  }
  
  /**
   * Count suggestions in a response
   */
  private countSuggestions(text: string): number {
    const suggestionPhrases = [
      'suggest', 'recommend', 'propose', 'perhaps', 'consider',
      'might', 'could', 'would be better', 'alternative', 'instead'
    ];
    
    return suggestionPhrases.reduce((count, phrase) => {
      const regex = new RegExp(`\\b${phrase}\\b`, 'gi');
      return count + (text.match(regex) || []).length;
    }, 0);
  }
}

// Singleton instance
export const conversationManager = ConversationManager.getInstance();