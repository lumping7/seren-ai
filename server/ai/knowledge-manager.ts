/**
 * Dynamic Knowledge Integration System
 * 
 * Enables the AI models to continuously learn and adapt by:
 * 1. Building knowledge libraries that persist over time
 * 2. Automatically capturing valuable insights from interactions
 * 3. Allowing manual knowledge injection from users
 * 4. Using knowledge similarity search for relevant retrieval
 */

import { db } from '../db';
import { aql, eq, inArray, desc, and, sql } from 'drizzle-orm';
import { storage } from '../storage';
import { v4 as uuidv4 } from 'uuid';
import { 
  aiKnowledgeBase, 
  aiKnowledgeDomains, 
  aiKnowledgeRelations, 
  KnowledgeEntry, 
  KnowledgeDomain,
  KnowledgeRelation,
  KnowledgeSource
} from '@shared/schema';

/**
 * Knowledge Manager Class
 * 
 * Handles knowledge acquisition, storage, retrieval, and integration
 * with self-learning capabilities for continuous improvement.
 */
export class KnowledgeManager {
  private static instance: KnowledgeManager;
  private cachedDomains: Map<string, KnowledgeDomain> = new Map();
  private lastCacheRefresh: number = 0;
  private readonly CACHE_TTL = 60 * 60 * 1000; // 1 hour cache lifetime
  
  private constructor() {
    // Initialize domains cache on startup
    this.refreshDomainCache();
    
    // Set up periodic cleanup of low-value knowledge
    setInterval(() => this.cleanupLowValueKnowledge(), 24 * 60 * 60 * 1000); // Daily
  }
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): KnowledgeManager {
    if (!KnowledgeManager.instance) {
      KnowledgeManager.instance = new KnowledgeManager();
    }
    return KnowledgeManager.instance;
  }
  
  /**
   * Add new knowledge to the system
   */
  public async addKnowledge(
    content: string,
    source: KnowledgeSource,
    domains: string[],
    userId?: number,
    metadata: any = {},
    importanceScore: number = 0.5
  ): Promise<KnowledgeEntry> {
    try {
      // Check if similar knowledge already exists to prevent duplicates
      const similarEntries = await this.findSimilarKnowledge(content, 0.85);
      
      if (similarEntries.length > 0) {
        // Update existing knowledge instead of creating duplicate
        const existingEntry = similarEntries[0];
        
        // Increase importance score if we're seeing this again
        const newImportanceScore = Math.min(existingEntry.importanceScore + 0.1, 1.0);
        
        // Update the entry with new information
        const [updatedEntry] = await db.update(aiKnowledgeBase)
          .set({
            content: this.mergeContents(existingEntry.content, content),
            lastAccessedAt: new Date(),
            accessCount: existingEntry.accessCount + 1,
            importanceScore: newImportanceScore,
            metadata: {
              ...existingEntry.metadata,
              ...metadata,
              sources: [...(existingEntry.metadata.sources || []), source]
            }
          })
          .where(eq(aiKnowledgeBase.id, existingEntry.id))
          .returning();
        
        console.log(`[KnowledgeManager] Updated existing knowledge entry: ${updatedEntry.id}`);
        
        // Update domain associations if new domains provided
        await this.updateKnowledgeDomains(updatedEntry.id, domains);
        
        return updatedEntry;
      } else {
        // Create new knowledge entry
        const domainIds = await this.resolveDomainIds(domains);
        
        // Calculate initial importance score based on source and content
        const calculatedScore = this.calculateImportanceScore(content, source, metadata);
        const finalScore = Math.max(importanceScore, calculatedScore);
        
        // Create the knowledge entry
        const [knowledgeEntry] = await db.insert(aiKnowledgeBase)
          .values({
            content,
            source,
            createdBy: userId,
            importanceScore: finalScore,
            metadata: {
              ...metadata,
              sources: [source],
              initialImportance: finalScore
            }
          })
          .returning();
        
        console.log(`[KnowledgeManager] Created new knowledge entry: ${knowledgeEntry.id}`);
        
        // Associate with domains
        await this.addKnowledgeToDomains(knowledgeEntry.id, domainIds);
        
        // Detect and create relationships between knowledge entries
        await this.detectAndCreateRelationships(knowledgeEntry.id, content);
        
        return knowledgeEntry;
      }
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error adding knowledge: ${error.message || 'Unknown error'}`);
      throw error;
    }
  }
  
  /**
   * Add user-provided knowledge with manual domain tagging
   */
  public async addUserKnowledge(
    content: string,
    domains: string[],
    userId: number,
    metadata: any = {},
    importanceScore: number = 0.8  // User knowledge starts with higher importance
  ): Promise<KnowledgeEntry> {
    return this.addKnowledge(
      content,
      'user',
      domains,
      userId,
      {
        ...metadata,
        userProvided: true,
        timestamp: new Date().toISOString()
      },
      importanceScore
    );
  }
  
  /**
   * Extract and save knowledge from conversation
   * Automatically identifies valuable information worth saving
   */
  public async extractKnowledgeFromConversation(
    conversationId: string,
    userId?: number
  ): Promise<KnowledgeEntry[]> {
    try {
      // Retrieve conversation messages
      const messages = await storage.getMessages(conversationId);
      if (!messages || messages.length === 0) {
        return [];
      }
      
      const extractedEntries: KnowledgeEntry[] = [];
      
      // Process each message for potential knowledge
      for (const message of messages) {
        // Skip user messages - we're looking for AI-generated knowledge
        if (message.role === 'user') continue;
        
        // Extract knowledge nuggets from message content
        const knowledgeNuggets = this.extractKnowledgeNuggets(message.content);
        
        // Save each valuable nugget
        for (const nugget of knowledgeNuggets) {
          // Determine domains automatically
          const domains = this.detectDomains(nugget.content);
          
          // Calculate importance based on context
          const importance = nugget.importance * 
            (message.role === 'conclusion' ? 1.2 : 1.0) * // Conclusions are more important
            (message.metadata?.criticalPoints > 2 ? 1.1 : 1.0); // Critical analysis is valuable
          
          // Add to knowledge base
          const entry = await this.addKnowledge(
            nugget.content,
            'conversation',
            domains,
            userId,
            {
              conversationId,
              messageId: message.id,
              modelRole: message.role,
              model: message.model,
              extractionMethod: 'automatic',
              extractionConfidence: nugget.confidence
            },
            importance
          );
          
          extractedEntries.push(entry);
        }
      }
      
      return extractedEntries;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error extracting knowledge: ${error.message || 'Unknown error'}`);
      return [];
    }
  }
  
  /**
   * Retrieve relevant knowledge for a given query
   */
  public async retrieveRelevantKnowledge(
    query: string,
    domains?: string[],
    limit: number = 5,
    minImportance: number = 0.3
  ): Promise<KnowledgeEntry[]> {
    try {
      // If cache is stale, refresh domains
      if (Date.now() - this.lastCacheRefresh > this.CACHE_TTL) {
        await this.refreshDomainCache();
      }
      
      // Get domain IDs if domains specified
      let domainIds: number[] = [];
      if (domains && domains.length > 0) {
        domainIds = await this.resolveDomainIds(domains);
      }
      
      // Base query for knowledge retrieval
      let knowledgeQuery = db.select()
        .from(aiKnowledgeBase)
        .where(
          and(
            sql`${aiKnowledgeBase.importanceScore} >= ${minImportance}`,
            sql`similarity(${aiKnowledgeBase.content}, ${query}) > 0.2`
          )
        )
        .orderBy(
          sql`similarity(${aiKnowledgeBase.content}, ${query}) DESC`
        )
        .limit(limit);
      
      // Add domain filter if specified
      if (domainIds.length > 0) {
        const entriesInDomains = db.select({ knowledgeId: aiKnowledgeDomains.knowledgeId })
          .from(aiKnowledgeDomains)
          .where(inArray(aiKnowledgeDomains.domainId, domainIds));
        
        knowledgeQuery = knowledgeQuery
          .where(inArray(aiKnowledgeBase.id, entriesInDomains));
      }
      
      // Execute query
      const results = await knowledgeQuery;
      
      // Update access statistics
      for (const entry of results) {
        this.updateKnowledgeUsage(entry.id).catch(err => {
          console.error(`[KnowledgeManager] Error updating knowledge usage: ${err.message || 'Unknown error'}`);
        });
      }
      
      return results;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error retrieving knowledge: ${error.message || 'Unknown error'}`);
      return [];
    }
  }
  
  /**
   * Find knowledge entries similar to the provided content
   */
  public async findSimilarKnowledge(
    content: string,
    threshold: number = 0.7,
    limit: number = 3
  ): Promise<KnowledgeEntry[]> {
    try {
      const results = await db.select()
        .from(aiKnowledgeBase)
        .where(
          sql`similarity(${aiKnowledgeBase.content}, ${content}) > ${threshold}`
        )
        .orderBy(
          sql`similarity(${aiKnowledgeBase.content}, ${content}) DESC`
        )
        .limit(limit);
      
      return results;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error finding similar knowledge: ${error.message || 'Unknown error'}`);
      return [];
    }
  }
  
  /**
   * Create a new knowledge domain
   */
  public async createDomain(
    name: string,
    description: string,
    userId?: number
  ): Promise<KnowledgeDomain> {
    try {
      // Check if domain already exists
      const existingDomains = await db.select()
        .from(aiKnowledgeDomains)
        .where(eq(aiKnowledgeDomains.name, name));
      
      if (existingDomains.length > 0) {
        return existingDomains[0];
      }
      
      // Create new domain
      const [domain] = await db.insert(aiKnowledgeDomains)
        .values({
          name,
          description,
          createdBy: userId
        })
        .returning();
      
      // Update cache
      this.cachedDomains.set(name.toLowerCase(), domain);
      
      return domain;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error creating domain: ${error.message || 'Unknown error'}`);
      throw error;
    }
  }
  
  /**
   * Get all knowledge domains
   */
  public async getDomains(): Promise<KnowledgeDomain[]> {
    try {
      const domains = await db.select().from(aiKnowledgeDomains);
      return domains;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error getting domains: ${error.message || 'Unknown error'}`);
      return [];
    }
  }
  
  /**
   * Get knowledge entries for a specific domain
   */
  public async getKnowledgeByDomain(
    domain: string,
    limit: number = 100
  ): Promise<KnowledgeEntry[]> {
    try {
      // Find domain ID
      const domainId = await this.resolveDomainId(domain);
      if (!domainId) {
        return [];
      }
      
      // Find all knowledge entries in this domain
      const domainEntries = await db.select({ knowledgeId: aiKnowledgeDomains.knowledgeId })
        .from(aiKnowledgeDomains)
        .where(eq(aiKnowledgeDomains.domainId, domainId));
      
      if (domainEntries.length === 0) {
        return [];
      }
      
      // Get the actual knowledge entries
      const knowledgeIds = domainEntries.map(entry => entry.knowledgeId);
      
      const knowledge = await db.select()
        .from(aiKnowledgeBase)
        .where(inArray(aiKnowledgeBase.id, knowledgeIds))
        .orderBy(desc(aiKnowledgeBase.importanceScore))
        .limit(limit);
      
      return knowledge;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error getting knowledge by domain: ${error.message || 'Unknown error'}`);
      return [];
    }
  }
  
  /**
   * Get related knowledge entries
   */
  public async getRelatedKnowledge(
    knowledgeId: number,
    limit: number = 5
  ): Promise<KnowledgeEntry[]> {
    try {
      // Find relations
      const relations = await db.select({ targetId: aiKnowledgeRelations.targetId })
        .from(aiKnowledgeRelations)
        .where(eq(aiKnowledgeRelations.sourceId, knowledgeId))
        .orderBy(desc(aiKnowledgeRelations.strength))
        .limit(limit);
      
      if (relations.length === 0) {
        return [];
      }
      
      // Get related knowledge entries
      const relatedIds = relations.map(relation => relation.targetId);
      
      const relatedKnowledge = await db.select()
        .from(aiKnowledgeBase)
        .where(inArray(aiKnowledgeBase.id, relatedIds));
      
      return relatedKnowledge;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error getting related knowledge: ${error.message || 'Unknown error'}`);
      return [];
    }
  }
  
  /**
   * Create a knowledge graph for visualization
   */
  public async createKnowledgeGraph(
    centralDomain?: string,
    depth: number = 2,
    maxNodes: number = 50
  ): Promise<any> {
    try {
      // Create graph structure
      const graph = {
        nodes: [],
        edges: []
      };
      
      // Start with central domain if provided
      let startingNodes: KnowledgeEntry[] = [];
      
      if (centralDomain) {
        // Get knowledge from the central domain
        startingNodes = await this.getKnowledgeByDomain(centralDomain, 10);
      } else {
        // Start with most important knowledge
        startingNodes = await db.select()
          .from(aiKnowledgeBase)
          .orderBy(desc(aiKnowledgeBase.importanceScore))
          .limit(10);
      }
      
      // Track processed nodes to avoid duplicates
      const processedNodes = new Set<number>();
      
      // Process the graph
      for (const node of startingNodes) {
        await this.expandGraphNode(graph, node, processedNodes, depth, maxNodes);
        
        // Stop if we've reached max nodes
        if (processedNodes.size >= maxNodes) {
          break;
        }
      }
      
      return graph;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error creating knowledge graph: ${error.message || 'Unknown error'}`);
      return { nodes: [], edges: [] };
    }
  }
  
  /**
   * Enhance a prompt with relevant knowledge
   */
  public async enhancePromptWithKnowledge(
    prompt: string,
    domains?: string[]
  ): Promise<string> {
    try {
      // Retrieve relevant knowledge
      const relevantKnowledge = await this.retrieveRelevantKnowledge(prompt, domains, 3);
      
      if (relevantKnowledge.length === 0) {
        return prompt;
      }
      
      // Build context section
      let knowledgeContext = "\n\nRelevant knowledge from system:\n";
      
      for (const [index, entry] of relevantKnowledge.entries()) {
        knowledgeContext += `[${index + 1}] ${entry.content.trim()}\n\n`;
      }
      
      // Add instruction to use knowledge
      knowledgeContext += "Use the above relevant knowledge where appropriate in your response.\n\n";
      
      // Add context to beginning of prompt
      return knowledgeContext + prompt;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error enhancing prompt: ${error.message || 'Unknown error'}`);
      return prompt; // Return original prompt on error
    }
  }
  
  /******************************************
   * PRIVATE METHODS
   ******************************************/
  
  /**
   * Update a knowledge entry's usage statistics
   */
  private async updateKnowledgeUsage(knowledgeId: number): Promise<void> {
    try {
      await db.update(aiKnowledgeBase)
        .set({
          lastAccessedAt: new Date(),
          accessCount: sql`${aiKnowledgeBase.accessCount} + 1`
        })
        .where(eq(aiKnowledgeBase.id, knowledgeId));
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error updating knowledge usage: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Resolve domain names to IDs, creating domains if they don't exist
   */
  private async resolveDomainIds(domains: string[]): Promise<number[]> {
    const domainIds: number[] = [];
    
    for (const domain of domains) {
      const domainId = await this.resolveDomainId(domain);
      if (domainId) {
        domainIds.push(domainId);
      }
    }
    
    return domainIds;
  }
  
  /**
   * Resolve a single domain name to ID, creating if it doesn't exist
   */
  private async resolveDomainId(domain: string): Promise<number | null> {
    const normalizedDomain = domain.toLowerCase().trim();
    
    // Check cache first
    if (this.cachedDomains.has(normalizedDomain)) {
      return this.cachedDomains.get(normalizedDomain)!.id;
    }
    
    // Check database
    const [existingDomain] = await db.select()
      .from(aiKnowledgeDomains)
      .where(sql`LOWER(${aiKnowledgeDomains.name}) = ${normalizedDomain}`);
    
    if (existingDomain) {
      // Update cache
      this.cachedDomains.set(normalizedDomain, existingDomain);
      return existingDomain.id;
    }
    
    // Create new domain
    try {
      const [newDomain] = await db.insert(aiKnowledgeDomains)
        .values({
          name: domain.trim(),
          description: `Auto-generated domain for: ${domain.trim()}`
        })
        .returning();
      
      // Update cache
      this.cachedDomains.set(normalizedDomain, newDomain);
      return newDomain.id;
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error creating domain: ${error.message || 'Unknown error'}`);
      return null;
    }
  }
  
  /**
   * Add knowledge to multiple domains
   */
  private async addKnowledgeToDomains(knowledgeId: number, domainIds: number[]): Promise<void> {
    try {
      if (domainIds.length === 0) return;
      
      const values = domainIds.map(domainId => ({
        knowledgeId,
        domainId
      }));
      
      await db.insert(aiKnowledgeDomains)
        .values(values)
        .onConflictDoNothing();
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error adding knowledge to domains: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Update domain associations for a knowledge entry
   */
  private async updateKnowledgeDomains(knowledgeId: number, domains: string[]): Promise<void> {
    try {
      // Get domain IDs, creating if needed
      const domainIds = await this.resolveDomainIds(domains);
      
      // Add associations
      await this.addKnowledgeToDomains(knowledgeId, domainIds);
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error updating knowledge domains: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Refresh domain cache from database
   */
  private async refreshDomainCache(): Promise<void> {
    try {
      const domains = await db.select().from(aiKnowledgeDomains);
      
      // Clear and rebuild cache
      this.cachedDomains.clear();
      for (const domain of domains) {
        this.cachedDomains.set(domain.name.toLowerCase(), domain);
      }
      
      this.lastCacheRefresh = Date.now();
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error refreshing domain cache: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Cleanup low-value knowledge entries
   */
  private async cleanupLowValueKnowledge(): Promise<void> {
    try {
      // Find candidates for removal:
      // 1. Low importance score
      // 2. Not accessed in a long time
      // 3. Low access count
      const threshold = new Date();
      threshold.setMonth(threshold.getMonth() - 3); // 3 months old
      
      const lowValueEntries = await db.select({ id: aiKnowledgeBase.id })
        .from(aiKnowledgeBase)
        .where(
          and(
            sql`${aiKnowledgeBase.importanceScore} < 0.3`,
            sql`${aiKnowledgeBase.lastAccessedAt} < ${threshold}`,
            sql`${aiKnowledgeBase.accessCount} < 5`,
            sql`${aiKnowledgeBase.source} != 'user'` // Never auto-delete user-provided knowledge
          )
        )
        .limit(100);
      
      if (lowValueEntries.length === 0) {
        return;
      }
      
      const entryIds = lowValueEntries.map(entry => entry.id);
      
      // Archive rather than delete
      console.log(`[KnowledgeManager] Archiving ${entryIds.length} low-value knowledge entries`);
      
      await db.update(aiKnowledgeBase)
        .set({
          archived: true,
          metadata: sql`jsonb_set(${aiKnowledgeBase.metadata}, '{archived_reason}', '"low_value"')`
        })
        .where(inArray(aiKnowledgeBase.id, entryIds));
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error cleaning up knowledge: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Calculate importance score for new knowledge
   */
  private calculateImportanceScore(
    content: string,
    source: KnowledgeSource,
    metadata: any
  ): number {
    // Base score dependent on source
    let score = 0.5; // default
    
    switch (source) {
      case 'user':
        score = 0.8; // User-provided knowledge starts with higher importance
        break;
      case 'conversation':
        score = 0.6; // Knowledge from model conversations
        break;
      case 'document':
        score = 0.7; // Imported documents
        break;
      case 'api':
        score = 0.65; // External API data
        break;
    }
    
    // Adjust based on content features
    
    // Length factor (longer content might be more detailed/valuable)
    const lengthFactor = Math.min(content.length / 1000, 1) * 0.1;
    
    // Structure factor (well-structured content with sections/formatting)
    const hasStructure = /\n\n|\*|#|-|[0-9]\./g.test(content);
    const structureFactor = hasStructure ? 0.1 : 0;
    
    // Specificity factor (specific details like numbers, technical terms)
    const hasSpecifics = /\d+(\.\d+)?|technical|specific|important|critical|essential/gi.test(content);
    const specificityFactor = hasSpecifics ? 0.15 : 0;
    
    // Connection factor (references to other knowledge domains)
    const domainCount = this.detectDomains(content).length;
    const connectionFactor = Math.min(domainCount * 0.03, 0.15);
    
    // Final score capped at 1.0
    return Math.min(score + lengthFactor + structureFactor + specificityFactor + connectionFactor, 1.0);
  }
  
  /**
   * Detect relationships between knowledge entries
   */
  private async detectAndCreateRelationships(
    sourceId: number,
    content: string
  ): Promise<void> {
    try {
      // Find potential related knowledge
      const potentialRelated = await this.findSimilarKnowledge(content, 0.2, 10);
      
      // Skip self-relations
      const relatedEntries = potentialRelated.filter(entry => entry.id !== sourceId);
      
      if (relatedEntries.length === 0) {
        return;
      }
      
      // Create relationships
      for (const target of relatedEntries) {
        // Calculate relationship strength based on similarity
        const strength = this.calculateRelationshipStrength(content, target.content);
        
        if (strength > 0.2) { // Only create meaningful relationships
          await db.insert(aiKnowledgeRelations)
            .values({
              sourceId,
              targetId: target.id,
              strength,
              type: 'similarity',
              metadata: {
                createdAt: new Date().toISOString(),
                automatedDetection: true
              }
            })
            .onConflictDoUpdate({
              target: [
                aiKnowledgeRelations.sourceId,
                aiKnowledgeRelations.targetId
              ],
              set: {
                strength: sql`GREATEST(${aiKnowledgeRelations.strength}, ${strength})`,
                metadata: sql`jsonb_set(${aiKnowledgeRelations.metadata}, '{last_updated}', '"${new Date().toISOString()}"')`
              }
            });
          
          // Create reverse relationship too
          await db.insert(aiKnowledgeRelations)
            .values({
              sourceId: target.id,
              targetId: sourceId,
              strength,
              type: 'similarity',
              metadata: {
                createdAt: new Date().toISOString(),
                automatedDetection: true
              }
            })
            .onConflictDoUpdate({
              target: [
                aiKnowledgeRelations.sourceId,
                aiKnowledgeRelations.targetId
              ],
              set: {
                strength: sql`GREATEST(${aiKnowledgeRelations.strength}, ${strength})`,
                metadata: sql`jsonb_set(${aiKnowledgeRelations.metadata}, '{last_updated}', '"${new Date().toISOString()}"')`
              }
            });
        }
      }
    } catch (error: any) {
      console.error(`[KnowledgeManager] Error detecting relationships: ${error.message || 'Unknown error'}`);
    }
  }
  
  /**
   * Calculate relationship strength between knowledge entries
   */
  private calculateRelationshipStrength(content1: string, content2: string): number {
    // This is a simplified version - in production would use embedding similarity
    
    // Normalize contents
    const norm1 = content1.toLowerCase();
    const norm2 = content2.toLowerCase();
    
    // Extract key terms
    const terms1 = new Set(norm1.split(/\W+/).filter(w => w.length > 3));
    const terms2 = new Set(norm2.split(/\W+/).filter(w => w.length > 3));
    
    // Find intersection
    const intersection = new Set([...terms1].filter(x => terms2.has(x)));
    
    // Simple Jaccard similarity
    const union = new Set([...terms1, ...terms2]);
    
    return intersection.size / union.size;
  }
  
  /**
   * Detect domains for a piece of content
   */
  private detectDomains(content: string): string[] {
    // Common knowledge domains
    const domainKeywords: Record<string, string[]> = {
      'programming': ['code', 'programming', 'function', 'api', 'algorithm', 'software', 'developer', 'library'],
      'databases': ['database', 'sql', 'nosql', 'query', 'schema', 'db', 'data storage', 'postgres'],
      'ai': ['ai', 'machine learning', 'ml', 'neural network', 'model', 'training', 'inference', 'llm', 'nlp'],
      'security': ['security', 'encryption', 'authentication', 'authorization', 'vulnerability', 'firewall', 'risk'],
      'cloud': ['cloud', 'aws', 'azure', 'gcp', 'serverless', 'container', 'kubernetes', 'docker'],
      'web': ['web', 'frontend', 'backend', 'http', 'javascript', 'css', 'html', 'browser', 'dom'],
      'mobile': ['mobile', 'android', 'ios', 'app', 'react native', 'flutter', 'swift', 'kotlin'],
      'devops': ['devops', 'ci/cd', 'pipeline', 'deployment', 'jenkins', 'github actions', 'ansible', 'terraform'],
      'architecture': ['architecture', 'design pattern', 'microservice', 'monolith', 'distributed', 'scaling'],
      'performance': ['performance', 'optimization', 'latency', 'throughput', 'caching', 'profiling', 'benchmark']
    };
    
    const normalizedContent = content.toLowerCase();
    const detectedDomains: string[] = [];
    
    for (const [domain, keywords] of Object.entries(domainKeywords)) {
      for (const keyword of keywords) {
        if (normalizedContent.includes(keyword)) {
          detectedDomains.push(domain);
          break; // Found one keyword for this domain, move to next domain
        }
      }
    }
    
    return detectedDomains;
  }
  
  /**
   * Extract knowledge nuggets from text
   */
  private extractKnowledgeNuggets(content: string): Array<{
    content: string;
    importance: number;
    confidence: number;
  }> {
    const results: Array<{
      content: string;
      importance: number;
      confidence: number;
    }> = [];
    
    // Split into paragraphs for analysis
    const paragraphs = content.split(/\n\n+/);
    
    for (const paragraph of paragraphs) {
      // Skip short paragraphs
      if (paragraph.length < 50) continue;
      
      // Check if paragraph contains valuable knowledge markers
      const hasKnowledgeMarkers = 
        /important|key|significant|critical|essential|fundamental|understand|remember|note that|consider/i.test(paragraph);
      
      const hasCodeExample = /```[\s\S]*?```/.test(paragraph);
      
      const hasTechnicalContent = 
        /function|method|api|database|algorithm|architecture|design pattern|security|performance|optimization/i.test(paragraph);
      
      const hasStructuredInfo = 
        /\d+\.\s|\*\s|-\s|step \d+|first|second|third|finally|consequently/i.test(paragraph);
      
      // Calculate importance score
      let importance = 0.5; // base score
      
      if (hasKnowledgeMarkers) importance += 0.2;
      if (hasCodeExample) importance += 0.15;
      if (hasTechnicalContent) importance += 0.1;
      if (hasStructuredInfo) importance += 0.05;
      
      // Only include if reasonably important
      if (importance > 0.6) {
        results.push({
          content: paragraph.trim(),
          importance: Math.min(importance, 1.0),
          confidence: 0.7 + (importance - 0.6) // Confidence correlates with importance
        });
      }
    }
    
    // Special case: look for bullet-point lists that might span paragraphs
    const bulletPointLists = content.match(/(?:\n\s*[\*\-\•]\s+.+){3,}/g);
    if (bulletPointLists) {
      for (const list of bulletPointLists) {
        if (list.length > 100) { // Only include substantial lists
          results.push({
            content: list.trim(),
            importance: 0.75,
            confidence: 0.8
          });
        }
      }
    }
    
    return results;
  }
  
  /**
   * Merge two content strings intelligently
   */
  private mergeContents(existing: string, newContent: string): string {
    // If they're very similar, keep the longer one
    if (this.calculateRelationshipStrength(existing, newContent) > 0.8) {
      return existing.length > newContent.length ? existing : newContent;
    }
    
    // Try to combine unique information
    const existingParts = existing.split('\n\n');
    const newParts = newContent.split('\n\n');
    
    // Find parts in newContent that aren't in existing
    const uniqueNewParts = newParts.filter(newPart => {
      return !existingParts.some(existingPart => 
        this.calculateRelationshipStrength(existingPart, newPart) > 0.7
      );
    });
    
    // If there are unique parts, append them
    if (uniqueNewParts.length > 0) {
      return existing + '\n\n' + uniqueNewParts.join('\n\n');
    }
    
    // Otherwise return existing
    return existing;
  }
  
  /**
   * Recursively expand graph node for knowledge graph visualization
   */
  private async expandGraphNode(
    graph: any,
    node: KnowledgeEntry,
    processedNodes: Set<number>,
    depth: number,
    maxNodes: number
  ): Promise<void> {
    // Stop if we've reached limits
    if (processedNodes.size >= maxNodes || processedNodes.has(node.id)) {
      return;
    }
    
    // Add node to graph
    processedNodes.add(node.id);
    graph.nodes.push({
      id: node.id.toString(),
      label: this.truncateForGraph(node.content),
      importance: node.importanceScore,
      type: 'knowledge'
    });
    
    // Stop if we've reached max depth
    if (depth <= 0) {
      return;
    }
    
    // Find relations
    const relations = await db.select()
      .from(aiKnowledgeRelations)
      .where(eq(aiKnowledgeRelations.sourceId, node.id))
      .orderBy(desc(aiKnowledgeRelations.strength))
      .limit(5);
    
    // Process relations
    for (const relation of relations) {
      if (processedNodes.size >= maxNodes) {
        return;
      }
      
      // Add edge
      graph.edges.push({
        source: relation.sourceId.toString(),
        target: relation.targetId.toString(),
        strength: relation.strength,
        type: relation.type
      });
      
      // Get target node if not already processed
      if (!processedNodes.has(relation.targetId)) {
        const [targetNode] = await db.select()
          .from(aiKnowledgeBase)
          .where(eq(aiKnowledgeBase.id, relation.targetId));
        
        if (targetNode) {
          // Recursively expand target node
          await this.expandGraphNode(
            graph,
            targetNode,
            processedNodes,
            depth - 1,
            maxNodes
          );
        }
      }
    }
    
    // Find domains
    const domainEntries = await db.select({ domainId: aiKnowledgeDomains.domainId })
      .from(aiKnowledgeDomains)
      .where(eq(aiKnowledgeDomains.knowledgeId, node.id));
    
    // Add domain nodes and edges
    for (const { domainId } of domainEntries) {
      const domainNodeId = `domain-${domainId}`;
      
      // Add domain node if not already added
      if (!graph.nodes.some((n: any) => n.id === domainNodeId)) {
        const [domain] = await db.select()
          .from(aiKnowledgeDomains)
          .where(eq(aiKnowledgeDomains.id, domainId));
        
        if (domain) {
          graph.nodes.push({
            id: domainNodeId,
            label: domain.name,
            type: 'domain'
          });
        }
      }
      
      // Add edge between knowledge and domain
      graph.edges.push({
        source: node.id.toString(),
        target: domainNodeId,
        type: 'in_domain'
      });
    }
  }
  
  /**
   * Truncate text for graph visualization
   */
  private truncateForGraph(text: string): string {
    const maxLength = 50;
    if (text.length <= maxLength) {
      return text;
    }
    return text.substring(0, maxLength) + '...';
  }
}

// Singleton instance
export const knowledgeManager = KnowledgeManager.getInstance();