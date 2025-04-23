import { users, aiMemories, aiMessages, aiSettings } from "@shared/schema";
import { type User, type InsertUser, type Memory, type Message, type Setting } from "@shared/schema";
import { type z } from "zod";
import { insertMemorySchema, insertMessageSchema, insertSettingSchema } from "@shared/schema";
import connectPg from "connect-pg-simple";
import session from "express-session";
import MemoryStore from "memorystore";
import { db, isDatabaseAvailable } from "./db";
import { eq, desc, and } from "drizzle-orm";
import { pool } from "./db";
import WebSocket from "ws";

// Define types from schema
type InsertMemory = z.infer<typeof insertMemorySchema>;
type InsertMessage = z.infer<typeof insertMessageSchema>;
type InsertSetting = z.infer<typeof insertSettingSchema>;

// In-memory fallback data structures
const memUsers: Map<number, User> = new Map();
const memUsersByUsername: Map<string, User> = new Map();
const memMemories: Memory[] = [];
const memMessages: Map<string, Message[]> = new Map(); // conversationId -> messages
const memSettings: Map<string, Setting> = new Map();
let nextUserId = 1;

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Memory operations
  getMemories(limit?: number): Promise<Memory[]>;
  getMemoriesByUser(userId: number, limit?: number): Promise<Memory[]>;
  createMemory(memory: InsertMemory): Promise<Memory>;
  
  // Message operations
  getMessages(conversationId: string): Promise<Message[]>;
  createMessage(message: InsertMessage): Promise<Message>;
  
  // Settings operations
  getSetting(key: string): Promise<Setting | undefined>;
  updateSetting(key: string, value: any, userId: number): Promise<Setting>;
  
  // Session store
  sessionStore: session.Store;
  
  // Status
  isUsingDatabase(): boolean;
}

// Helper class for error handling
class StorageError extends Error {
  constructor(message: string, public operation: string, public originalError?: any) {
    super(message);
    this.name = 'StorageError';
  }
}

// Resilient storage implementation
export class DatabaseStorage implements IStorage {
  sessionStore: session.Store;
  private dbAvailable: boolean;
  
  constructor() {
    this.dbAvailable = isDatabaseAvailable();
    
    // Initialize session store based on DB availability
    if (this.dbAvailable) {
      try {
        const PostgresSessionStore = connectPg(session);
        this.sessionStore = new PostgresSessionStore({ 
          pool,
          createTableIfMissing: true
        });
        console.log("Using PostgreSQL session store");
      } catch (error) {
        console.warn("Failed to initialize PostgreSQL session store, falling back to memory store:", error);
        const MemoryStoreWithSession = MemoryStore(session);
        this.sessionStore = new MemoryStoreWithSession({
          checkPeriod: 86400000 // prune expired entries every 24h
        });
      }
    } else {
      console.warn("Database not available, using in-memory session store");
      const MemoryStoreWithSession = MemoryStore(session);
      this.sessionStore = new MemoryStoreWithSession({
        checkPeriod: 86400000 // prune expired entries every 24h
      });
    }
  }
  
  // Check if we're using the database
  isUsingDatabase(): boolean {
    return this.dbAvailable;
  }
  
  // Method to initialize settings with a valid user ID
  async initializeSettings(userId: number) {
    try {
      const modelConfigSetting = await this.getSetting("modelConfig");
      if (!modelConfigSetting) {
        await this.updateSetting("modelConfig", {
          defaultModel: "hybrid",
          llama3Enabled: true,
          gemma3Enabled: true,
          temperature: 0.7,
          maxTokens: 1024
        }, userId);
      }
      
      const featureFlagsSetting = await this.getSetting("featureFlags");
      if (!featureFlagsSetting) {
        await this.updateSetting("featureFlags", {
          selfUpgradeEnabled: true,
          codeExecutionEnabled: true,
          autonomousActionsEnabled: false,
          internetAccessEnabled: true
        }, userId);
      }
      
      console.log("Settings initialized successfully");
    } catch (error) {
      console.error("Error initializing settings:", error);
    }
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    try {
      if (this.dbAvailable) {
        const [user] = await db.select().from(users).where(eq(users.id, id));
        return user;
      } else {
        return memUsers.get(id);
      }
    } catch (error) {
      console.error(`Error getting user ${id}:`, error);
      // Fall back to in-memory
      return memUsers.get(id);
    }
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    try {
      if (this.dbAvailable) {
        const [user] = await db.select().from(users).where(eq(users.username, username));
        return user;
      } else {
        return memUsersByUsername.get(username);
      }
    } catch (error) {
      console.error(`Error getting user by username ${username}:`, error);
      // Fall back to in-memory
      return memUsersByUsername.get(username);
    }
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    try {
      if (this.dbAvailable) {
        const [user] = await db.insert(users).values(insertUser).returning();
        return user;
      } else {
        // In-memory user creation
        const id = nextUserId++;
        const now = new Date();
        const user: User = {
          id,
          ...insertUser,
          createdAt: now,
          updatedAt: now,
          lastLoginAt: null,
          isAdmin: insertUser.isAdmin || false
        };
        memUsers.set(id, user);
        memUsersByUsername.set(user.username, user);
        return user;
      }
    } catch (error) {
      console.error(`Error creating user ${insertUser.username}:`, error);
      if (!this.dbAvailable) {
        throw new StorageError('Failed to create user', 'createUser', error);
      }
      
      // Fall back to in-memory if DB fails
      const id = nextUserId++;
      const now = new Date();
      const user: User = {
        id,
        ...insertUser,
        createdAt: now,
        updatedAt: now,
        lastLoginAt: null,
        isAdmin: insertUser.isAdmin || false
      };
      memUsers.set(id, user);
      memUsersByUsername.set(user.username, user);
      return user;
    }
  }
  
  // Memory operations
  async getMemories(limit?: number): Promise<Memory[]> {
    try {
      if (this.dbAvailable) {
        // Create base query
        const baseQuery = db.select().from(aiMemories).orderBy(desc(aiMemories.timestamp));
        
        // Execute with or without limit
        if (limit) {
          return await baseQuery.limit(limit);
        } else {
          return await baseQuery;
        }
      } else {
        // Return in-memory memories, sorted by timestamp descending
        const sorted = [...memMemories].sort((a, b) => 
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
        return limit ? sorted.slice(0, limit) : sorted;
      }
    } catch (error) {
      console.error(`Error getting memories:`, error);
      
      // Fall back to in-memory
      const sorted = [...memMemories].sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      return limit ? sorted.slice(0, limit) : sorted;
    }
  }
  
  async getMemoriesByUser(userId: number, limit?: number): Promise<Memory[]> {
    try {
      if (this.dbAvailable) {
        // Create base query with user filter
        const baseQuery = db.select()
          .from(aiMemories)
          .where(eq(aiMemories.userId, userId))
          .orderBy(desc(aiMemories.timestamp));
        
        // Execute with or without limit
        if (limit) {
          return await baseQuery.limit(limit);
        } else {
          return await baseQuery;
        }
      } else {
        // Filter in-memory memories by userId, sorted by timestamp descending
        const filtered = memMemories
          .filter(m => m.userId === userId)
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        
        return limit ? filtered.slice(0, limit) : filtered;
      }
    } catch (error) {
      console.error(`Error getting memories for user ${userId}:`, error);
      
      // Fall back to in-memory
      const filtered = memMemories
        .filter(m => m.userId === userId)
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      
      return limit ? filtered.slice(0, limit) : filtered;
    }
  }
  
  async createMemory(memory: InsertMemory): Promise<Memory> {
    try {
      if (this.dbAvailable) {
        const [newMemory] = await db.insert(aiMemories).values(memory).returning();
        return newMemory;
      } else {
        // Create in-memory memory
        const id = Date.now();
        const timestamp = new Date();
        const newMemory: Memory = {
          id,
          ...memory,
          timestamp
        };
        memMemories.push(newMemory);
        return newMemory;
      }
    } catch (error) {
      console.error(`Error creating memory:`, error);
      
      // Fall back to in-memory
      const id = Date.now();
      const timestamp = new Date();
      const newMemory: Memory = {
        id,
        ...memory,
        timestamp
      };
      memMemories.push(newMemory);
      return newMemory;
    }
  }
  
  // Message operations
  async getMessages(conversationId: string): Promise<Message[]> {
    try {
      if (this.dbAvailable) {
        return await db.select()
          .from(aiMessages)
          .where(eq(aiMessages.conversationId, conversationId))
          .orderBy(aiMessages.timestamp);
      } else {
        // Get in-memory messages for conversation
        return memMessages.get(conversationId) || [];
      }
    } catch (error) {
      console.error(`Error getting messages for conversation ${conversationId}:`, error);
      
      // Fall back to in-memory
      return memMessages.get(conversationId) || [];
    }
  }
  
  async createMessage(message: InsertMessage): Promise<Message> {
    try {
      if (this.dbAvailable) {
        const [newMessage] = await db.insert(aiMessages).values(message).returning();
        return newMessage;
      } else {
        // Create in-memory message
        const id = Date.now();
        const timestamp = new Date();
        const newMessage: Message = {
          id,
          ...message,
          timestamp
        };
        
        // Add to messages map
        const conversationMessages = memMessages.get(message.conversationId) || [];
        conversationMessages.push(newMessage);
        memMessages.set(message.conversationId, conversationMessages);
        
        return newMessage;
      }
    } catch (error) {
      console.error(`Error creating message:`, error);
      
      // Fall back to in-memory
      const id = Date.now();
      const timestamp = new Date();
      const newMessage: Message = {
        id,
        ...message,
        timestamp
      };
      
      // Add to messages map
      const conversationMessages = memMessages.get(message.conversationId) || [];
      conversationMessages.push(newMessage);
      memMessages.set(message.conversationId, conversationMessages);
      
      return newMessage;
    }
  }
  
  // Settings operations
  async getSetting(key: string): Promise<Setting | undefined> {
    try {
      if (this.dbAvailable) {
        const [setting] = await db.select()
          .from(aiSettings)
          .where(eq(aiSettings.settingKey, key));
        
        return setting;
      } else {
        return memSettings.get(key);
      }
    } catch (error) {
      console.error(`Error getting setting ${key}:`, error);
      
      // Fall back to in-memory
      return memSettings.get(key);
    }
  }
  
  async updateSetting(key: string, value: any, userId: number): Promise<Setting> {
    try {
      if (this.dbAvailable) {
        // Check if setting exists
        const existingSetting = await this.getSetting(key);
        
        if (existingSetting) {
          // Update existing setting
          const [updatedSetting] = await db.update(aiSettings)
            .set({ 
              settingValue: value,
              updatedAt: new Date(),
              updatedBy: userId
            })
            .where(eq(aiSettings.settingKey, key))
            .returning();
          
          return updatedSetting;
        } else {
          // Create new setting
          const [newSetting] = await db.insert(aiSettings)
            .values({
              settingKey: key,
              settingValue: value,
              updatedBy: userId
            })
            .returning();
          
          return newSetting;
        }
      } else {
        // Check if setting exists in memory
        const existingSetting = memSettings.get(key);
        const now = new Date();
        
        if (existingSetting) {
          // Update existing setting
          const updatedSetting: Setting = {
            ...existingSetting,
            settingValue: value,
            updatedAt: now,
            updatedBy: userId
          };
          
          memSettings.set(key, updatedSetting);
          return updatedSetting;
        } else {
          // Create new setting
          const newSetting: Setting = {
            id: Date.now(),
            settingKey: key,
            settingValue: value,
            updatedAt: now,
            updatedBy: userId
          };
          
          memSettings.set(key, newSetting);
          return newSetting;
        }
      }
    } catch (error) {
      console.error(`Error updating setting ${key}:`, error);
      
      // Fall back to in-memory
      const existingSetting = memSettings.get(key);
      const now = new Date();
      
      if (existingSetting) {
        // Update existing setting
        const updatedSetting: Setting = {
          ...existingSetting,
          settingValue: value,
          updatedAt: now,
          updatedBy: userId
        };
        
        memSettings.set(key, updatedSetting);
        return updatedSetting;
      } else {
        // Create new setting
        const newSetting: Setting = {
          id: Date.now(),
          settingKey: key,
          settingValue: value,
          updatedAt: now,
          updatedBy: userId
        };
        
        memSettings.set(key, newSetting);
        return newSetting;
      }
    }
  }
}

// Export storage instance
export const storage = new DatabaseStorage();
