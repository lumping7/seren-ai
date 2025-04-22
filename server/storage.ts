import { users, aiMemories, aiMessages, aiSettings } from "@shared/schema";
import { type User, type InsertUser, type Memory, type Message, type Setting } from "@shared/schema";
import { type z } from "zod";
import { insertMemorySchema, insertMessageSchema, insertSettingSchema } from "@shared/schema";
import connectPg from "connect-pg-simple";
import session from "express-session";
import { db } from "./db";
import { eq, desc, and } from "drizzle-orm";
import { pool } from "./db";
import WebSocket from "ws";

// Define types from schema
type InsertMemory = z.infer<typeof insertMemorySchema>;
type InsertMessage = z.infer<typeof insertMessageSchema>;
type InsertSetting = z.infer<typeof insertSettingSchema>;

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
  sessionStore: session.SessionStore;
}

export class DatabaseStorage implements IStorage {
  sessionStore: session.SessionStore;

  constructor() {
    const PostgresSessionStore = connectPg(session);
    this.sessionStore = new PostgresSessionStore({ 
      pool,
      createTableIfMissing: true
    });
    
    // We'll initialize settings after we confirm a user exists in auth.ts
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

  async getUser(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values(insertUser).returning();
    return user;
  }
  
  async getMemories(limit?: number): Promise<Memory[]> {
    let query = db.select().from(aiMemories).orderBy(desc(aiMemories.timestamp));
    
    if (limit) {
      query = query.limit(limit);
    }
    
    return await query;
  }
  
  async getMemoriesByUser(userId: number, limit?: number): Promise<Memory[]> {
    let query = db.select()
      .from(aiMemories)
      .where(eq(aiMemories.userId, userId))
      .orderBy(desc(aiMemories.timestamp));
    
    if (limit) {
      query = query.limit(limit);
    }
    
    return await query;
  }
  
  async createMemory(memory: InsertMemory): Promise<Memory> {
    const [newMemory] = await db.insert(aiMemories).values(memory).returning();
    return newMemory;
  }
  
  async getMessages(conversationId: string): Promise<Message[]> {
    return await db.select()
      .from(aiMessages)
      .where(eq(aiMessages.conversationId, conversationId))
      .orderBy(aiMessages.timestamp);
  }
  
  async createMessage(message: InsertMessage): Promise<Message> {
    const [newMessage] = await db.insert(aiMessages).values(message).returning();
    return newMessage;
  }
  
  async getSetting(key: string): Promise<Setting | undefined> {
    const [setting] = await db.select()
      .from(aiSettings)
      .where(eq(aiSettings.settingKey, key));
    
    return setting;
  }
  
  async updateSetting(key: string, value: any, userId: number): Promise<Setting> {
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
  }
}

export const storage = new DatabaseStorage();
