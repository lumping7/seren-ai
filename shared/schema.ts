import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  isAdmin: boolean("is_admin").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const aiMemories = pgTable("ai_memories", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  content: text("content").notNull(),
  type: text("type").notNull(), // chat, execution, system, etc.
  userId: integer("user_id").references(() => users.id),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
  metadata: jsonb("metadata"),
});

export const aiMessages = pgTable("ai_messages", {
  id: serial("id").primaryKey(),
  conversationId: text("conversation_id").notNull(),
  role: text("role").notNull(), // user, assistant, system
  content: text("content").notNull(),
  model: text("model"), // llama3, gemma3, hybrid
  timestamp: timestamp("timestamp").defaultNow().notNull(),
  userId: integer("user_id").references(() => users.id),
  metadata: jsonb("metadata"),
});

export const aiSettings = pgTable("ai_settings", {
  id: serial("id").primaryKey(),
  settingKey: text("setting_key").notNull().unique(),
  settingValue: jsonb("setting_value").notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
  updatedBy: integer("updated_by").references(() => users.id),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  isAdmin: true,
});

export const insertMemorySchema = createInsertSchema(aiMemories).pick({
  title: true,
  content: true,
  type: true,
  userId: true,
  metadata: true,
});

export const insertMessageSchema = createInsertSchema(aiMessages).pick({
  conversationId: true,
  role: true,
  content: true,
  model: true,
  userId: true,
  metadata: true,
});

export const insertSettingSchema = createInsertSchema(aiSettings).pick({
  settingKey: true,
  settingValue: true,
  updatedBy: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type Memory = typeof aiMemories.$inferSelect;
export type Message = typeof aiMessages.$inferSelect;
export type Setting = typeof aiSettings.$inferSelect;
