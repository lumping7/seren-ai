import { pgTable, serial, varchar, timestamp, text, integer, boolean, jsonb, primaryKey, real, unique } from 'drizzle-orm/pg-core';
import { createInsertSchema } from 'drizzle-zod';
import { z } from 'zod';
type Json = any;

// Users table
export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  username: varchar('username', { length: 255 }).notNull().unique(),
  password: varchar('password', { length: 255 }).notNull(),
  email: varchar('email', { length: 255 }),
  displayName: varchar('display_name', { length: 255 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  lastLoginAt: timestamp('last_login_at').default(null),
  isAdmin: boolean('is_admin').default(false),
  preferences: jsonb('preferences').default({})
});

// AI Memories table for long-term storage
export const aiMemories = pgTable('ai_memories', {
  id: serial('id').primaryKey(),
  userId: integer('user_id').references(() => users.id),
  type: varchar('type', { length: 50 }).notNull().default('general'),
  content: text('content').notNull(),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  metadata: jsonb('metadata').default({})
});

// AI Messages table for conversation history
export const aiMessages = pgTable('ai_messages', {
  id: serial('id').primaryKey(),
  userId: integer('user_id').references(() => users.id),
  conversationId: varchar('conversation_id', { length: 255 }).notNull(),
  role: varchar('role', { length: 50 }).notNull(),
  content: text('content').notNull(),
  model: varchar('model', { length: 50 }),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  metadata: jsonb('metadata').default({})
});

// AI Settings table
export const aiSettings = pgTable('ai_settings', {
  id: serial('id').primaryKey(),
  settingKey: varchar('setting_key', { length: 255 }).notNull().unique(),
  settingValue: jsonb('setting_value').notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  updatedBy: integer('updated_by').references(() => users.id)
});

// Knowledge base table
export const aiKnowledgeBase = pgTable('ai_knowledge_base', {
  id: serial('id').primaryKey(),
  content: text('content').notNull(),
  source: varchar('source', { length: 50 }).notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  createdBy: integer('created_by').references(() => users.id),
  lastAccessedAt: timestamp('last_accessed_at').defaultNow().notNull(),
  accessCount: integer('access_count').notNull().default(0),
  importanceScore: real('importance_score').notNull().default(0.5),
  archived: boolean('archived').notNull().default(false),
  metadata: jsonb('metadata').default({})
});

// Knowledge domains table
export const aiKnowledgeDomains = pgTable('ai_knowledge_domains', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 255 }).notNull(),
  description: text('description'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  createdBy: integer('created_by').references(() => users.id),
  metadata: jsonb('metadata').default({})
});

// Knowledge-domain mapping table
export const aiKnowledgeToDomains = pgTable('ai_knowledge_to_domains', {
  knowledgeId: integer('knowledge_id').notNull().references(() => aiKnowledgeBase.id, { onDelete: 'cascade' }),
  domainId: integer('domain_id').notNull().references(() => aiKnowledgeDomains.id, { onDelete: 'cascade' }),
}, (t) => ({
  pk: primaryKey({ columns: [t.knowledgeId, t.domainId] })
}));

// Knowledge relationships table
export const aiKnowledgeRelations = pgTable('ai_knowledge_relations', {
  id: serial('id').primaryKey(),
  sourceId: integer('source_id').notNull().references(() => aiKnowledgeBase.id, { onDelete: 'cascade' }),
  targetId: integer('target_id').notNull().references(() => aiKnowledgeBase.id, { onDelete: 'cascade' }),
  type: varchar('type', { length: 50 }).notNull(),
  strength: real('strength').notNull().default(0.5),
  metadata: jsonb('metadata').default({})
}, (t) => ({
  uniqueRelation: unique().on(t.sourceId, t.targetId)
}));

// Knowledge tags table
export const aiKnowledgeTags = pgTable('ai_knowledge_tags', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 100 }).notNull().unique(),
  description: text('description'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  metadata: jsonb('metadata').default({})
});

// Knowledge-tag mapping table
export const aiKnowledgeToTags = pgTable('ai_knowledge_to_tags', {
  knowledgeId: integer('knowledge_id').notNull().references(() => aiKnowledgeBase.id, { onDelete: 'cascade' }),
  tagId: integer('tag_id').notNull().references(() => aiKnowledgeTags.id, { onDelete: 'cascade' }),
}, (t) => ({
  pk: primaryKey({ columns: [t.knowledgeId, t.tagId] })
}));

// User feedback on knowledge entries
export const aiKnowledgeFeedback = pgTable('ai_knowledge_feedback', {
  id: serial('id').primaryKey(),
  knowledgeId: integer('knowledge_id').notNull().references(() => aiKnowledgeBase.id, { onDelete: 'cascade' }),
  userId: integer('user_id').references(() => users.id),
  rating: integer('rating').notNull(), // 1-5 rating scale
  comment: text('comment'),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  metadata: jsonb('metadata').default({})
});

// Define source types for knowledge
export type KnowledgeSource = 'user' | 'conversation' | 'document' | 'api' | 'reasoning';

// Zod schemas for validation
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  email: true,
  displayName: true,
  isAdmin: true,
  preferences: true
});

export const insertMemorySchema = createInsertSchema(aiMemories).pick({
  userId: true,
  type: true,
  content: true,
  metadata: true
});

export const insertMessageSchema = createInsertSchema(aiMessages).pick({
  userId: true,
  conversationId: true,
  role: true,
  content: true,
  model: true,
  metadata: true
});

export const insertSettingSchema = createInsertSchema(aiSettings).pick({
  settingKey: true,
  settingValue: true,
  updatedBy: true
});

export const insertKnowledgeSchema = createInsertSchema(aiKnowledgeBase).pick({
  content: true,
  source: true,
  createdBy: true,
  importanceScore: true,
  metadata: true
});

export const insertKnowledgeDomainSchema = createInsertSchema(aiKnowledgeDomains).pick({
  name: true,
  description: true,
  createdBy: true,
  metadata: true
});

export const insertKnowledgeRelationSchema = createInsertSchema(aiKnowledgeRelations).pick({
  sourceId: true,
  targetId: true,
  type: true,
  strength: true,
  metadata: true
});

// Type exports
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type Memory = typeof aiMemories.$inferSelect;
export type Message = typeof aiMessages.$inferSelect;
export type Setting = typeof aiSettings.$inferSelect;
export type KnowledgeEntry = typeof aiKnowledgeBase.$inferSelect;
export type KnowledgeDomain = typeof aiKnowledgeDomains.$inferSelect;
export type KnowledgeRelation = typeof aiKnowledgeRelations.$inferSelect;