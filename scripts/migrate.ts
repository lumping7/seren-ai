import { drizzle } from 'drizzle-orm/neon-serverless';
import { migrate } from 'drizzle-orm/neon-serverless/migrator';
import { Pool, neonConfig } from '@neondatabase/serverless';
import ws from 'ws';
import * as schema from '../shared/schema';

// WebSocket is required for Neon serverless
neonConfig.webSocketConstructor = ws;

const runMigration = async () => {
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL environment variable is not set');
  }

  console.log('Starting database migration...');
  
  // Create connection pool
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });
  const db = drizzle(pool, { schema });
  
  try {
    // Drop all tables if they exist
    console.log('Dropping existing tables...');
    
    // Drop in reverse order of creation to handle foreign key constraints
    const tables = [
      'ai_knowledge_to_tags',
      'ai_knowledge_feedback',
      'ai_knowledge_to_domains',
      'ai_knowledge_relations',
      'ai_knowledge_tags',
      'ai_knowledge_domains',
      'ai_knowledge_base',
      'ai_settings',
      'ai_messages',
      'ai_memories',
      'users'
    ];
    
    for (const table of tables) {
      try {
        await pool.query(`DROP TABLE IF EXISTS ${table} CASCADE`);
        console.log(`Dropped table ${table}`);
      } catch (dropError) {
        console.error(`Error dropping table ${table}:`, dropError);
      }
    }
    
    // Create tables based on schema
    console.log('Creating tables from schema...');
    
    // Users table
    await pool.query(`
      CREATE TABLE "users" (
        "id" SERIAL PRIMARY KEY,
        "username" VARCHAR(255) NOT NULL UNIQUE,
        "password" VARCHAR(255) NOT NULL,
        "email" VARCHAR(255),
        "display_name" VARCHAR(255),
        "created_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "updated_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "last_login_at" TIMESTAMP,
        "preferences" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created users table');
    
    // AI Memories table
    await pool.query(`
      CREATE TABLE "ai_memories" (
        "id" SERIAL PRIMARY KEY,
        "user_id" INTEGER REFERENCES "users"("id"),
        "type" VARCHAR(50) NOT NULL DEFAULT 'general',
        "content" TEXT NOT NULL,
        "timestamp" TIMESTAMP DEFAULT NOW() NOT NULL,
        "metadata" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created ai_memories table');
    
    // AI Messages table
    await pool.query(`
      CREATE TABLE "ai_messages" (
        "id" SERIAL PRIMARY KEY,
        "user_id" INTEGER REFERENCES "users"("id"),
        "conversation_id" VARCHAR(255) NOT NULL,
        "role" VARCHAR(50) NOT NULL,
        "content" TEXT NOT NULL,
        "model" VARCHAR(50),
        "timestamp" TIMESTAMP DEFAULT NOW() NOT NULL,
        "metadata" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created ai_messages table');
    
    // AI Settings table
    await pool.query(`
      CREATE TABLE "ai_settings" (
        "id" SERIAL PRIMARY KEY,
        "setting_key" VARCHAR(255) NOT NULL UNIQUE,
        "setting_value" JSONB NOT NULL,
        "updated_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "updated_by" INTEGER REFERENCES "users"("id")
      )
    `);
    console.log('Created ai_settings table');
    
    // AI Knowledge Base table
    await pool.query(`
      CREATE TABLE "ai_knowledge_base" (
        "id" SERIAL PRIMARY KEY,
        "content" TEXT NOT NULL,
        "source" VARCHAR(50) NOT NULL,
        "created_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "created_by" INTEGER REFERENCES "users"("id"),
        "last_accessed_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "access_count" INTEGER NOT NULL DEFAULT 0,
        "importance_score" REAL NOT NULL DEFAULT 0.5,
        "archived" BOOLEAN NOT NULL DEFAULT FALSE,
        "metadata" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created ai_knowledge_base table');
    
    // AI Knowledge Domains table
    await pool.query(`
      CREATE TABLE "ai_knowledge_domains" (
        "id" SERIAL PRIMARY KEY,
        "name" VARCHAR(255) NOT NULL,
        "description" TEXT,
        "created_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "created_by" INTEGER REFERENCES "users"("id"),
        "metadata" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created ai_knowledge_domains table');
    
    // AI Knowledge Relations table
    await pool.query(`
      CREATE TABLE "ai_knowledge_relations" (
        "id" SERIAL PRIMARY KEY,
        "source_id" INTEGER NOT NULL REFERENCES "ai_knowledge_base"("id") ON DELETE CASCADE,
        "target_id" INTEGER NOT NULL REFERENCES "ai_knowledge_base"("id") ON DELETE CASCADE,
        "type" VARCHAR(50) NOT NULL,
        "strength" REAL NOT NULL DEFAULT 0.5,
        "metadata" JSONB DEFAULT '{}',
        UNIQUE("source_id", "target_id")
      )
    `);
    console.log('Created ai_knowledge_relations table');
    
    // AI Knowledge-Domain Mapping table
    await pool.query(`
      CREATE TABLE "ai_knowledge_to_domains" (
        "knowledge_id" INTEGER NOT NULL REFERENCES "ai_knowledge_base"("id") ON DELETE CASCADE,
        "domain_id" INTEGER NOT NULL REFERENCES "ai_knowledge_domains"("id") ON DELETE CASCADE,
        PRIMARY KEY ("knowledge_id", "domain_id")
      )
    `);
    console.log('Created ai_knowledge_to_domains table');
    
    // AI Knowledge Tags table
    await pool.query(`
      CREATE TABLE "ai_knowledge_tags" (
        "id" SERIAL PRIMARY KEY,
        "name" VARCHAR(100) NOT NULL UNIQUE,
        "description" TEXT,
        "created_at" TIMESTAMP DEFAULT NOW() NOT NULL,
        "metadata" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created ai_knowledge_tags table');
    
    // AI Knowledge-Tag Mapping table
    await pool.query(`
      CREATE TABLE "ai_knowledge_to_tags" (
        "knowledge_id" INTEGER NOT NULL REFERENCES "ai_knowledge_base"("id") ON DELETE CASCADE,
        "tag_id" INTEGER NOT NULL REFERENCES "ai_knowledge_tags"("id") ON DELETE CASCADE,
        PRIMARY KEY ("knowledge_id", "tag_id")
      )
    `);
    console.log('Created ai_knowledge_to_tags table');
    
    // AI Knowledge Feedback table
    await pool.query(`
      CREATE TABLE "ai_knowledge_feedback" (
        "id" SERIAL PRIMARY KEY,
        "knowledge_id" INTEGER NOT NULL REFERENCES "ai_knowledge_base"("id") ON DELETE CASCADE,
        "user_id" INTEGER REFERENCES "users"("id"),
        "rating" INTEGER NOT NULL,
        "comment" TEXT,
        "timestamp" TIMESTAMP DEFAULT NOW() NOT NULL,
        "metadata" JSONB DEFAULT '{}'
      )
    `);
    console.log('Created ai_knowledge_feedback table');
    
    // Add indexes
    console.log('Adding indexes...');
    
    // Knowledge base indexes
    await pool.query(`
      CREATE INDEX knowledge_content_idx ON ai_knowledge_base USING gin(to_tsvector('english', content));
      CREATE INDEX knowledge_importance_idx ON ai_knowledge_base(importance_score);
      CREATE INDEX knowledge_source_idx ON ai_knowledge_base(source);
      CREATE INDEX knowledge_accessed_idx ON ai_knowledge_base(last_accessed_at);
    `);
    
    // Add extension for similarity search
    await pool.query(`
      CREATE EXTENSION IF NOT EXISTS pg_trgm;
    `);
    
    console.log('Migration completed successfully');
  } catch (error) {
    console.error('Migration failed:', error);
    throw error;
  } finally {
    await pool.end();
  }
};

runMigration().catch(console.error);