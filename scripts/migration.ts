import { Pool } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-serverless';
import { migrate } from 'drizzle-orm/neon-serverless/migrator';
import ws from "ws";

// Configure WebSocket for Neon
import { neonConfig } from '@neondatabase/serverless';
neonConfig.webSocketConstructor = ws;

// Configuration
const migrationTable = 'drizzle_migrations';
const migrationsFolder = './drizzle';

// Check for database URL
if (!process.env.DATABASE_URL) {
  console.error("DATABASE_URL must be set to run migrations");
  process.exit(1);
}

// Create connection pool
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

/**
 * Apply schema changes for fields added after initial schema push
 */
async function addMissingColumns() {
  console.log("Checking and adding missing columns...");
  
  try {
    // Check if the is_admin column exists in the users table
    const checkIsAdminColumn = await pool.query(`
      SELECT column_name 
      FROM information_schema.columns 
      WHERE table_name = 'users' AND column_name = 'is_admin'
    `);
    
    // Add the is_admin column if it doesn't exist
    if (checkIsAdminColumn.rowCount === 0) {
      console.log("Adding 'is_admin' column to users table...");
      await pool.query(`
        ALTER TABLE users 
        ADD COLUMN is_admin BOOLEAN DEFAULT false
      `);
      console.log("Added 'is_admin' column to users table");
    } else {
      console.log("'is_admin' column already exists in users table");
    }
    
    // Check if the lastLoginAt column exists in the users table
    const checkLastLoginAtColumn = await pool.query(`
      SELECT column_name 
      FROM information_schema.columns 
      WHERE table_name = 'users' AND column_name = 'last_login_at'
    `);
    
    // Add the last_login_at column if it doesn't exist
    if (checkLastLoginAtColumn.rowCount === 0) {
      console.log("Adding 'last_login_at' column to users table...");
      await pool.query(`
        ALTER TABLE users 
        ADD COLUMN last_login_at TIMESTAMP
      `);
      console.log("Added 'last_login_at' column to users table");
    } else {
      console.log("'last_login_at' column already exists in users table");
    }
    
    console.log("Schema update completed successfully");
  } catch (error) {
    console.error("Error updating schema:", error);
    throw error;
  }
}

/**
 * Main migration function
 */
async function main() {
  try {
    console.log("Starting database migration...");
    
    // Add any missing columns needed for existing tables
    await addMissingColumns();
    
    console.log("Migration completed successfully");
    process.exit(0);
  } catch (error) {
    console.error("Migration failed:", error);
    process.exit(1);
  } finally {
    // Close the pool
    await pool.end();
  }
}

// Run the migration
main();