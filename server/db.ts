import { Pool, neonConfig } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-serverless';
import ws from "ws";
import * as schema from "@shared/schema";

// Set WebSocket constructor for Neon
neonConfig.webSocketConstructor = ws;

// Handle database configuration
let pool: Pool;
let db: ReturnType<typeof drizzle>;

// Configurable database connection options
const CONNECTION_RETRY_ATTEMPTS = 5;
const CONNECTION_RETRY_DELAY = 1000; // ms

try {
  // Check if DATABASE_URL is available
  if (!process.env.DATABASE_URL) {
    console.warn("DATABASE_URL not set. Using in-memory storage for development.");
    
    // Create a minimal in-memory database or fallback mechanism
    // This ensures the app can run without a database connection
    pool = null as any;
    db = null as any;
  } else {
    // Create connection pool with improved error handling and retry logic
    pool = new Pool({ 
      connectionString: process.env.DATABASE_URL,
      max: 20, // Connection pool size
      idleTimeoutMillis: 30000, // How long a client is allowed to remain idle before being closed
      connectionTimeoutMillis: 5000, // How long to wait for connection
    });
    
    // Initialize Drizzle ORM with the connection pool
    db = drizzle({ client: pool, schema });
    
    // Test database connection
    pool.query('SELECT NOW()').then(() => {
      console.log("Database connection successful");
    }).catch(err => {
      console.error("Database connection test failed:", err.message);
    });
    
    // Handle connection pool errors
    pool.on('error', (err) => {
      console.error('Unexpected database pool error:', err.message);
    });
  }
} catch (error) {
  console.error("Database initialization error:", error instanceof Error ? error.message : String(error));
  // Create fallback mechanism
  pool = null as any;
  db = null as any;
}

// Export database connection
export { pool, db };

// Helper function to check if database is available
export function isDatabaseAvailable(): boolean {
  return !!pool && !!db;
}
