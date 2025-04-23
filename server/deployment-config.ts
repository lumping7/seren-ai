/**
 * Deployment Configuration for Production VDS
 * 
 * This file contains configuration settings for deploying the Seren system
 * to a production Virtual Dedicated Server (VDS) environment.
 */

export const deploymentConfig = {
  // Server configuration
  server: {
    host: process.env.HOST || '0.0.0.0',
    port: process.env.PORT || 3000,
    domain: process.env.DOMAIN || 'luxima.icu',
    useHttps: process.env.USE_HTTPS === 'true' || process.env.NODE_ENV === 'production',
    corsOrigins: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['https://luxima.icu'],
    trustProxy: process.env.TRUST_PROXY === 'true' || process.env.NODE_ENV === 'production',
    sessionSecret: process.env.SESSION_SECRET || 'replace-with-strong-secret-in-production',
    rateLimiting: {
      enabled: process.env.RATE_LIMITING !== 'false',
      windowMs: parseInt(process.env.RATE_LIMITING_WINDOW || '900000', 10), // 15 minutes
      max: parseInt(process.env.RATE_LIMITING_MAX || '100', 10), // 100 requests per window
    }
  },
  
  // Database configuration
  database: {
    url: process.env.DATABASE_URL,
    maxPoolSize: parseInt(process.env.DB_POOL_SIZE || '10', 10),
    ssl: process.env.DB_SSL === 'true' || process.env.NODE_ENV === 'production',
  },
  
  // AI model configuration
  ai: {
    defaultModel: process.env.DEFAULT_MODEL || 'hybrid',
    modelsPath: process.env.MODELS_PATH || './ai_core/models',
    useOfflineMode: process.env.USE_OFFLINE_MODE === 'true' || process.env.NODE_ENV === 'production',
    maxConcurrentRequests: parseInt(process.env.MAX_CONCURRENT_REQUESTS || '10', 10),
    requestTimeout: parseInt(process.env.REQUEST_TIMEOUT || '300000', 10), // 5 minutes
    logging: {
      requests: process.env.LOG_AI_REQUESTS === 'true',
      responses: process.env.LOG_AI_RESPONSES === 'true',
      errors: process.env.LOG_AI_ERRORS !== 'false',
    }
  },
  
  // Security configuration
  security: {
    jwtSecret: process.env.JWT_SECRET || 'replace-with-strong-secret-in-production',
    jwtExpiration: process.env.JWT_EXPIRATION || '1d',
    bcryptRounds: parseInt(process.env.BCRYPT_ROUNDS || '12', 10),
    sessionMaxAge: parseInt(process.env.SESSION_MAX_AGE || '86400000', 10), // 24 hours
    csrfProtection: process.env.CSRF_PROTECTION !== 'false',
    xssProtection: process.env.XSS_PROTECTION !== 'false',
    contentSecurityPolicy: process.env.CONTENT_SECURITY_POLICY !== 'false',
    hsts: process.env.HSTS !== 'false' && process.env.NODE_ENV === 'production',
  },
  
  // Monitoring and logging configuration
  monitoring: {
    logLevel: process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug'),
    enablePerformanceMonitoring: process.env.PERFORMANCE_MONITORING !== 'false',
    enableErrorTracking: process.env.ERROR_TRACKING !== 'false',
    logDirectory: process.env.LOG_DIRECTORY || './logs',
    logRotation: {
      enabled: process.env.LOG_ROTATION !== 'false',
      maxFiles: parseInt(process.env.LOG_MAX_FILES || '7', 10), // 7 days
      maxSize: process.env.LOG_MAX_SIZE || '10m',
    }
  },
  
  // System services configuration
  services: {
    enableOfflineGeneration: true, // Always use offline generation for production
    enableWebSocketServer: true,   // Enable WebSocket server for real-time communication
    enableSystemStatus: true,      // Enable system status API
    enableAdminAPI: true,          // Enable admin API for system management
    adminAPIKey: process.env.ADMIN_API_KEY || 'replace-with-strong-api-key-in-production',
  }
};

/**
 * Get a configuration value with environment variable fallback
 */
export function getConfig(path: string, defaultValue?: any): any {
  const parts = path.split('.');
  let current: any = deploymentConfig;
  
  for (const part of parts) {
    if (current && typeof current === 'object' && part in current) {
      current = current[part];
    } else {
      return defaultValue;
    }
  }
  
  return current !== undefined ? current : defaultValue;
}

/**
 * Validate the deployment configuration
 */
export function validateDeploymentConfig(): { valid: boolean, errors: string[] } {
  const errors: string[] = [];
  
  // Check critical configuration settings
  if (process.env.NODE_ENV === 'production') {
    // Check for secure session secret
    if (!process.env.SESSION_SECRET || process.env.SESSION_SECRET === 'replace-with-strong-secret-in-production') {
      errors.push('Production environment requires a secure SESSION_SECRET');
    }
    
    // Check for secure JWT secret
    if (!process.env.JWT_SECRET || process.env.JWT_SECRET === 'replace-with-strong-secret-in-production') {
      errors.push('Production environment requires a secure JWT_SECRET');
    }
    
    // Check for database URL
    if (!process.env.DATABASE_URL) {
      errors.push('Production environment requires a DATABASE_URL');
    }
    
    // Check for admin API key
    if (!process.env.ADMIN_API_KEY || process.env.ADMIN_API_KEY === 'replace-with-strong-api-key-in-production') {
      errors.push('Production environment requires a secure ADMIN_API_KEY');
    }
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Get the VDS deployment requirements
 */
export function getVdsRequirements(): { memory: string, cpu: string, storage: string } {
  return {
    memory: '4GB RAM minimum, 8GB recommended',
    cpu: '2 vCPUs minimum, 4 vCPUs recommended',
    storage: '20GB SSD minimum, 50GB recommended'
  };
}

/**
 * Get the production deployment checklist
 */
export function getProductionDeploymentChecklist(): string[] {
  return [
    'Set strong SESSION_SECRET environment variable',
    'Set strong JWT_SECRET environment variable',
    'Set strong ADMIN_API_KEY environment variable',
    'Configure DATABASE_URL with production database credentials',
    'Set NODE_ENV to "production"',
    'Configure DOMAIN to "luxima.icu" or your custom domain',
    'Set up SSL certificate for HTTPS',
    'Configure firewall to allow only necessary ports (80, 443)',
    'Set up automatic backups for the database',
    'Configure log rotation and monitoring',
    'Set up a process manager (PM2) for application resilience',
    'Configure a reverse proxy (Nginx) for SSL termination and caching'
  ];
}

/**
 * Create a sample deployment .env file
 */
export function createSampleEnvFile(): string {
  return `# Seren Production Environment Configuration
# Generated on ${new Date().toISOString()}

# Environment
NODE_ENV=production

# Server Configuration
PORT=3000
HOST=0.0.0.0
DOMAIN=luxima.icu
USE_HTTPS=true
TRUST_PROXY=true
SESSION_SECRET=generate-a-secure-random-string-here
CORS_ORIGINS=https://luxima.icu,https://admin.luxima.icu

# Database Configuration
DATABASE_URL=postgres://username:password@localhost:5432/seren
DB_POOL_SIZE=10
DB_SSL=true

# AI Model Configuration
DEFAULT_MODEL=hybrid
USE_OFFLINE_MODE=true
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300000

# Security Configuration
JWT_SECRET=generate-another-secure-random-string-here
JWT_EXPIRATION=1d
BCRYPT_ROUNDS=12
CSRF_PROTECTION=true
XSS_PROTECTION=true
CONTENT_SECURITY_POLICY=true
HSTS=true

# Monitoring and Logging
LOG_LEVEL=info
PERFORMANCE_MONITORING=true
ERROR_TRACKING=true
LOG_DIRECTORY=./logs
LOG_ROTATION=true
LOG_MAX_FILES=7
LOG_MAX_SIZE=10m

# Admin Configuration
ADMIN_API_KEY=generate-a-third-secure-random-string-here
`;
}