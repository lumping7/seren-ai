import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { errorHandler } from "./ai/error-handler";
import { performanceMonitor } from "./ai/performance-monitor";
import compression from "compression";
import helmet from "helmet";
import hpp from "hpp";
import rateLimit from "express-rate-limit";
import { v4 as uuidv4 } from "uuid";
import cluster from "cluster";
import os from "os";

// Only use cluster in production for better performance
const ENABLE_CLUSTERING = process.env.NODE_ENV === "production" && process.env.DISABLE_CLUSTERING !== "true";
const numCPUs = ENABLE_CLUSTERING ? os.cpus().length : 1;

// If clustering is enabled and this is the primary process
if (ENABLE_CLUSTERING && cluster.isPrimary) {
  console.log(`Primary ${process.pid} is running`);
  console.log(`Starting ${numCPUs} workers...`);

  // Fork workers
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  // Handle worker exits and restart them
  cluster.on('exit', (worker, code, signal) => {
    console.log(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}`);
    console.log('Starting a new worker...');
    cluster.fork();
  });
} else {
  // This is a worker process or clustering is disabled
  const app = express();

  // Security middleware (production only)
  if (process.env.NODE_ENV === "production") {
    app.use(helmet()); // Set security headers
    app.use(hpp()); // Protect against HTTP Parameter Pollution attacks
  }

  // Apply rate limiting
  const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    standardHeaders: true,
    legacyHeaders: false,
    message: {
      error: "Too many requests, please try again later.",
      retryAfter: 15 * 60, // 15 minutes
    },
    skip: (req: express.Request) => !req.path.startsWith("/api") // Only rate limit API requests
  });
  app.use("/api/", apiLimiter);

  // Compression middleware
  app.use(compression());

  // Body parsing middleware with size limits
  app.use(express.json({ limit: '2mb' }));
  app.use(express.urlencoded({ extended: false, limit: '2mb' }));

  // Request tracking and logging middleware
  app.use((req, res, next) => {
    // Add request ID for tracing
    const requestId = req.headers['x-request-id'] as string || uuidv4();
    req.headers['x-request-id'] = requestId;
    res.setHeader('x-request-id', requestId);
    
    // Performance monitoring
    performanceMonitor.startOperation('api_request', requestId);
    
    const start = Date.now();
    const path = req.path;
    let capturedJsonResponse: Record<string, any> | undefined = undefined;

    // Capture JSON responses for logging
    const originalResJson = res.json;
    res.json = function (bodyJson, ...args) {
      capturedJsonResponse = bodyJson;
      return originalResJson.apply(res, [bodyJson, ...args]);
    };

    // Log completed requests
    res.on("finish", () => {
      const duration = Date.now() - start;
      if (path.startsWith("/api")) {
        let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
        if (capturedJsonResponse) {
          logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
        }

        // Truncate very long log lines
        if (logLine.length > 500) {
          logLine = logLine.slice(0, 499) + "…";
        }

        log(logLine);
        
        // End performance monitoring
        performanceMonitor.endOperation(requestId, res.statusCode >= 400);
      }
    });

    next();
  });

  // Use our sophisticated error handler middleware
  app.use(errorHandler.errorMiddleware());

  (async () => {
    try {
      const server = await registerRoutes(app);

      // Final error handling middleware
      app.use((err: any, req: Request, res: Response, next: NextFunction) => {
        // Use our error handler for a proper structured error response
        errorHandler.handleError(err, req, res);
      });

      // Set up unhandled rejection handlers
      process.on('unhandledRejection', (reason: any, promise: Promise<any>) => {
        console.error('Unhandled Rejection at:', promise, 'reason:', reason);
        // Create a formatted error for the error handler
        const error = reason instanceof Error ? reason : new Error(String(reason));
        errorHandler.handleError(error);
      });

      // Set up uncaught exception handler
      process.on('uncaughtException', (error: Error) => {
        console.error('Uncaught Exception:', error);
        errorHandler.handleError(error);
        
        // Give the error handler some time to process the error before exiting
        setTimeout(() => {
          console.error('Shutting down due to uncaught exception');
          process.exit(1);
        }, 1000);
      });

      // importantly only setup vite in development and after
      // setting up all the other routes so the catch-all route
      // doesn't interfere with the other routes
      if (app.get("env") === "development") {
        await setupVite(app, server);
      } else {
        serveStatic(app);
      }

      // ALWAYS serve the app on port 5000
      // this serves both the API and the client.
      // It is the only port that is not firewalled.
      const port = 5000;
      server.listen({
        port,
        host: "0.0.0.0",
        reusePort: true,
      }, () => {
        log(`serving on port ${port}`);
        
        // Log system information
        const memoryUsage = process.memoryUsage();
        const resourceInfo = {
          pid: process.pid,
          memoryRSS: `${Math.round(memoryUsage.rss / 1024 / 1024)} MB`,
          memoryHeapTotal: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)} MB`,
          memoryHeapUsed: `${Math.round(memoryUsage.heapUsed / 1024 / 1024)} MB`,
          cpuCount: os.cpus().length,
          nodeVersion: process.version,
          environment: process.env.NODE_ENV || 'development'
        };
        
        console.log(`[Server] Started with configuration:`, resourceInfo);
      });
      
      // Graceful shutdown
      const gracefulShutdown = (signal: string) => {
        console.log(`[Server] ${signal} received. Shutting down gracefully...`);
        server.close(() => {
          console.log('[Server] HTTP server closed');
          
          // Close database connections or any other resources here
          
          process.exit(0);
        });
        
        // Force shutdown after 30 seconds if graceful shutdown fails
        setTimeout(() => {
          console.error('[Server] Shutdown timed out, forcing exit...');
          process.exit(1);
        }, 30000);
      };
      
      // Listen for termination signals
      process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
      process.on('SIGINT', () => gracefulShutdown('SIGINT'));
      
    } catch (error) {
      console.error('[Server] Failed to start server:', error);
      process.exit(1);
    }
  })();
}
