/**
 * Health Monitoring System
 * 
 * Provides endpoints and utilities for monitoring the health and status
 * of the Seren AI system in production environments.
 */

import { Router, Request, Response } from 'express';
import { isDatabaseAvailable } from './db';
import { modelServices } from './ai';

// Create router instance
export const healthRouter = Router();

// System Version
const VERSION = '1.0.0';

// Overall system status API endpoint
healthRouter.get('/', (req: Request, res: Response) => {
  const qwenStatus = modelServices.get('qwen2.5-7b-omni')?.ready || false;
  const olympicStatus = modelServices.get('olympiccoder-7b')?.ready || false;

  // Compile comprehensive system health data
  const healthData = {
    status: "operational",
    version: VERSION,
    environment: process.env.NODE_ENV || "development",
    timestamp: new Date().toISOString(),
    components: {
      models: {
        qwen: qwenStatus ? "ready" : "unavailable",
        olympic: olympicStatus ? "ready" : "unavailable",
        hybrid: (qwenStatus && olympicStatus) ? "ready" : "partial"
      },
      database: isDatabaseAvailable() ? "connected" : "fallback",
      webServer: "operational"
    },
    systemResources: {
      memory: {
        rss: Math.round(process.memoryUsage().rss / 1024 / 1024),
        heapTotal: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
        heapUsed: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        external: Math.round(process.memoryUsage().external / 1024 / 1024)
      },
      uptime: Math.round(process.uptime()),
      cpu: process.cpuUsage()
    }
  };

  res.json(healthData);
});

// Detailed AI system health endpoint
healthRouter.get('/ai', (req: Request, res: Response) => {
  const modelStatuses = Array.from(modelServices.entries()).map(([name, service]) => {
    return {
      name,
      status: service.ready ? "ready" : "unavailable",
      lastHeartbeat: service.lastHeartbeat,
      heartbeatInterval: service.heartbeatInterval
    };
  });

  res.json({
    aiSystem: "operational",
    models: modelStatuses,
    timestamp: new Date().toISOString()
  });
});

// Database health endpoint
healthRouter.get('/database', (req: Request, res: Response) => {
  const dbStatus = isDatabaseAvailable();
  
  res.json({
    database: dbStatus ? "connected" : "fallback",
    mode: dbStatus ? "persistent" : "in-memory",
    timestamp: new Date().toISOString()
  });
});

// Memory usage endpoint
healthRouter.get('/memory', (req: Request, res: Response) => {
  const memoryUsage = process.memoryUsage();
  
  res.json({
    rss: Math.round(memoryUsage.rss / 1024 / 1024),
    heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024),
    heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024),
    external: Math.round(memoryUsage.external / 1024 / 1024),
    arrayBuffers: Math.round(memoryUsage.arrayBuffers / 1024 / 1024),
    timestamp: new Date().toISOString()
  });
});