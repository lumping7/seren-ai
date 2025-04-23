# Seren AI Production Deployment Guide

## Overview

Seren is an autonomous AI development platform specializing in fully offline, self-contained intelligent system creation. It features a multi-model architecture with Qwen2.5-7b-omni and OlympicCoder-7B, a neuro-symbolic reasoning framework, and advanced deployment and security infrastructure.

## System Requirements

- VDS (Virtual Dedicated Server) with at least 4GB RAM, 2 vCPUs
- Ubuntu 20.04 LTS or newer
- 20GB+ free disk space
- No GPU required - the system is optimized for CPU-only operation

## Installation

1. Unpack the compressed archive:
   ```bash
   tar -xzf seren-ai.tar.gz
   cd seren
   ```

2. Run the installation script:
   ```bash
   ./setup-production.sh
   ```

The installation script will:
- Install all required dependencies
- Configure the database
- Set up the security environment
- Prepare the model integration

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PORT=5000
NODE_ENV=production
SESSION_SECRET=your_session_secret_here
DATABASE_URL=postgres://username:password@localhost:5432/seren
```

### Database Setup

The system requires PostgreSQL. If you're using an external PostgreSQL instance, update the `DATABASE_URL` in your `.env` file.

For local PostgreSQL setup:

```bash
./scripts/setup-database.sh
```

## Starting the System

Run the production server:

```bash
npm run start:prod
```

Or use the provided systemd service:

```bash
sudo cp seren.service /etc/systemd/system/
sudo systemctl enable seren
sudo systemctl start seren
```

## Security Hardening

Run the security hardening script to apply military-grade security protocols:

```bash
./security-hardening.sh
```

This script:
- Sets up proper file permissions
- Configures a firewall
- Implements rate limiting
- Sets up secure HTTP headers
- Enables CORS protection

## Model Integration

### Local Models (Offline Mode)

By default, Seren operates in offline mode using local emulated models. This works without requiring external dependencies.

### Integrating Ollama (Optional)

For enhanced capabilities, you can integrate with Ollama:

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull the required models:
   ```bash
   ollama pull qwen2.5:7b-omni
   ollama pull olympiccoder:7b
   ```

3. Update your `.env` file:
   ```
   USE_OLLAMA=true
   OLLAMA_HOST=http://localhost:11434
   ```

## API Documentation

### Virtual Computer API

The Virtual Computer API provides access to AI model capabilities:

#### POST /api/virtual-computer

Request:
```json
{
  "prompt": "Your query or code request",
  "model": "qwen|olympic|hybrid",
  "operationId": "optional-tracking-id"
}
```

Response:
```json
{
  "model": "qwen|olympic|hybrid",
  "generated_text": "Response from the AI model",
  "metadata": {
    "system": "virtual-computer",
    "timestamp": "2025-04-23T05:03:34.000Z",
    "model_version": "qwen2.5-7b-omni|olympiccoder-7b|hybrid"
  }
}
```

#### GET /api/health

Returns the system health status.

Response:
```json
{
  "status": "ok",
  "services": {
    "qwen": true,
    "olympic": true,
    "database": true
  },
  "uptime": 3600 
}
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   
   Check your PostgreSQL service:
   ```bash
   systemctl status postgresql
   ```
   Verify DATABASE_URL in your .env file.

2. **Model Initialization Failures**
   
   Check model logs:
   ```bash
   tail -f logs/model-integration.log
   ```
   
3. **Performance Issues**
   
   Adjust memory allocation in `config.json` based on your server resources.

### Support

For technical issues, refer to the troubleshooting documentation in the `docs` directory or contact support@luxima.ai.

## Production Optimization

For VDS deployment, consider:

1. Setting up Nginx as a reverse proxy
2. Enabling HTTP/2
3. Configuring proper caching
4. Using PM2 for process management

A sample Nginx configuration is provided in `docs/nginx-config.conf`.

## Monitoring

To monitor the system in production:

```bash
npm run monitor
```

This provides real-time metrics on:
- API response times
- Memory usage
- Model performance
- Error rates

## Backup and Recovery

Regular backups are recommended:

```bash
./scripts/backup.sh
```

This will create a complete backup of your database and configuration in `backups/`.

## License

All rights reserved. This software is provided for your exclusive use and may not be redistributed.