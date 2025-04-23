# Seren AI System - Production Deployment Guide

## Overview

Seren is a production-ready, bleeding-edge autonomous AI development platform that enables intelligent system creation through advanced technological integrations and adaptive AI workflows. This document provides detailed instructions for deploying Seren to a production VDS (Virtual Dedicated Server) environment.

## System Architecture

Seren is built on a multi-model AI architecture utilizing:

- **OpenManus Framework**: An integrated agent system that coordinates autonomous AI models
- **Neuro-Symbolic Reasoning**: Advanced reasoning capabilities combining neural networks and symbolic logic
- **Hybrid Model Integration**: Direct integration of Qwen2.5-7b-omni and OlympicCoder-7B models
- **WebSocket Communication**: Real-time communication between client and server
- **Express API Backend**: RESTful services for data persistence and authentication
- **Postgres Database**: SQL database for persistent storage
- **React Client Interface**: Modern web interface for interacting with the AI system

## Key Features

- **Completely Offline Operation**: Runs without external dependencies using locally hosted AI models
- **Bleeding-Edge AI Integration**: Beyond state-of-the-art AI combining multiple models into a unified system
- **Self-Contained Architecture**: Everything needed runs within the application (no GPU required)
- **Military-Grade Security**: Robust security features throughout the system
- **VDS Optimization**: Optimized for running on Virtual Dedicated Server environments
- **Domain Integration**: Configured for luxima.icu domain

## Prerequisites

Before deploying Seren to production, ensure your VDS meets the following requirements:

- **Operating System**: Ubuntu 20.04 LTS or later (or equivalent Linux distribution)
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **CPU**: Minimum 2 vCPUs, recommended 4+ vCPUs
- **Storage**: Minimum 20GB SSD, recommended 50GB+
- **Node.js**: Version 18 or later
- **PostgreSQL**: Version 12 or later
- **Nginx**: Latest stable version for reverse proxy
- **Domain**: Configured domain (luxima.icu) pointing to your server

## Installation

### 1. Server Preparation

First, ensure your server is updated and has the necessary dependencies:

```bash
# Update package lists
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install -y curl git build-essential nginx

# Install Node.js 18 (if not already installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install PM2 for process management
sudo npm install -g pm2

# Install PostgreSQL (if hosting the database on the same server)
sudo apt install -y postgresql postgresql-contrib
```

### 2. Database Setup

If you're hosting the PostgreSQL database on the same server:

```bash
# Create a new PostgreSQL user and database
sudo -i -u postgres psql -c "CREATE USER seren WITH PASSWORD 'your_strong_password';"
sudo -i -u postgres psql -c "CREATE DATABASE seren OWNER seren;"
sudo -i -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE seren TO seren;"
```

### 3. Deployment

Deploy the Seren AI System to your VDS:

```bash
# Create application directory
sudo mkdir -p /opt/seren
sudo chown $USER:$USER /opt/seren

# Extract the deployment package
tar -xzf seren-production-YYYYMMDD.tar.gz -C /opt/seren
cd /opt/seren

# Update environment configuration
nano .env
# Update DATABASE_URL and other settings as needed

# Run database migrations
./migrate-database.sh

# Start the application with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### 4. Nginx Configuration

Configure Nginx as a reverse proxy for the Seren AI System:

```bash
# Copy Nginx configuration
sudo cp nginx-seren.conf /etc/nginx/sites-available/

# Create a symlink
sudo ln -s /etc/nginx/sites-available/nginx-seren.conf /etc/nginx/sites-enabled/

# Remove default configuration if necessary
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 5. SSL Certificate

Set up SSL certificates using Let's Encrypt:

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d luxima.icu -d www.luxima.icu

# Test automatic renewal
sudo certbot renew --dry-run
```

### 6. Firewall Configuration

Configure the firewall to allow only necessary traffic:

```bash
# Install UFW if not already installed
sudo apt install -y ufw

# Allow SSH, HTTP, and HTTPS
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https

# Enable firewall
sudo ufw enable
```

### 7. Domain Configuration

Configure your domain to point to your server:

1. Access your domain registrar's DNS settings
2. Create an A record pointing `luxima.icu` to your server's IP address
3. Create an A record pointing `www.luxima.icu` to your server's IP address
4. If your server supports IPv6, add AAAA records as well

## System Architecture Details

### Client-Side Components

The client-side of Seren is built with:

- **React**: Frontend framework for building the user interface
- **TypeScript**: Type-safe JavaScript for improved code quality
- **TailwindCSS**: Utility-first CSS framework for styling
- **Shadcn UI**: Component library for consistent design
- **React Query**: Data fetching and state management
- **WebSocket Client**: Real-time communication with the server

### Server-Side Components

The server-side of Seren is built with:

- **Node.js**: JavaScript runtime for the server
- **Express**: Web framework for handling HTTP requests
- **WebSocket Server**: Real-time communication with clients
- **PostgreSQL**: Relational database for data storage
- **Drizzle ORM**: Database ORM for type-safe SQL queries
- **Direct AI Integration**: Self-contained AI model integration without external dependencies

### AI Components

The AI system in Seren consists of:

- **OpenManus Framework**: Integrated agent system for autonomous operation
- **Hybrid Model Integration**: Combined models for enhanced capabilities
- **Direct Integration Module**: Self-contained AI response generation
- **Neuro-Symbolic Reasoning**: Advanced reasoning capabilities
- **Metacognitive System**: Self-improvement and optimization

## Configuration

### Environment Variables

The `.env` file contains all configuration settings for Seren. Key settings include:

- **SERVER_PORT**: The port the server listens on (default: 3000)
- **DATABASE_URL**: PostgreSQL connection string
- **NODE_ENV**: Environment setting (should be "production")
- **SESSION_SECRET**: Secret for session encryption
- **JWT_SECRET**: Secret for JWT token encryption
- **DOMAIN**: Domain name for the application (luxima.icu)
- **USE_OFFLINE_MODE**: Should always be "true" for the offline AI integration

### Database Configuration

The PostgreSQL database requires:

- **Username**: Database username
- **Password**: Strong database password
- **Database Name**: Name of the database (e.g., "seren")
- **Host**: Database host (localhost if on same server)
- **Port**: Database port (default: 5432)

## Maintenance

### Backups

Set up regular backups for your database and application:

```bash
# Database backup script
sudo -i -u postgres pg_dump seren > /opt/backups/seren_db_$(date +%Y%m%d).sql

# Application backup
tar -czf /opt/backups/seren_app_$(date +%Y%m%d).tar.gz /opt/seren
```

### Monitoring

Use PM2 to monitor the application:

```bash
# Check application status
pm2 status

# View logs
pm2 logs seren

# Monitor in real-time
pm2 monit
```

### Updates

To update the Seren AI System:

1. Back up the current application and database
2. Extract the new package to a temporary directory
3. Copy the new files to the application directory
4. Update the .env file if needed
5. Run database migrations
6. Restart the application with PM2

## Troubleshooting

### Common Issues

1. **Server won't start**:
   - Check the logs: `pm2 logs seren`
   - Verify Node.js version: `node -v`
   - Ensure database is running: `sudo systemctl status postgresql`

2. **Database connection errors**:
   - Verify database credentials in .env
   - Check PostgreSQL is running: `sudo systemctl status postgresql`
   - Test connection: `psql -U seren -h localhost -d seren`

3. **WebSocket connection issues**:
   - Check Nginx configuration for WebSocket support
   - Verify that the application is running: `pm2 status`
   - Check client-side console for connection errors

4. **AI model not responding**:
   - Verify the system has enough memory
   - Check server logs for model initialization: `pm2 logs seren`
   - Restart the application: `pm2 restart seren`

## Security Considerations

Seren implements several security measures:

- **HTTP Security Headers**: Set via Nginx and Express
- **TLS/SSL Encryption**: HTTPS only with strong ciphers
- **CSRF Protection**: Cross-Site Request Forgery prevention
- **XSS Protection**: Cross-Site Scripting prevention
- **Rate Limiting**: Prevents abuse of the API
- **Strong Cryptography**: Secure password hashing and encryption
- **Input Validation**: Strict validation of all user input
- **Principle of Least Privilege**: Limited database user permissions

## Performance Optimization

To optimize performance:

- **Database Indexes**: Ensure proper indexes are in place
- **Connection Pooling**: Configure database connection pooling
- **Resource Limits**: Set appropriate memory limits in PM2 configuration
- **Static Asset Caching**: Configure caching headers in Nginx
- **Compression**: Enable gzip compression in Nginx

## Additional Resources

- **OpenManus Framework**: Documentation in `docs/openmanus.md`
- **API Documentation**: API endpoints in `docs/api.md`
- **Troubleshooting Guide**: Extended troubleshooting in `docs/troubleshooting.md`
- **Security Guidelines**: Security best practices in `docs/security.md`

## Contact Information

For additional support, contact:

- **Admin**: admin@luxima.icu
- **Support**: support@luxima.icu

## License

Proprietary software. All rights reserved.

---

© 2025 Luxima, Inc. All Rights Reserved.