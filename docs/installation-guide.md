# Installation Guide

This guide will walk you through installing and deploying the Advanced AI Collaboration System in various environments, from development to production.

## System Requirements

### Minimum Requirements

- **Node.js**: v18 or higher
- **PostgreSQL**: v14 or higher
- **RAM**: 16GB minimum
- **CPU**: 4 cores recommended
- **Storage**: 10GB available space
- **Operating System**: Ubuntu 20.04+ (recommended), Windows 10+, macOS 12+

### Recommended Requirements

- **RAM**: 32GB or more for optimal performance
- **CPU**: 8+ cores for parallel processing
- **Storage**: SSD with 20GB+ available space
- **Database**: Dedicated PostgreSQL instance or managed service (e.g., Neon)
- **Network**: High-bandwidth connection for distributed deployments

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/your-organization/advanced-ai-collaboration-system.git
cd advanced-ai-collaboration-system
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/ai_collaboration

# Server Configuration
PORT=5000
NODE_ENV=production

# Security
SESSION_SECRET=your-secure-random-session-secret

# Optional: Configure resource limits
MAX_CONCURRENT_REQUESTS=5
MAX_CONTEXT_SIZE=8192
```

### 4. Database Setup

#### Option A: Local PostgreSQL

1. Install PostgreSQL:

```bash
# Ubuntu
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# Windows
# Download installer from https://www.postgresql.org/download/windows/
```

2. Create a database:

```bash
sudo -u postgres psql
CREATE DATABASE ai_collaboration;
CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';
GRANT ALL PRIVILEGES ON DATABASE ai_collaboration TO myuser;
\q
```

3. Update your `.env` file with the connection string:

```
DATABASE_URL=postgresql://myuser:mypassword@localhost:5432/ai_collaboration
```

#### Option B: Managed PostgreSQL (Neon, AWS RDS, etc.)

1. Create a database using your provider's interface
2. Update the `DATABASE_URL` in your `.env` file with the provided connection string
3. Ensure your firewall allows connections to the database

### 5. Run Database Migrations

```bash
# Create the database schema
npx tsx scripts/migrate.ts
```

### 6. Build the Application

```bash
npm run build
```

### 7. Start the Server

```bash
npm start
```

The system should now be running on http://localhost:5000 (or the port you specified).

## Deployment Options

### Option 1: Standalone Server

The simplest deployment option is running the system on a single server:

1. Follow installation steps above
2. Configure a process manager like PM2:

```bash
npm install -g pm2
pm2 start dist/index.js --name ai-collaboration
pm2 save
pm2 startup
```

3. Set up Nginx as a reverse proxy:

```
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

4. Configure SSL with Certbot:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Option 2: Docker Deployment

For containerized deployment:

1. Build the Docker image:

```bash
docker build -t ai-collaboration-system .
```

2. Run the container:

```bash
docker run -d \
  --name ai-collaboration \
  -p 5000:5000 \
  -e DATABASE_URL=postgresql://user:password@db-host:5432/ai_collaboration \
  -e SESSION_SECRET=your-session-secret \
  -e NODE_ENV=production \
  ai-collaboration-system
```

3. For Docker Compose, create a `docker-compose.yml`:

```yaml
version: '3'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/ai_collaboration
      - SESSION_SECRET=your-session-secret
      - NODE_ENV=production
    depends_on:
      - db
  
  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=ai_collaboration
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Then run:

```bash
docker-compose up -d
```

### Option 3: Cloud Deployment

#### AWS Elastic Beanstalk

1. Install EB CLI:

```bash
pip install awsebcli
```

2. Initialize EB:

```bash
eb init
```

3. Create an environment:

```bash
eb create production-environment
```

4. Set environment variables:

```bash
eb setenv DATABASE_URL=postgresql://user:password@db-host:5432/ai_collaboration SESSION_SECRET=your-session-secret
```

5. Deploy:

```bash
eb deploy
```

#### Heroku

1. Install Heroku CLI and log in:

```bash
npm install -g heroku
heroku login
```

2. Create a Heroku app:

```bash
heroku create ai-collaboration-system
```

3. Add PostgreSQL:

```bash
heroku addons:create heroku-postgresql:hobby-dev
```

4. Set environment variables:

```bash
heroku config:set SESSION_SECRET=your-session-secret NODE_ENV=production
```

5. Deploy:

```bash
git push heroku main
```

## Post-Installation Setup

### 1. Creating an Admin User

The system automatically creates an admin user with these credentials:
- Username: `admin`
- Password: `admin123`

**Important**: Change this password immediately after first login!

### 2. Initial Configuration

1. Navigate to the admin dashboard at `/admin`
2. Update the default settings in the Settings tab
3. Create initial knowledge domains if needed

### 3. Security Considerations

1. Ensure your server's firewall allows only necessary connections
2. Set up proper user roles and permissions
3. Implement rate limiting at the network level
4. Configure regular database backups
5. Rotate session secrets and credentials regularly

## Resource Scaling

The system automatically adapts to the available resources, but you can tune it for your specific environment:

### For 16GB RAM Servers

```
MAX_CONCURRENT_REQUESTS=2
MAX_CONTEXT_SIZE=4096
USE_QUANTIZATION=true
```

### For 32GB RAM Servers

```
MAX_CONCURRENT_REQUESTS=5
MAX_CONTEXT_SIZE=8192
USE_QUANTIZATION=true
```

### For 64GB+ RAM Servers

```
MAX_CONCURRENT_REQUESTS=10
MAX_CONTEXT_SIZE=16384
USE_QUANTIZATION=false
```

## Health Monitoring

### Setting up Monitoring

1. Configure system monitoring using your preferred tool (e.g., Prometheus, Grafana)
2. Monitor the `/api/ai/status` endpoint for system health
3. Set up alerts for resource constraints or extended response times

### Log Management

The system outputs structured logs that can be processed by standard tools:

1. For production, set up log aggregation with ELK Stack or similar
2. Configure log rotation to prevent disk space issues
3. Regular audit of error logs for potential issues

## Troubleshooting

### Common Issues

#### Database Connection Errors

1. Verify your DATABASE_URL environment variable
2. Check that PostgreSQL is running and accessible
3. Ensure the database user has proper permissions
4. Verify there are no network/firewall restrictions

#### Memory Issues

1. Check resource usage with `top` or a monitoring tool
2. Adjust MAX_CONCURRENT_REQUESTS to a lower value
3. Consider upgrading to a server with more RAM
4. Enable USE_QUANTIZATION for reduced memory footprint

#### Slow Response Times

1. Check database query performance
2. Verify CPU isn't bottlenecked
3. Monitor network latency between components
4. Consider adjusting the resource profile downward

### Support Resources

If you encounter issues not covered in this guide:

1. Check the [troubleshooting documentation](./troubleshooting.md)
2. Visit the GitHub repository issues section
3. Reach out to the community support forum

## Upgrading

To upgrade to a newer version:

1. Back up your data:

```bash
pg_dump -U your_user -d ai_collaboration > backup.sql
```

2. Pull the latest changes:

```bash
git pull origin main
```

3. Install dependencies:

```bash
npm install
```

4. Run migrations:

```bash
npx tsx scripts/migrate.ts
```

5. Rebuild and restart:

```bash
npm run build
npm start   # or restart your process manager
```

## Uninstallation

If you need to remove the system:

1. Stop the running server:

```bash
# If using PM2
pm2 stop ai-collaboration
pm2 delete ai-collaboration

# If using systemd
sudo systemctl stop ai-collaboration

# If using Docker
docker stop ai-collaboration
docker rm ai-collaboration
```

2. Back up data if needed:

```bash
pg_dump -U your_user -d ai_collaboration > final_backup.sql
```

3. Remove the database:

```bash
sudo -u postgres psql
DROP DATABASE ai_collaboration;
DROP USER myuser;
\q
```

4. Remove the application files:

```bash
rm -rf /path/to/advanced-ai-collaboration-system
```