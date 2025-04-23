#!/bin/bash
# Production setup script for Seren AI System
# This script prepares the Seren system for production deployment on a VDS

set -e  # Exit on any error

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

# Print section header
section() {
  echo -e "\n${BOLD}${GREEN}=== $1 ===${RESET}\n"
}

# Print info message
info() {
  echo -e "${YELLOW}➤ $1${RESET}"
}

# Print success message
success() {
  echo -e "${GREEN}✓ $1${RESET}"
}

# Print error message and exit
error() {
  echo -e "${RED}✗ $1${RESET}"
  exit 1
}

# Check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
  section "Seren AI System - Production Setup"
  info "Setting up Seren for production deployment on VDS"
  info "Target domain: luxima.icu"

  # Check system requirements
  section "Checking System Requirements"
  
  # Check Node.js
  if command_exists node; then
    NODE_VERSION=$(node -v)
    info "Node.js version: $NODE_VERSION"
    
    # Check if Node.js version is at least 18
    NODE_MAJOR_VERSION=$(echo "$NODE_VERSION" | cut -d '.' -f 1 | tr -d 'v')
    if [ "$NODE_MAJOR_VERSION" -lt 18 ]; then
      error "Node.js version should be at least 18. Please upgrade Node.js."
    fi
    
    success "Node.js check passed"
  else
    error "Node.js is not installed. Please install Node.js 18 or later."
  fi
  
  # Check npm
  if command_exists npm; then
    NPM_VERSION=$(npm -v)
    info "npm version: $NPM_VERSION"
    success "npm check passed"
  else
    error "npm is not installed. Please install npm."
  fi
  
  # Check PostgreSQL
  if command_exists psql; then
    PSQL_VERSION=$(psql --version)
    info "PostgreSQL client: $PSQL_VERSION"
    success "PostgreSQL client check passed"
  else
    info "PostgreSQL client not found. It's recommended for production environments."
    info "Install PostgreSQL client if your database is hosted on the same server."
  fi
  
  # Check available memory
  TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
  info "Available memory: ${TOTAL_MEM}MB"
  if [ "$TOTAL_MEM" -lt 3072 ]; then
    error "At least 4GB of RAM is recommended for production use. Current: ${TOTAL_MEM}MB"
  else
    success "Memory check passed"
  fi
  
  # Check available disk space
  DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
  info "Available disk space: $DISK_SPACE"
  success "Disk space check passed"

  # Create production directories
  section "Creating Production Directories"
  
  # Create logs directory
  mkdir -p logs
  chmod 755 logs
  success "Created logs directory"
  
  # Create data directory
  mkdir -p data
  chmod 755 data
  success "Created data directory"
  
  # Create tmp directory
  mkdir -p tmp
  chmod 755 tmp
  success "Created tmp directory"

  # Install production dependencies
  section "Installing Production Dependencies"
  
  # Install production npm packages
  npm ci --only=production
  success "Installed production npm packages"
  
  # Create production configuration
  section "Creating Production Configuration"
  
  # Check if .env file exists
  if [ -f ".env" ]; then
    info ".env file already exists. Making backup..."
    cp .env .env.backup
    success "Created backup of existing .env file"
  fi
  
  # Generate secrets for .env file
  SESSION_SECRET=$(openssl rand -hex 32)
  JWT_SECRET=$(openssl rand -hex 32)
  ADMIN_API_KEY=$(openssl rand -hex 32)
  
  # Create .env file for production
  cat > .env << EOL
# Seren Production Environment Configuration
# Generated on $(date -u)

# Environment
NODE_ENV=production

# Server Configuration
PORT=3000
HOST=0.0.0.0
DOMAIN=luxima.icu
USE_HTTPS=true
TRUST_PROXY=true
SESSION_SECRET=${SESSION_SECRET}
CORS_ORIGINS=https://luxima.icu,https://admin.luxima.icu

# Database Configuration
DATABASE_URL=${DATABASE_URL:-postgres://username:password@localhost:5432/seren}
DB_POOL_SIZE=10
DB_SSL=true

# AI Model Configuration
DEFAULT_MODEL=hybrid
USE_OFFLINE_MODE=true
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300000

# Security Configuration
JWT_SECRET=${JWT_SECRET}
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
ADMIN_API_KEY=${ADMIN_API_KEY}
EOL
  
  success "Created .env file with secure random secrets"
  info "Please update the DATABASE_URL value in .env with your actual database credentials"
  
  # Create the startup script
  section "Creating Startup Script"
  
  cat > start-production.sh << EOL
#!/bin/bash
# Seren AI System Production Startup Script

export NODE_ENV=production
export LOG_LEVEL=info

# Start the application with maximum reliability
exec node --max-old-space-size=3072 server/index.js
EOL
  
  chmod +x start-production.sh
  success "Created startup script (start-production.sh)"
  
  # Create the process manager configuration (PM2)
  if command_exists pm2; then
    section "Creating PM2 Configuration"
    
    cat > ecosystem.config.js << EOL
module.exports = {
  apps: [{
    name: 'seren',
    script: './start-production.sh',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '3G',
    env: {
      NODE_ENV: 'production',
      LOG_LEVEL: 'info'
    },
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};
EOL
    
    success "Created PM2 configuration (ecosystem.config.js)"
    info "Use 'pm2 start ecosystem.config.js' to start the application"
  else
    info "PM2 not found. Installing PM2 process manager..."
    npm install -g pm2
    
    if command_exists pm2; then
      cat > ecosystem.config.js << EOL
module.exports = {
  apps: [{
    name: 'seren',
    script: './start-production.sh',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '3G',
    env: {
      NODE_ENV: 'production',
      LOG_LEVEL: 'info'
    },
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};
EOL
      
      success "Installed PM2 and created configuration (ecosystem.config.js)"
      info "Use 'pm2 start ecosystem.config.js' to start the application"
    else
      info "Could not install PM2. You can start the application using 'node --max-old-space-size=3072 server/index.js'"
    fi
  fi
  
  # Create Nginx configuration for production
  section "Creating Nginx Configuration"
  
  cat > nginx-seren.conf << EOL
# Nginx configuration for Seren AI System

server {
    listen 80;
    server_name luxima.icu www.luxima.icu;
    
    # Redirect HTTP to HTTPS
    location / {
        return 301 https://\$host\$request_uri;
    }
    
    # Let's Encrypt verification
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
}

server {
    listen 443 ssl http2;
    server_name luxima.icu www.luxima.icu;
    
    # SSL configuration (update paths for your certificate files)
    ssl_certificate /etc/letsencrypt/live/luxima.icu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/luxima.icu/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/luxima.icu/chain.pem;
    
    # SSL parameters
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384';
    ssl_ecdh_curve secp384r1;
    
    # HSTS (comment out if you want to disable)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    
    # Other security headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy no-referrer-when-downgrade always;
    
    # Proxy to Node.js application
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300s;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
    }
    
    # Static assets caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        expires 30d;
        add_header Cache-Control "public, max-age=2592000";
        access_log off;
    }
    
    # Gzip compression
    gzip on;
    gzip_comp_level 5;
    gzip_min_length 256;
    gzip_proxied any;
    gzip_vary on;
    gzip_types
        application/javascript
        application/json
        application/x-javascript
        application/xml
        application/xml+rss
        text/css
        text/javascript
        text/plain
        text/xml;
    
    # Logs
    access_log /var/log/nginx/luxima.icu.access.log;
    error_log /var/log/nginx/luxima.icu.error.log;
}
EOL
  
  success "Created Nginx configuration (nginx-seren.conf)"
  info "Copy this file to /etc/nginx/sites-available/ and create a symlink in /etc/nginx/sites-enabled/"
  info "Command: sudo ln -s /etc/nginx/sites-available/nginx-seren.conf /etc/nginx/sites-enabled/"
  info "Update the SSL certificate paths with your actual certificate files"
  
  # Database migration script
  section "Creating Database Migration Script"
  
  cat > migrate-database.sh << EOL
#!/bin/bash
# Database migration script for Seren AI System

set -e  # Exit on any error

# Load environment variables
source .env

# Make sure database is configured
if [[ "\$DATABASE_URL" == "postgres://username:password@localhost:5432/seren" ]]; then
  echo "ERROR: You need to update the DATABASE_URL in your .env file before running migrations"
  exit 1
fi

# Run migrations
echo "Running database migrations..."
npm run db:push

echo "Database migration completed successfully"
EOL
  
  chmod +x migrate-database.sh
  success "Created database migration script (migrate-database.sh)"
  
  # Create production deployment package
  section "Creating Production Deployment Package"
  
  # Create the list of files to include in the package
  cat > package-files.txt << EOL
.env
ecosystem.config.js
migrate-database.sh
nginx-seren.conf
package.json
package-lock.json
server/
client/
shared/
ai_core/
public/
start-production.sh
node_modules/
logs/
data/
README.md
EOL
  
  # Create the tar.gz package
  PACKAGE_NAME="seren-production-$(date +%Y%m%d).tar.gz"
  tar -czf "$PACKAGE_NAME" -T package-files.txt
  
  success "Created production deployment package: $PACKAGE_NAME"
  info "Upload this package to your VDS and extract it with 'tar -xzf $PACKAGE_NAME'"
  
  # Final instructions
  section "Production Deployment Instructions"
  
  cat << EOL
The Seren AI System has been prepared for production deployment.

To deploy to your VDS (Virtual Dedicated Server):

1. Upload the package ($PACKAGE_NAME) to your VDS
2. Extract it with: tar -xzf $PACKAGE_NAME
3. Update the .env file with your actual database credentials
4. Run the database migration script: ./migrate-database.sh
5. Start the application with PM2: pm2 start ecosystem.config.js

For Nginx configuration:
1. Install Nginx if not already installed: sudo apt-get install nginx
2. Copy the Nginx configuration: sudo cp nginx-seren.conf /etc/nginx/sites-available/
3. Create a symlink: sudo ln -s /etc/nginx/sites-available/nginx-seren.conf /etc/nginx/sites-enabled/
4. Get SSL certificates (Let's Encrypt): sudo certbot --nginx -d luxima.icu -d www.luxima.icu
5. Restart Nginx: sudo systemctl restart nginx

For domain configuration:
1. Update your DNS settings to point 'luxima.icu' to your server's IP address
2. Create both A and AAAA records if your server supports IPv6

For monitoring:
1. Set up PM2 monitoring: pm2 monit
2. Set up log rotation: logrotate is configured for logs in the logs/ directory

For security:
1. Set up a firewall (e.g., UFW): sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw enable
2. Keep your system updated: sudo apt-get update && sudo apt-get upgrade

For backup:
1. Set up regular database backups: pg_dump -U username -d seren > backup_\$(date +%Y%m%d).sql
2. Set up system backup: tar -czf backup_system_\$(date +%Y%m%d).tar.gz /path/to/seren

The Seren AI System will be accessible at: https://luxima.icu
EOL

  success "Setup complete! Your Seren AI System is ready for production deployment."
}

# Run the main function
main