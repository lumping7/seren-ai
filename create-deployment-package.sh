#!/bin/bash
# Script to create a production deployment package for Seren AI System

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

# Main function
main() {
  section "Seren AI System - Creating Deployment Package"
  info "Building a production-ready deployment package"
  
  # Check if necessary files exist
  if [ ! -f "setup-production.sh" ]; then
    error "setup-production.sh not found. Run this script from the project root."
  fi
  
  if [ ! -f "README-PRODUCTION.md" ]; then
    error "README-PRODUCTION.md not found. Run this script from the project root."
  fi
  
  # Create build directory
  BUILD_DIR="seren-build"
  PACKAGE_NAME="seren-production-$(date +%Y%m%d).tar.gz"
  
  info "Creating build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR" 2>/dev/null || true
  mkdir -p "$BUILD_DIR"
  
  # Copy necessary files
  section "Copying Files"
  
  # Project files
  info "Copying project files..."
  
  # Create directory structure
  mkdir -p "$BUILD_DIR/server"
  mkdir -p "$BUILD_DIR/client"
  mkdir -p "$BUILD_DIR/shared"
  mkdir -p "$BUILD_DIR/ai_core"
  mkdir -p "$BUILD_DIR/logs"
  mkdir -p "$BUILD_DIR/data"
  
  # Copy server files
  cp -r server/* "$BUILD_DIR/server/" 2>/dev/null || true
  success "Copied server files"
  
  # Copy client files
  cp -r client/* "$BUILD_DIR/client/" 2>/dev/null || true
  success "Copied client files"
  
  # Copy shared files
  cp -r shared/* "$BUILD_DIR/shared/" 2>/dev/null || true
  success "Copied shared files"
  
  # Copy AI core
  cp -r ai_core/* "$BUILD_DIR/ai_core/" 2>/dev/null || true
  success "Copied AI core files"
  
  # Copy configuration files
  cp package.json "$BUILD_DIR/" 2>/dev/null || true
  cp package-lock.json "$BUILD_DIR/" 2>/dev/null || true
  cp tsconfig.json "$BUILD_DIR/" 2>/dev/null || true
  cp README-PRODUCTION.md "$BUILD_DIR/README.md" 2>/dev/null || true
  success "Copied configuration files"
  
  # Copy deployment scripts
  cp setup-production.sh "$BUILD_DIR/" 2>/dev/null || true
  success "Copied deployment scripts"
  
  # Clean up unnecessary files
  section "Cleaning Up"
  
  # Remove development files
  find "$BUILD_DIR" -name "*.test.ts" -delete
  find "$BUILD_DIR" -name "*.test.tsx" -delete
  find "$BUILD_DIR" -name "*.spec.ts" -delete
  find "$BUILD_DIR" -name "*.spec.tsx" -delete
  find "$BUILD_DIR" -name "__tests__" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -name ".DS_Store" -delete
  success "Removed development files"
  
  # Build the client
  section "Building Client"
  
  # Create production build script
  cat > "$BUILD_DIR/build-client.sh" << EOL
#!/bin/bash
# Build the client for production

cd \$(dirname "\$0")
echo "Installing dependencies..."
npm ci
echo "Building client..."
npm run build
echo "Client built successfully"
EOL
  
  chmod +x "$BUILD_DIR/build-client.sh"
  success "Created client build script"
  
  # Create production configuration
  section "Creating Production Configuration"
  
  # Create example .env file
  cat > "$BUILD_DIR/.env.example" << EOL
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
SESSION_SECRET=replace-with-strong-secret-in-production
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
JWT_SECRET=replace-with-strong-secret-in-production
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
ADMIN_API_KEY=replace-with-strong-api-key-in-production
EOL
  
  success "Created example .env file"
  
  # Create Nginx configuration
  cat > "$BUILD_DIR/nginx-seren.conf" << EOL
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
  
  success "Created Nginx configuration"
  
  # Create startup script
  cat > "$BUILD_DIR/start-production.sh" << EOL
#!/bin/bash
# Seren AI System Production Startup Script

export NODE_ENV=production
export LOG_LEVEL=info

# Load environment variables
if [ -f ".env" ]; then
  source .env
fi

# Configure server
echo "Starting Seren AI System in production mode..."
echo "Using domain: \${DOMAIN:-luxima.icu}"
echo "Using port: \${PORT:-3000}"

# Start the application with maximum reliability
exec node --max-old-space-size=3072 server/index.js
EOL
  
  chmod +x "$BUILD_DIR/start-production.sh"
  success "Created startup script"
  
  # Create the PM2 configuration
  cat > "$BUILD_DIR/ecosystem.config.js" << EOL
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
  
  success "Created PM2 configuration"
  
  # Database migration script
  cat > "$BUILD_DIR/migrate-database.sh" << EOL
#!/bin/bash
# Database migration script for Seren AI System

set -e  # Exit on any error

# Load environment variables
if [ -f ".env" ]; then
  source .env
fi

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
  
  chmod +x "$BUILD_DIR/migrate-database.sh"
  success "Created database migration script"
  
  # Create deployment instructions
  cat > "$BUILD_DIR/DEPLOY.md" << EOL
# Seren AI System - Deployment Instructions

This package contains the Seren AI System, a production-ready autonomous AI development platform.

## Quick Start

1. Extract the package to your server
2. Create a .env file (use .env.example as a template)
3. Install dependencies: \`npm ci\`
4. Build the client: \`./build-client.sh\`
5. Run database migrations: \`./migrate-database.sh\`
6. Start the application: \`pm2 start ecosystem.config.js\`

## Configuration

Edit the .env file to configure the system for your environment.
Key settings to update:

- \`DATABASE_URL\`: Your PostgreSQL connection string
- \`SESSION_SECRET\`: A random string for session encryption
- \`JWT_SECRET\`: A random string for JWT token encryption
- \`ADMIN_API_KEY\`: A random string for admin API access
- \`DOMAIN\`: Your domain name (default: luxima.icu)

## Nginx Configuration

Copy nginx-seren.conf to /etc/nginx/sites-available/ and create a symlink:

\`\`\`
sudo cp nginx-seren.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/nginx-seren.conf /etc/nginx/sites-enabled/
sudo systemctl restart nginx
\`\`\`

## SSL Certificate (Let's Encrypt)

\`\`\`
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d luxima.icu -d www.luxima.icu
\`\`\`

## Need Help?

Refer to README.md for detailed documentation.
EOL
  
  success "Created deployment instructions"
  
  # Create the deployment package
  section "Creating Deployment Package"
  
  # Create the tar.gz package
  tar -czf "$PACKAGE_NAME" -C "$BUILD_DIR" .
  
  # Clean up the build directory
  rm -rf "$BUILD_DIR"
  
  success "Created deployment package: $PACKAGE_NAME"
  info "Upload this package to your VDS and extract it with 'tar -xzf $PACKAGE_NAME'"
  
  section "Deployment Package Created Successfully"
  info "Package: $PACKAGE_NAME"
  info "Size: $(du -h "$PACKAGE_NAME" | cut -f1)"
  info "Created: $(date)"
  echo ""
  info "This package contains everything needed to deploy Seren AI System to a production VDS environment."
  info "Follow the instructions in DEPLOY.md after extracting the package."
}

# Run the main function
main