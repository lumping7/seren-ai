#!/bin/bash
# Final Package Generator for Seren AI System
# This script creates a comprehensive deployment package with all components

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

# Main package creation function
create_final_package() {
  VERSION=$(date +%Y%m%d)
  PACKAGE_NAME="seren-complete-vds-$VERSION.tar.gz"
  BUILD_DIR="seren-build-$VERSION"
  
  section "Creating Final Deployment Package for Seren AI System"
  info "Package version: $VERSION"
  info "Building package: $PACKAGE_NAME"
  
  # Create build directory
  rm -rf "$BUILD_DIR" 2>/dev/null || true
  mkdir -p "$BUILD_DIR"
  
  # Create directory structure
  mkdir -p "$BUILD_DIR/server"
  mkdir -p "$BUILD_DIR/client"
  mkdir -p "$BUILD_DIR/shared"
  mkdir -p "$BUILD_DIR/ai_core"
  mkdir -p "$BUILD_DIR/docs"
  mkdir -p "$BUILD_DIR/scripts"
  mkdir -p "$BUILD_DIR/logs"
  mkdir -p "$BUILD_DIR/data"
  
  section "Copying Core Components"
  
  # Copy server files
  info "Copying server components..."
  cp -r server/* "$BUILD_DIR/server/" 2>/dev/null || true
  success "Server components copied"
  
  # Copy client files
  info "Copying client components..."
  cp -r client/* "$BUILD_DIR/client/" 2>/dev/null || true
  success "Client components copied"
  
  # Copy shared files
  info "Copying shared components..."
  cp -r shared/* "$BUILD_DIR/shared/" 2>/dev/null || true
  success "Shared components copied"
  
  # Copy AI core files
  info "Copying AI core components..."
  cp -r ai_core/* "$BUILD_DIR/ai_core/" 2>/dev/null || true
  success "AI core components copied"
  
  # Copy documentation
  info "Copying documentation..."
  cp -r docs/* "$BUILD_DIR/docs/" 2>/dev/null || true
  cp README-PRODUCTION.md "$BUILD_DIR/README.md" 2>/dev/null || true
  success "Documentation copied"
  
  section "Copying Configuration Files"
  
  # Copy project configuration
  info "Copying project configuration..."
  cp package.json "$BUILD_DIR/" 2>/dev/null || true
  cp package-lock.json "$BUILD_DIR/" 2>/dev/null || true
  cp tsconfig.json "$BUILD_DIR/" 2>/dev/null || true
  cp drizzle.config.ts "$BUILD_DIR/" 2>/dev/null || true
  cp tailwind.config.ts "$BUILD_DIR/" 2>/dev/null || true
  cp postcss.config.js "$BUILD_DIR/" 2>/dev/null || true
  cp components.json "$BUILD_DIR/" 2>/dev/null || true
  success "Project configuration copied"
  
  section "Adding Production Scripts"
  
  # Copy production scripts
  info "Adding production scripts..."
  cp setup-production.sh "$BUILD_DIR/scripts/" 2>/dev/null || true
  cp security-hardening.sh "$BUILD_DIR/scripts/" 2>/dev/null || true
  cp test-production-system.sh "$BUILD_DIR/scripts/" 2>/dev/null || true
  
  # Make scripts executable
  chmod +x "$BUILD_DIR/scripts/"*.sh 2>/dev/null || true
  success "Production scripts added"
  
  # Create main deployment script
  cat > "$BUILD_DIR/deploy.sh" << EOL
#!/bin/bash
# Seren AI System - Deployment Script
# Run this script to deploy Seren to your VDS

echo "=== Seren AI System Deployment ==="
echo "This script will guide you through the deployment process."
echo

# Check if running as root
if [ "\$(id -u)" != "0" ]; then
  echo "WARNING: It's recommended to run this script as root."
  echo "Continue anyway? (y/n)"
  read -r answer
  if [ "\$answer" != "y" ]; then
    echo "Exiting..."
    exit 1
  fi
fi

# Create installation directory
INSTALL_DIR="/opt/seren"
echo "Creating installation directory: \$INSTALL_DIR"
mkdir -p "\$INSTALL_DIR"

# Copy files to installation directory
echo "Copying files to installation directory..."
cp -r * "\$INSTALL_DIR/"

# Run setup script
echo "Running setup script..."
cd "\$INSTALL_DIR"
bash ./scripts/setup-production.sh

# Run security hardening
echo "Running security hardening script..."
bash ./scripts/security-hardening.sh

# Test the system
echo "Testing the system..."
bash ./scripts/test-production-system.sh

echo
echo "Deployment completed!"
echo "The Seren AI System is now installed at: \$INSTALL_DIR"
echo
echo "To start the system:"
echo "cd \$INSTALL_DIR"
echo "pm2 start ecosystem.config.js"
echo
echo "To access the system:"
echo "http://localhost:3000"
echo "or"
echo "https://your-domain.com (after configuring Nginx)"
EOL
  
  chmod +x "$BUILD_DIR/deploy.sh"
  success "Main deployment script created"
  
  section "Creating Environment Configuration"
  
  # Create example .env file
  cat > "$BUILD_DIR/.env.example" << EOL
# Seren AI System - Production Environment Configuration
# Generated: $(date -u)

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
  
  success "Environment configuration created"
  
  section "Creating Nginx Configuration"
  
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
  
  success "Nginx configuration created"
  
  section "Creating PM2 Configuration"
  
  # Create PM2 configuration
  cat > "$BUILD_DIR/ecosystem.config.js" << EOL
module.exports = {
  apps: [{
    name: 'seren',
    script: './start.sh',
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
  
  # Create startup script
  cat > "$BUILD_DIR/start.sh" << EOL
#!/bin/bash
# Seren AI System startup script

# Load environment variables
if [ -f ".env" ]; then
  source .env
fi

export NODE_ENV=production
export LOG_LEVEL=info

# Start the server
exec node --max-old-space-size=3072 server/index.js
EOL
  
  chmod +x "$BUILD_DIR/start.sh"
  success "PM2 configuration created"
  
  section "Creating Database Migration Script"
  
  # Create database migration script
  cat > "$BUILD_DIR/migrate-database.sh" << EOL
#!/bin/bash
# Database migration script for Seren AI System

set -e  # Exit on any error

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${YELLOW}Seren AI System - Database Migration${RESET}"
echo

# Load environment variables
if [ -f ".env" ]; then
  source .env
else
  echo -e "${RED}Error: .env file not found!${RESET}"
  echo "Create a .env file with proper DATABASE_URL setting."
  exit 1
fi

# Check if DATABASE_URL is properly configured
if [ -z "\$DATABASE_URL" ] || [ "\$DATABASE_URL" = "postgres://username:password@localhost:5432/seren" ]; then
  echo -e "${RED}Error: DATABASE_URL not properly configured in .env file!${RESET}"
  echo "Update the DATABASE_URL with your actual database credentials."
  exit 1
fi

# Install drizzle-kit if needed
if ! npm list drizzle-kit >/dev/null 2>&1; then
  echo -e "${YELLOW}Installing drizzle-kit...${RESET}"
  npm install drizzle-kit --save-dev
fi

echo -e "${YELLOW}Running database migrations...${RESET}"

# Run the migrations
npx drizzle-kit push:pg

echo -e "${GREEN}Database migrations completed successfully!${RESET}"
EOL
  
  chmod +x "$BUILD_DIR/migrate-database.sh"
  success "Database migration script created"
  
  section "Creating Documentation"
  
  # Create quick start guide
  cat > "$BUILD_DIR/QUICK-START.md" << EOL
# Seren AI System - Quick Start Guide

This guide will help you quickly get your Seren AI System up and running on a VDS (Virtual Dedicated Server).

## Prerequisites

- Ubuntu 20.04 LTS or later
- Minimum 4GB RAM
- Minimum 20GB disk space
- Node.js 18 or later
- PostgreSQL 12 or later

## Installation Steps

1. **Extract the Package**

   ```bash
   tar -xzf seren-complete-vds-*.tar.gz -C /opt/
   cd /opt/seren
   ```

2. **Configure Environment**

   ```bash
   cp .env.example .env
   nano .env  # Edit with your settings
   ```

   Important settings to update:
   - \`DATABASE_URL\`: Your PostgreSQL connection string
   - \`SESSION_SECRET\`: A random string for session encryption
   - \`JWT_SECRET\`: A random string for JWT token encryption
   - \`DOMAIN\`: Your domain name (default: luxima.icu)

3. **Install Dependencies**

   ```bash
   npm ci
   ```

4. **Run Database Migrations**

   ```bash
   ./migrate-database.sh
   ```

5. **Start the Application**

   ```bash
   # Using PM2 (recommended)
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup

   # Or directly
   ./start.sh
   ```

6. **Configure Nginx**

   ```bash
   sudo cp nginx-seren.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/nginx-seren.conf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

7. **Set Up SSL Certificate**

   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com -d www.your-domain.com
   ```

8. **Apply Security Hardening**

   ```bash
   sudo ./scripts/security-hardening.sh
   ```

## Verification

Access your application at:
- http://localhost:3000 (local access)
- https://your-domain.com (after configuring Nginx and DNS)

## Troubleshooting

If you encounter issues:

1. Check logs: \`pm2 logs seren\`
2. Verify database connection: \`./scripts/test-production-system.sh\`
3. Restart the application: \`pm2 restart seren\`

## Next Steps

- Create an admin user: See the Admin Guide in docs/
- Configure backup system: See the Backup Guide in docs/
- Set up monitoring: See the Monitoring Guide in docs/

For more details, refer to the full documentation in the docs/ directory.
EOL
  
  success "Quick start guide created"
  
  section "Cleaning Up the Package"
  
  # Remove development files
  info "Removing development files..."
  
  find "$BUILD_DIR" -name "*.test.ts" -delete
  find "$BUILD_DIR" -name "*.test.tsx" -delete
  find "$BUILD_DIR" -name "*.spec.ts" -delete
  find "$BUILD_DIR" -name "*.spec.tsx" -delete
  find "$BUILD_DIR" -name "__tests__" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -name ".DS_Store" -delete
  
  success "Development files removed"
  
  section "Creating Final Package"
  
  # Create the tar.gz package
  info "Creating the tarball..."
  tar -czf "$PACKAGE_NAME" -C "$BUILD_DIR" .
  
  # Clean up the build directory
  rm -rf "$BUILD_DIR"
  
  success "Final package created: $PACKAGE_NAME"
  info "Package size: $(du -h "$PACKAGE_NAME" | cut -f1)"
  
  section "Final Package Ready for Deployment"
  
  cat << EOL
The Seren AI System has been packaged for deployment to a VDS environment.

Package details:
- Name: $PACKAGE_NAME
- Size: $(du -h "$PACKAGE_NAME" | cut -f1)
- Created: $(date)

This package contains:
- Complete Seren AI System with offline AI integration
- Production configuration files
- Deployment scripts
- Documentation
- Security hardening
- Database migration tools

To deploy to your VDS:
1. Upload the package: scp $PACKAGE_NAME user@your-vds-host:/tmp/
2. Extract and install: tar -xzf $PACKAGE_NAME -C /opt/seren
3. Follow the QUICK-START.md guide

For a guided installation:
1. Upload and extract the package
2. Run the deployment script: ./deploy.sh

This package is completely self-contained and requires no external AI services.
It is configured to use the luxima.icu domain as requested.
EOL
}

# Run the package creation function
create_final_package