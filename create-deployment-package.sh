#!/bin/bash

# Seren AI - Deployment Package Creator
# This script creates a compressed archive of the Seren AI system ready for deployment

set -e

echo "============================================================"
echo "  Seren AI - Deployment Package Creator"
echo "  Creating optimized production-ready package"
echo "============================================================"
echo ""

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p logs
LOG_FILE="logs/package-$(date +%Y%m%d-%H%M%S).log"
touch $LOG_FILE

log() {
  echo -e "$1" | tee -a $LOG_FILE
}

PACKAGE_NAME="seren-ai-$(date +%Y%m%d).tar.gz"
BUILD_DIR="./build-temp"

log "${GREEN}Step 1: Preparing environment...${NC}"

# Clean up any previous build
rm -rf $BUILD_DIR 2>/dev/null || true
mkdir -p $BUILD_DIR

# Create required directories
mkdir -p $BUILD_DIR/logs
mkdir -p $BUILD_DIR/data
mkdir -p $BUILD_DIR/backups
mkdir -p $BUILD_DIR/scripts
mkdir -p $BUILD_DIR/docs

log "${GREEN}✓ Build directory prepared${NC}"

log "${GREEN}Step 2: Building production assets...${NC}"

# Run build process
NODE_ENV=production npm run build >$LOG_FILE 2>&1 || {
  log "${RED}Failed to build production assets. See ${LOG_FILE} for details.${NC}"
  exit 1
}

log "${GREEN}✓ Production assets built successfully${NC}"

log "${GREEN}Step 3: Copying files...${NC}"

# Copy server files
cp -r server $BUILD_DIR/
cp -r shared $BUILD_DIR/
cp -r ai_core $BUILD_DIR/

# Copy client build
cp -r dist $BUILD_DIR/

# Copy scripts
cp setup-production.sh $BUILD_DIR/
cp security-hardening.sh $BUILD_DIR/
cp scripts/*.sh $BUILD_DIR/scripts/ 2>/dev/null || true

# Copy documentation
cp README-PRODUCTION.md $BUILD_DIR/README.md
cp -r docs/* $BUILD_DIR/docs/ 2>/dev/null || true

# Copy configuration files
cp package.json $BUILD_DIR/
cp package-lock.json $BUILD_DIR/
cp tsconfig.json $BUILD_DIR/
cp -r .eslintrc.* $BUILD_DIR/ 2>/dev/null || true
cp drizzle.config.ts $BUILD_DIR/

# Create empty .env file with example values
cat > $BUILD_DIR/.env.example << EOF
PORT=5000
NODE_ENV=production
SESSION_SECRET=replace_with_your_secret
DATABASE_URL=postgres://postgres:postgres@localhost:5432/seren
USE_OLLAMA=false
# OLLAMA_HOST=http://localhost:11434
EOF

log "${GREEN}✓ Files copied successfully${NC}"

log "${GREEN}Step 4: Cleaning up development files...${NC}"

# Remove unnecessary development files
find $BUILD_DIR -name "*.test.ts" -delete
find $BUILD_DIR -name "*.test.tsx" -delete
find $BUILD_DIR -name "*.spec.ts" -delete
find $BUILD_DIR -name "*.spec.tsx" -delete
find $BUILD_DIR -name ".DS_Store" -delete

# Remove development directories
rm -rf $BUILD_DIR/node_modules 2>/dev/null || true

log "${GREEN}✓ Cleaned up development files${NC}"

log "${GREEN}Step 5: Creating startup script...${NC}"

# Create production startup script
cat > $BUILD_DIR/start.sh << 'EOF'
#!/bin/bash

# Seren AI - Startup Script
NODE_ENV=production node server/index.js
EOF

chmod +x $BUILD_DIR/start.sh

log "${GREEN}✓ Startup script created${NC}"

log "${GREEN}Step 6: Creating systemd service file...${NC}"

# Create systemd service file
cat > $BUILD_DIR/seren.service << 'EOF'
[Unit]
Description=Seren AI - Offline AI Development Platform
After=network.target postgresql.service

[Service]
ExecStart=/usr/bin/node /opt/seren/server/index.js
WorkingDirectory=/opt/seren
Restart=always
User=seren
Group=seren
Environment=NODE_ENV=production
Environment=PORT=5000

[Install]
WantedBy=multi-user.target
EOF

log "${GREEN}✓ Systemd service file created${NC}"

log "${GREEN}Step 7: Creating Nginx configuration...${NC}"

# Create Nginx configuration
mkdir -p $BUILD_DIR/docs
cat > $BUILD_DIR/docs/nginx-config.conf << 'EOF'
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
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

log "${GREEN}✓ Nginx configuration created${NC}"

log "${GREEN}Step 8: Creating production installation readme...${NC}"

# Create quick installation guide
cat > $BUILD_DIR/INSTALL.md << 'EOF'
# Seren AI Quick Installation Guide

1. Unpack the archive:
   ```bash
   tar -xzf seren-ai.tar.gz
   cd seren
   ```

2. Run the installation script:
   ```bash
   ./setup-production.sh
   ```

3. Start the application:
   ```bash
   npm run start:prod
   ```

For detailed instructions, see the README.md file.
EOF

log "${GREEN}✓ Installation guide created${NC}"

log "${GREEN}Step 9: Creating compressed archive...${NC}"

# Create the compressed archive
tar -czf $PACKAGE_NAME -C $BUILD_DIR .

# Get the size of the package
PACKAGE_SIZE=$(du -h $PACKAGE_NAME | cut -f1)

log "${GREEN}✓ Compressed archive created: ${PACKAGE_NAME} (${PACKAGE_SIZE})${NC}"

# Clean up build directory
rm -rf $BUILD_DIR

log ""
log "============================================================"
log "${GREEN}Seren AI deployment package created successfully!${NC}"
log "============================================================"
log ""
log "Package: ${YELLOW}${PACKAGE_NAME}${NC} (${PACKAGE_SIZE})"
log "The package contains everything needed for production deployment."
log ""
log "To deploy:"
log "1. Copy the package to your server"
log "2. Extract: ${YELLOW}tar -xzf ${PACKAGE_NAME}${NC}"
log "3. Run setup: ${YELLOW}./setup-production.sh${NC}"
log ""
log "For detailed instructions, see README.md in the package."
log "============================================================"