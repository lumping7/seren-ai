#!/bin/bash

# Seren AI - Production Setup Script
# This script sets up the Seren AI system for production use

set -e

echo "============================================================"
echo "  Seren AI - Production Environment Setup"
echo "  Fully Offline AI Development Platform"
echo "============================================================"
echo ""

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Warning: This script isn't running as root. Some operations might fail.${NC}"
  echo "Consider running with sudo if you encounter permission issues."
  echo ""
fi

# Create log directory
mkdir -p logs
LOG_FILE="logs/setup-$(date +%Y%m%d-%H%M%S).log"
touch $LOG_FILE

log() {
  echo -e "$1" | tee -a $LOG_FILE
}

log "${GREEN}Step 1: Checking system requirements...${NC}"

# Check memory
MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
if [ "$MEM_TOTAL" -lt 4000000 ]; then
  log "${RED}Warning: You have less than 4GB of RAM. Performance may be affected.${NC}"
else
  log "${GREEN}✓ Memory check passed${NC}"
fi

# Check disk space
DISK_SPACE=$(df -h . | tail -1 | awk '{print $4}' | sed 's/G//')
if (( $(echo "$DISK_SPACE < 20" | bc -l) )); then
  log "${RED}Warning: You have less than 20GB of free disk space. Consider freeing up space.${NC}"
else
  log "${GREEN}✓ Disk space check passed${NC}"
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
  log "${YELLOW}Node.js not found. Installing...${NC}"
  
  # Add Node.js repository and install
  if ! command -v curl &> /dev/null; then
    apt-get update && apt-get install -y curl
  fi

  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
  
  log "${GREEN}✓ Node.js installed successfully${NC}"
else
  NODE_VERSION=$(node -v)
  log "${GREEN}✓ Node.js is already installed (${NODE_VERSION})${NC}"
fi

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
  log "${YELLOW}PostgreSQL not found. Installing...${NC}"
  
  # Install PostgreSQL
  apt-get update
  apt-get install -y postgresql postgresql-contrib
  
  # Start PostgreSQL service
  systemctl enable postgresql
  systemctl start postgresql
  
  log "${GREEN}✓ PostgreSQL installed successfully${NC}"
else
  PSQL_VERSION=$(psql --version)
  log "${GREEN}✓ PostgreSQL is already installed (${PSQL_VERSION})${NC}"
fi

log "${GREEN}Step 2: Installing dependencies...${NC}"

# Install production dependencies
npm ci --production > /dev/null 2>> $LOG_FILE || npm install --production > /dev/null 2>> $LOG_FILE

# Create required directories
mkdir -p logs
mkdir -p data
mkdir -p backups

log "${GREEN}✓ Dependencies installed successfully${NC}"

log "${GREEN}Step 3: Configuring environment...${NC}"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  log "${YELLOW}Creating .env file...${NC}"
  
  # Generate a random secret for session
  SESSION_SECRET=$(openssl rand -hex 32)
  
cat > .env << EOF
PORT=5000
NODE_ENV=production
SESSION_SECRET=${SESSION_SECRET}
DATABASE_URL=postgres://postgres:postgres@localhost:5432/seren
USE_OLLAMA=false
EOF

  log "${GREEN}✓ .env file created${NC}"
else
  log "${GREEN}✓ .env file already exists${NC}"
fi

# Configure PostgreSQL database
log "${GREEN}Step 4: Setting up database...${NC}"

log "${YELLOW}Attempting to create PostgreSQL database and user...${NC}"

# Create user and database if they don't exist
sudo -u postgres psql -c "SELECT 1 FROM pg_user WHERE usename = 'postgres'" | grep -q 1 || \
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"

sudo -u postgres psql -c "SELECT 1 FROM pg_database WHERE datname = 'seren'" | grep -q 1 || \
sudo -u postgres psql -c "CREATE DATABASE seren OWNER postgres;"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE seren TO postgres;"

log "${GREEN}✓ Database setup completed${NC}"

# Run database migrations
log "${GREEN}Step 5: Running database migrations...${NC}"
NODE_ENV=production npm run db:push >> $LOG_FILE 2>&1 || {
  log "${RED}Failed to run database migrations. See ${LOG_FILE} for details.${NC}"
  exit 1
}

log "${GREEN}✓ Database migrations completed${NC}"

# Set up systemd service
log "${GREEN}Step 6: Creating systemd service...${NC}"

# Create systemd service file
cat > seren.service << EOF
[Unit]
Description=Seren AI - Offline AI Development Platform
After=network.target postgresql.service

[Service]
ExecStart=$(which node) $(pwd)/server/index.js
WorkingDirectory=$(pwd)
Restart=always
User=$(whoami)
Environment=NODE_ENV=production
Environment=PORT=5000

[Install]
WantedBy=multi-user.target
EOF

log "${GREEN}✓ Systemd service file created${NC}"
log "${YELLOW}Run 'sudo cp seren.service /etc/systemd/system/' to install the service${NC}"

# Create helper scripts directory if it doesn't exist
mkdir -p scripts

# Create backup script
cat > scripts/backup.sh << EOF
#!/bin/bash
# Backup script for Seren AI

DATE=\$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="./backups"

mkdir -p \$BACKUP_DIR

# Backup database
echo "Backing up database..."
pg_dump -U postgres seren > \$BACKUP_DIR/seren-db-\$DATE.sql

# Backup configuration
echo "Backing up configuration..."
cp .env \$BACKUP_DIR/env-\$DATE.backup

echo "Backup completed: \$BACKUP_DIR/seren-db-\$DATE.sql"
EOF

chmod +x scripts/backup.sh

# Create database setup script
cat > scripts/setup-database.sh << EOF
#!/bin/bash
# PostgreSQL setup script for Seren AI

# Create user and database if they don't exist
sudo -u postgres psql -c "SELECT 1 FROM pg_user WHERE usename = 'postgres'" | grep -q 1 || \\
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"

sudo -u postgres psql -c "SELECT 1 FROM pg_database WHERE datname = 'seren'" | grep -q 1 || \\
sudo -u postgres psql -c "CREATE DATABASE seren OWNER postgres;"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE seren TO postgres;"

echo "Database setup completed"
EOF

chmod +x scripts/setup-database.sh

# Create monitoring script
cat > scripts/monitor.sh << EOF
#!/bin/bash
# Monitoring script for Seren AI

watch -n 5 "ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | grep -E 'node|python' | grep -v grep"
EOF

chmod +x scripts/monitor.sh

log "${GREEN}Step 7: Final preparations...${NC}"

# Create directory for model data
mkdir -p data/models

# Create docs directory if it doesn't exist
mkdir -p docs

# Create Nginx configuration example
mkdir -p docs
cat > docs/nginx-config.conf << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Configure some basic security steps
chmod 750 data
chmod 750 backups
chmod 640 .env 2>/dev/null || true

log ""
log "============================================================"
log "${GREEN}Seren AI installation completed successfully!${NC}"
log "============================================================"
log ""
log "To start Seren AI, run: ${YELLOW}npm run start:prod${NC}"
log "Or use systemd: ${YELLOW}sudo systemctl start seren${NC} (after installing the service)"
log ""
log "Documentation:"
log "  - README-PRODUCTION.md - Main production documentation"
log "  - docs/ - Additional documentation and configuration examples"
log ""
log "For security hardening, run: ${YELLOW}./security-hardening.sh${NC}"
log ""
log "Thank you for choosing Seren AI!"
log "============================================================"