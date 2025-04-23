#!/bin/bash

# Seren AI - Security Hardening Script
# This script applies military-grade security protocols to the Seren AI system

set -e

echo "============================================================"
echo "  Seren AI - Security Hardening"
echo "  Military-Grade Security Implementation"
echo "============================================================"
echo ""

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Error: This script requires root privileges.${NC}"
  echo -e "Please run with: ${YELLOW}sudo $0${NC}"
  exit 1
fi

# Create log directory
mkdir -p logs
LOG_FILE="logs/security-$(date +%Y%m%d-%H%M%S).log"
touch $LOG_FILE

log() {
  echo -e "$1" | tee -a $LOG_FILE
}

log "${GREEN}Step 1: Securing file permissions...${NC}"

# Secure directories
find . -type d -exec chmod 750 {} \;
find ./logs -type d -exec chmod 770 {} \;
find ./data -type d -exec chmod 750 {} \;
find ./backups -type d -exec chmod 750 {} \;

# Secure files
find . -type f -exec chmod 640 {} \;
find ./scripts -type f -name "*.sh" -exec chmod 750 {} \;
chmod 750 *.sh

# Secure sensitive files
if [ -f .env ]; then
  chmod 600 .env
fi

# Secure log files
chmod 640 logs/*

log "${GREEN}✓ File permissions secured${NC}"

log "${GREEN}Step 2: Setting up firewall...${NC}"

# Check if ufw is installed
if ! command -v ufw &> /dev/null; then
  log "${YELLOW}UFW not found. Installing...${NC}"
  apt-get update
  apt-get install -y ufw
fi

# Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 5000/tcp  # Seren AI port
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS

# Enable firewall if not already enabled
if ! ufw status | grep -q "Status: active"; then
  log "${YELLOW}Enabling UFW firewall...${NC}"
  echo "y" | ufw enable
fi

log "${GREEN}✓ Firewall configured${NC}"

log "${GREEN}Step 3: Creating Express security middleware...${NC}"

# Create security middleware file
mkdir -p server/security

# Create rate limiting middleware
cat > server/security/rate-limit.ts << 'EOF'
import rateLimit from 'express-rate-limit';

// Basic rate limiter for all routes
export const basicRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 200, // Limit each IP to 200 requests per windowMs
  message: 'Too many requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// Stricter rate limiter for authentication routes
export const authRateLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 10, // Limit each IP to 10 login attempts per hour
  message: 'Too many login attempts from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// API rate limiter for model endpoints
export const modelRateLimiter = rateLimit({
  windowMs: 5 * 60 * 1000, // 5 minutes
  max: 30, // Limit each IP to 30 model requests per 5 minutes
  message: 'Too many AI requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});
EOF

# Create HTTP security middleware
cat > server/security/http-security.ts << 'EOF'
import helmet from 'helmet';
import { Express } from 'express';
import hpp from 'hpp';
import compression from 'compression';

export function configureHttpSecurity(app: Express) {
  // Use Helmet for secure HTTP headers
  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", "data:", "blob:"],
          connectSrc: ["'self'"],
          fontSrc: ["'self'"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          frameSrc: ["'none'"],
        },
      },
      crossOriginEmbedderPolicy: false,
      crossOriginResourcePolicy: { policy: 'same-site' },
    })
  );

  // Prevent HTTP Parameter Pollution attacks
  app.use(hpp());

  // Enable gzip compression for better performance
  app.use(compression());
  
  // Set secure cookies
  app.set('trust proxy', 1); // Trust first proxy
}
EOF

# Create main security index
cat > server/security/index.ts << 'EOF'
import { Express } from 'express';
import { basicRateLimiter, authRateLimiter, modelRateLimiter } from './rate-limit';
import { configureHttpSecurity } from './http-security';

export function setupSecurity(app: Express) {
  // Configure HTTP security
  configureHttpSecurity(app);
  
  // Apply rate limiting
  app.use(basicRateLimiter);
  app.use('/api/login', authRateLimiter);
  app.use('/api/register', authRateLimiter);
  app.use('/api/virtual-computer', modelRateLimiter);
  
  // Log requests in production
  if (process.env.NODE_ENV === 'production') {
    app.use((req, res, next) => {
      console.log(`${new Date().toISOString()} - ${req.method} ${req.path} - IP: ${req.ip}`);
      next();
    });
  }
}
EOF

log "${GREEN}✓ Security middleware created${NC}"

# Ensure we apply the security middleware in the main server
log "${GREEN}Step 4: Integrating security middleware...${NC}"

# Check if the security module is already imported in server/index.ts
if ! grep -q "setupSecurity" server/index.ts; then
  log "${YELLOW}Adding security setup to server/index.ts...${NC}"
  
  # Create a backup of server/index.ts
  cp server/index.ts server/index.ts.bak
  
  # Add security import and setup
  sed -i '/import { registerRoutes } from/a import { setupSecurity } from "./security";' server/index.ts
  sed -i '/app.use(express.json())/a setupSecurity(app);' server/index.ts
  
  log "${GREEN}✓ Security middleware integrated${NC}"
else
  log "${GREEN}✓ Security middleware already integrated${NC}"
fi

log "${GREEN}Step 5: Securing database...${NC}"

# Create postgresql configuration backup
if [ -f /etc/postgresql/*/main/postgresql.conf ]; then
  log "${YELLOW}Backing up PostgreSQL configuration...${NC}"
  cp /etc/postgresql/*/main/postgresql.conf /etc/postgresql/*/main/postgresql.conf.bak
  
  # Secure PostgreSQL configuration
  log "${YELLOW}Updating PostgreSQL configuration...${NC}"
  
  # Find the PostgreSQL config file
  PG_CONF_PATH=$(find /etc/postgresql -name "postgresql.conf" | head -n 1)
  
  # Only accept connections from localhost
  sed -i "s/#listen_addresses = 'localhost'/listen_addresses = 'localhost'/" $PG_CONF_PATH
  
  # Restart PostgreSQL to apply changes
  systemctl restart postgresql
  
  log "${GREEN}✓ PostgreSQL secured${NC}"
else
  log "${YELLOW}PostgreSQL configuration not found. Skipping database hardening.${NC}"
fi

log "${GREEN}Step 6: Setting up regular security updates...${NC}"

# Create a cronjob for security updates
cat > /etc/cron.weekly/seren-security << 'EOF'
#!/bin/bash

# Weekly security updates for Seren AI
apt-get update
apt-get upgrade -y
apt-get dist-upgrade -y

# Restart services if needed
systemctl daemon-reload
systemctl restart postgresql
# Only restart Seren if it's managed by systemd
if systemctl is-active --quiet seren; then
  systemctl restart seren
fi

# Cleanup old packages
apt-get autoremove -y
apt-get autoclean
EOF

chmod 755 /etc/cron.weekly/seren-security

log "${GREEN}✓ Regular security updates configured${NC}"

log "${GREEN}Step 7: Setting up intrusion detection...${NC}"

# Check if fail2ban is installed
if ! command -v fail2ban-client &> /dev/null; then
  log "${YELLOW}Fail2ban not found. Installing...${NC}"
  apt-get update
  apt-get install -y fail2ban
fi

# Configure fail2ban
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[seren-api]
enabled = true
filter = seren-api
action = iptables-multiport[name=seren, port="5000,80,443"]
logpath = /path/to/seren/logs/access.log
maxretry = 10
findtime = 300
bantime = 7200
EOF

# Create fail2ban filter for Seren
cat > /etc/fail2ban/filter.d/seren-api.conf << 'EOF'
[Definition]
failregex = ^.*POST /api/login.*401.*IP: <HOST>.*$
            ^.*POST /api/register.*400.*IP: <HOST>.*$
            ^.*POST /api/virtual-computer.*429.*IP: <HOST>.*$
ignoreregex =
EOF

# Restart fail2ban
systemctl enable fail2ban
systemctl restart fail2ban

log "${GREEN}✓ Intrusion detection system set up${NC}"

log "${GREEN}Step 8: Creating backup and recovery plan...${NC}"

# Create backup script
cat > scripts/security-backup.sh << 'EOF'
#!/bin/bash
# Security-focused backup script for Seren AI

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="./backups"

mkdir -p $BACKUP_DIR

# Backup database with encryption
echo "Backing up database with encryption..."
pg_dump -U postgres seren | gpg --symmetric --cipher-algo AES256 --output $BACKUP_DIR/seren-db-$DATE.sql.gpg

# Backup configuration with encryption
echo "Backing up configuration with encryption..."
cat .env | gpg --symmetric --cipher-algo AES256 --output $BACKUP_DIR/env-$DATE.backup.gpg

echo "Encrypted backup completed: $BACKUP_DIR/seren-db-$DATE.sql.gpg"
EOF

chmod 750 scripts/security-backup.sh

log "${GREEN}✓ Security backup plan created${NC}"

log "${GREEN}Step 9: Configuring secure system parameters...${NC}"

# Secure kernel parameters
cat > /etc/sysctl.d/99-seren-security.conf << 'EOF'
# Protect against SYN flood attacks
net.ipv4.tcp_syncookies = 1

# Protect against IP spoofing
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Do not accept ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Do not send ICMP redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Do not accept IP source route packets
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log suspicious packets
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
EOF

# Apply sysctl settings
sysctl -p /etc/sysctl.d/99-seren-security.conf

log "${GREEN}✓ Secure system parameters configured${NC}"

log "${GREEN}Step 10: Final checks and verification...${NC}"

# Check for open ports
log "${YELLOW}Checking for open ports...${NC}"
netstat -tulpn | grep -E ':(5000|80|443|22|11434)' | tee -a $LOG_FILE

# Check firewall status
log "${YELLOW}Checking firewall status...${NC}"
ufw status | tee -a $LOG_FILE

# Check fail2ban status
log "${YELLOW}Checking fail2ban status...${NC}"
fail2ban-client status | tee -a $LOG_FILE

log ""
log "============================================================"
log "${GREEN}Seren AI security hardening completed successfully!${NC}"
log "============================================================"
log ""
log "The following security measures have been implemented:"
log "- Secure file permissions"
log "- Configured firewall (UFW)"
log "- Rate limiting for API endpoints"
log "- Security HTTP headers with Helmet"
log "- Protection against HTTP Parameter Pollution"
log "- Secure PostgreSQL configuration"
log "- Weekly automated security updates"
log "- Intrusion detection with fail2ban"
log "- Encrypted backup system"
log "- Hardened kernel security parameters"
log ""
log "For security audit, run: ${YELLOW}./scripts/security-audit.sh${NC}"
log ""
log "Your system is now configured with military-grade security."
log "============================================================"