#!/bin/bash
# Security Hardening Script for Seren AI System
# This script applies security best practices to a VDS running Seren

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

# Check if running as root
check_root() {
  if [ "$(id -u)" != "0" ]; then
    error "This script must be run as root. Try 'sudo $0'"
  fi
}

# Main security hardening function
main() {
  check_root
  
  section "Seren AI System - Security Hardening"
  info "Applying security best practices to your VDS"
  
  # System updates
  section "System Updates"
  info "Updating system packages..."
  
  apt update
  apt upgrade -y
  
  success "System packages updated"
  
  # Firewall configuration
  section "Firewall Configuration"
  
  if ! command -v ufw >/dev/null 2>&1; then
    info "Installing UFW (Uncomplicated Firewall)..."
    apt install -y ufw
  fi
  
  info "Configuring firewall..."
  
  # Reset UFW to default
  ufw --force reset
  
  # Default policies
  ufw default deny incoming
  ufw default allow outgoing
  
  # Allow SSH, HTTP, HTTPS
  ufw allow ssh
  ufw allow http
  ufw allow https
  
  # Enable firewall
  ufw --force enable
  
  success "Firewall configured and enabled"
  
  # Secure SSH
  section "SSH Hardening"
  
  info "Securing SSH configuration..."
  
  # Backup original SSH config
  cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
  
  # Apply more secure SSH settings
  cat > /etc/ssh/sshd_config << EOL
# Seren AI System - Hardened SSH Configuration

# Basic settings
Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key

# Authentication restrictions
LoginGraceTime 30
PermitRootLogin no
StrictModes yes
MaxAuthTries 3
MaxSessions 5

# Authentication
PubkeyAuthentication yes
PasswordAuthentication yes  # Consider changing to 'no' after setting up key-based auth
PermitEmptyPasswords no
ChallengeResponseAuthentication no

# Features
X11Forwarding no
PrintMotd no
UsePAM yes

# Timeout settings
ClientAliveInterval 300
ClientAliveCountMax 2

# Disable obsolete features
UsePrivilegeSeparation sandbox
UseDNS no
AllowTcpForwarding no
PermitTunnel no

# Ensure access to SSH is limited to SSH group
AllowGroups sudo ssh

# Accepted environment variables
AcceptEnv LANG LC_*

# Subsystem settings
Subsystem sftp /usr/lib/openssh/sftp-server
EOL
  
  # Create SSH group if it doesn't exist
  if ! getent group ssh > /dev/null; then
    groupadd ssh
  fi
  
  # Add current user to SSH group
  if [ -n "$SUDO_USER" ]; then
    usermod -a -G ssh "$SUDO_USER"
  fi
  
  # Restart SSH service
  systemctl restart sshd
  
  success "SSH hardened"
  
  # Fail2Ban installation
  section "Fail2Ban Installation"
  
  if ! command -v fail2ban-server >/dev/null 2>&1; then
    info "Installing Fail2Ban..."
    apt install -y fail2ban
  else
    info "Fail2Ban already installed"
  fi
  
  # Configure Fail2Ban
  cat > /etc/fail2ban/jail.local << EOL
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
banaction = ufw

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 5
EOL
  
  # Restart Fail2Ban
  systemctl enable fail2ban
  systemctl restart fail2ban
  
  success "Fail2Ban installed and configured"
  
  # Set up automatic security updates
  section "Automatic Security Updates"
  
  info "Setting up automatic security updates..."
  
  apt install -y unattended-upgrades apt-listchanges
  
  cat > /etc/apt/apt.conf.d/20auto-upgrades << EOL
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOL
  
  cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOL
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}";
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};

Unattended-Upgrade::Package-Blacklist {
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::InstallOnShutdown "false";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOL
  
  # Enable the service
  systemctl restart unattended-upgrades
  
  success "Automatic security updates configured"
  
  # Set up logrotate for application logs
  section "Log Rotation"
  
  info "Setting up log rotation for application logs..."
  
  cat > /etc/logrotate.d/seren << EOL
/opt/seren/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 node node
    sharedscripts
    postrotate
        systemctl reload nginx
    endscript
}
EOL
  
  success "Log rotation configured"
  
  # Hardening system parameters
  section "System Hardening"
  
  info "Hardening system parameters..."
  
  # Backup sysctl.conf
  cp /etc/sysctl.conf /etc/sysctl.conf.bak
  
  # Add security parameters
  cat >> /etc/sysctl.conf << EOL

# Seren AI System - Security Hardening

# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Block SYN attacks
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore Directed pings
net.ipv4.icmp_echo_ignore_all = 0

# Increase system file descriptor limit
fs.file-max = 65535

# Protect against kernel vulnerabilities
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
EOL
  
  # Apply sysctl settings
  sysctl -p
  
  success "System parameters hardened"
  
  # Secure shared memory
  info "Securing shared memory..."
  
  if ! grep -q '/run/shm' /etc/fstab; then
    echo "tmpfs     /run/shm     tmpfs     defaults,noexec,nosuid     0     0" >> /etc/fstab
    mount -o remount /run/shm
  fi
  
  success "Shared memory secured"
  
  # Harden networking
  info "Hardening network configuration..."
  
  # Ensure IPv6 is properly configured if enabled
  if [ "$(sysctl -n net.ipv6.conf.all.disable_ipv6)" = "0" ]; then
    cat >> /etc/sysctl.conf << EOL
# IPv6 Security Settings
net.ipv6.conf.all.accept_ra = 0
net.ipv6.conf.default.accept_ra = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
EOL
    sysctl -p
  fi
  
  success "Network configuration hardened"
  
  # Secure nginx
  section "Nginx Hardening"
  
  if command -v nginx >/dev/null 2>&1; then
    info "Hardening Nginx configuration..."
    
    # Check for Nginx security config file
    if [ ! -f /etc/nginx/conf.d/security.conf ]; then
      cat > /etc/nginx/conf.d/security.conf << EOL
# Security Settings for Nginx

# Hide server information
server_tokens off;

# Security headers
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options SAMEORIGIN;
add_header X-XSS-Protection "1; mode=block";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self' wss: ws:";
add_header Referrer-Policy no-referrer-when-downgrade;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

# Slow client protection
client_body_timeout 10s;
client_header_timeout 10s;
keepalive_timeout 65s;
send_timeout 10s;
client_max_body_size 20m;

# Buffer sizes
client_body_buffer_size 128k;
client_header_buffer_size 1k;
large_client_header_buffers 4 8k;

# Limits
limit_conn_zone \$binary_remote_addr zone=conn_limit_per_ip:10m;
limit_req_zone \$binary_remote_addr zone=req_limit_per_ip:10m rate=5r/s;
EOL
      
      # Test and reload Nginx
      nginx -t && systemctl reload nginx
      success "Nginx security configuration added"
    else
      info "Nginx security configuration already exists"
    fi
  else
    info "Nginx not installed, skipping Nginx hardening"
  fi
  
  # Secure Node.js application
  section "Node.js Application Hardening"
  
  info "Ensuring Node.js application runs with limited privileges..."
  
  # Create a dedicated user for the application if it doesn't exist
  if ! id -u seren >/dev/null 2>&1; then
    useradd -m -s /bin/bash seren
    success "Created dedicated user 'seren' for the application"
  else
    info "User 'seren' already exists"
  fi
  
  # Create PM2 systemd service file
  if command -v pm2 >/dev/null 2>&1; then
    info "Creating systemd service for PM2..."
    
    # Generate PM2 startup script
    sudo -u seren bash -c 'PM2_HOME=/home/seren/.pm2 pm2 startup systemd -u seren --hp /home/seren'
    
    success "PM2 systemd service created"
  fi
  
  # Create application directory with proper permissions
  if [ ! -d /opt/seren ]; then
    mkdir -p /opt/seren
    chown seren:seren /opt/seren
    chmod 750 /opt/seren
    success "Created application directory with proper permissions"
  fi
  
  # Final recommendations
  section "Security Hardening Complete"
  
  cat << EOL
The following security measures have been applied to your VDS:

1. System packages updated
2. Firewall (UFW) configured to allow only SSH, HTTP, and HTTPS
3. SSH hardened with secure configuration
4. Fail2Ban installed to protect against brute force attacks
5. Automatic security updates configured
6. Log rotation set up for application logs
7. System parameters hardened against common vulnerabilities
8. Shared memory secured
9. Network settings hardened
10. Nginx configured with security best practices (if installed)
11. Node.js application set up to run with limited privileges

Additional recommendations:

1. Consider setting up key-based SSH authentication and disabling password authentication
2. Implement regular database backups
3. Set up monitoring and alerting
4. Perform regular security audits
5. Keep all software up to date

Your VDS is now configured with security best practices for running the Seren AI System.
EOL
}

# Run the main function
main