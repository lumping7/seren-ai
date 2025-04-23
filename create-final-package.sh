#!/bin/bash

# Seren AI - Final Deployment Package Creator
# This script creates a compressed archive ready for production deployment

set -e

echo "============================================================"
echo "  Seren AI - Final Deployment Package"
echo "  Creating production-ready archive"
echo "============================================================"
echo ""

# Define output filename
PACKAGE_NAME="seren-ai-$(date +%Y%m%d).tar.gz"
TEMP_DIR="./temp-package"

# Prepare temporary directory
rm -rf $TEMP_DIR 2>/dev/null || true
mkdir -p $TEMP_DIR
mkdir -p $TEMP_DIR/scripts
mkdir -p $TEMP_DIR/docs
mkdir -p $TEMP_DIR/logs
mkdir -p $TEMP_DIR/data
mkdir -p $TEMP_DIR/backups

# Copy essential files
echo "Copying project files..."
cp -r ai_core $TEMP_DIR/
cp -r client $TEMP_DIR/
cp -r server $TEMP_DIR/
cp -r shared $TEMP_DIR/
cp -r scripts/* $TEMP_DIR/scripts/ 2>/dev/null || true

# Copy configuration files
cp package.json $TEMP_DIR/
cp package-lock.json $TEMP_DIR/
cp tsconfig.json $TEMP_DIR/
cp README.md $TEMP_DIR/
cp -r docs/* $TEMP_DIR/docs/ 2>/dev/null || true

# Copy deployment scripts
cp setup-production.sh $TEMP_DIR/
cp security-hardening.sh $TEMP_DIR/
cp README-PRODUCTION.md $TEMP_DIR/

# Create example .env file
cat > $TEMP_DIR/.env.example << EOF
PORT=5000
NODE_ENV=production
SESSION_SECRET=your_secret_key_here
DATABASE_URL=postgres://postgres:postgres@localhost:5432/seren
USE_OLLAMA=false
OLLAMA_HOST=http://localhost:11434
EOF

# Create the compressed archive
echo "Creating compressed archive..."
tar -czf $PACKAGE_NAME -C $TEMP_DIR .

# Clean up
rm -rf $TEMP_DIR

echo ""
echo "============================================================"
echo "  Deployment package created: $PACKAGE_NAME"
echo "============================================================"
echo ""
echo "This archive contains everything needed for production deployment."
echo "Follow these steps to deploy:"
echo ""
echo "1. Copy the package to your VDS server"
echo "2. Extract the archive: tar -xzf $PACKAGE_NAME"
echo "3. Run the setup script: ./setup-production.sh"
echo "4. Apply security hardening: sudo ./security-hardening.sh"
echo "5. Start the application: ./scripts/start-prod.sh"
echo ""
echo "For detailed documentation, see README-PRODUCTION.md"
echo "============================================================"