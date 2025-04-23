#!/bin/bash

# Seren AI - Backup Script
# This script creates a backup of the Seren AI system

DATE=$(date +"%Y%m%d-%H%M%S")
BACKUP_DIR="./backups"
BACKUP_FILE="$BACKUP_DIR/seren-backup-$DATE.tar.gz"

echo "============================================================"
echo "  Seren AI - Backup Utility"
echo "  Creating backup: $BACKUP_FILE"
echo "============================================================"
echo ""

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create a list of important files and directories to backup
echo "Preparing backup..."
BACKUP_LIST=(
  "./server"
  "./client"
  "./shared"
  "./ai_core"
  "./.env"
  "./package.json"
  "./package-lock.json"
  "./tsconfig.json"
  "./data"
)

# Create the backup
echo "Creating backup archive..."
tar -czf $BACKUP_FILE ${BACKUP_LIST[@]} 2>/dev/null

# Check if backup was successful
if [ $? -eq 0 ]; then
  SIZE=$(du -h $BACKUP_FILE | cut -f1)
  echo ""
  echo "✓ Backup completed successfully!"
  echo "  Backup file: $BACKUP_FILE ($SIZE)"
  echo ""
else
  echo ""
  echo "✗ Backup failed!"
  echo ""
  exit 1
fi

# Database backup
if [ -n "$DATABASE_URL" ]; then
  echo "Backing up database..."
  DB_BACKUP_FILE="$BACKUP_DIR/seren-db-backup-$DATE.sql"
  
  # Extract database connection details from DATABASE_URL
  if [[ $DATABASE_URL == postgres://* ]]; then
    # Format is: postgres://username:password@hostname:port/database
    DB_USER=$(echo $DATABASE_URL | sed -e 's/postgres:\/\///' -e 's/:.*$//')
    DB_PASS=$(echo $DATABASE_URL | sed -e 's/postgres:\/\/[^:]*://' -e 's/@.*$//')
    DB_HOST=$(echo $DATABASE_URL | sed -e 's/postgres:\/\/[^@]*@//' -e 's/:.*$//')
    DB_PORT=$(echo $DATABASE_URL | sed -e 's/postgres:\/\/[^:]*:[^:]*:[^:]*://' -e 's/\/.*$//')
    DB_NAME=$(echo $DATABASE_URL | sed -e 's/postgres:\/\/[^\/]*\///')
    
    # Check if pg_dump is available
    if command -v pg_dump &> /dev/null; then
      PGPASSWORD=$DB_PASS pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $DB_BACKUP_FILE
      
      if [ $? -eq 0 ]; then
        DB_SIZE=$(du -h $DB_BACKUP_FILE | cut -f1)
        echo "✓ Database backup completed: $DB_BACKUP_FILE ($DB_SIZE)"
      else
        echo "✗ Database backup failed!"
      fi
    else
      echo "⚠ pg_dump not found. Database backup skipped."
    fi
  else
    echo "⚠ Unsupported database type. Only PostgreSQL is supported for automated backup."
  fi
fi

echo ""
echo "Backup process completed!"
echo "============================================================"