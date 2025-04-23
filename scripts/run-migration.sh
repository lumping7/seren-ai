#!/bin/bash

# Make the script exit on any error
set -e

echo "Running database migration..."

# Run the migration script
npx tsx scripts/migration.ts

echo "Migration completed successfully!"