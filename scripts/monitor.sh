#!/bin/bash

# Seren AI - Monitoring Script
# This script monitors system resources used by Seren AI

echo "============================================================"
echo "  Seren AI - System Monitoring"
echo "  Press Ctrl+C to exit"
echo "============================================================"
echo ""

# Check if watch is installed
if ! command -v watch &> /dev/null; then
    echo "The 'watch' command is not installed. Falling back to a basic monitor."
    
    while true; do
        clear
        echo "Monitoring Seren AI processes..."
        echo "Time: $(date)"
        echo ""
        ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | grep -E 'node|python' | grep -v grep
        echo ""
        echo "Press Ctrl+C to exit"
        sleep 5
    done
else
    # Use watch for better UI
    watch -n 5 "ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | grep -E 'node|python' | grep -v grep"
fi