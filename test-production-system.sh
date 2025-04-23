#!/bin/bash

# Seren AI - Production Testing Script
# This script verifies that all components are working correctly in production

set -e

echo "============================================================"
echo "  Seren AI - Production Testing"
echo "  Verifying system components"
echo "============================================================"
echo ""

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the hostname and port from environment or use defaults
HOST=${HOST:-localhost}
PORT=${PORT:-5000}
API_BASE="http://${HOST}:${PORT}"

# Function to test API endpoint
test_endpoint() {
  local endpoint=$1
  local expected_status=$2
  local description=$3

  echo -n "Testing ${endpoint}... "
  
  # Make the request and store the status code
  status=$(curl -s -o /dev/null -w "%{http_code}" ${API_BASE}${endpoint})
  
  if [ "$status" -eq "$expected_status" ]; then
    echo -e "${GREEN}✓ OK${NC} (Status: $status)"
    return 0
  else
    echo -e "${RED}✗ FAILED${NC} (Expected: $expected_status, Got: $status)"
    return 1
  fi
}

# Function to test virtual computer endpoint
test_virtual_computer() {
  local model=$1
  
  echo -n "Testing virtual computer with model '$model'... "
  
  # Create a JSON payload with a test query
  payload="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello, are you working?\"}],\"operationId\":\"test-$(date +%s)\"}"
  
  # Make the request
  response=$(curl -s -X POST -H "Content-Type: application/json" -d "$payload" ${API_BASE}/api/virtual-computer)
  
  # Check if we got a valid response
  if [[ $response == *"content"* ]] && [[ $response == *"role"* ]]; then
    echo -e "${GREEN}✓ OK${NC} (Got valid response)"
    return 0
  else
    echo -e "${RED}✗ FAILED${NC} (Invalid response)"
    echo "Response: $response"
    return 1
  fi
}

# Main testing sequence
echo "Step: Verifying API endpoints"
echo "------------------------"

# Test basic endpoints
test_endpoint "/health" 200 "Health check endpoint"
test_endpoint "/api/health" 200 "API health check endpoint"

echo ""
echo "Step: Testing virtual computer"
echo "------------------------"

# Test virtual computer with each model
test_virtual_computer "qwen2.5-7b-omni"
test_virtual_computer "olympiccoder-7b"
test_virtual_computer "hybrid"

echo ""
echo "============================================================"
echo "  Testing Complete"
echo "============================================================"
echo ""
echo "If all tests passed, your Seren AI system is configured correctly."
echo "If any tests failed, please check the error messages and server logs."
echo ""