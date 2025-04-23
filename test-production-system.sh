#!/bin/bash
# Production System Test Script for Seren AI System
# This script tests all components of the Seren system in a production environment

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

# Print error message but continue
warning() {
  echo -e "${RED}! $1${RESET}"
}

# Print error message and exit
error() {
  echo -e "${RED}✗ $1${RESET}"
  exit 1
}

# Test function with results tracking
test_component() {
  local component="$1"
  local test_function="$2"
  local retry_count=0
  local max_retries=3
  
  echo -e "\n${YELLOW}Testing: ${BOLD}$component${RESET}"
  
  while [ $retry_count -lt $max_retries ]; do
    if $test_function; then
      success "$component - PASSED"
      return 0
    else
      retry_count=$((retry_count + 1))
      if [ $retry_count -lt $max_retries ]; then
        warning "$component - FAILED (Retry $retry_count/$max_retries)"
        sleep 2
      else
        warning "$component - FAILED after $max_retries attempts"
        failed_tests+=("$component")
        return 1
      fi
    fi
  done
}

# Test database connection
test_database() {
  info "Testing database connection..."
  
  # Source the .env file to get database connection details
  if [ -f ".env" ]; then
    source .env
  fi
  
  # Check if database URL is configured
  if [ -z "$DATABASE_URL" ] || [ "$DATABASE_URL" = "postgres://username:password@localhost:5432/seren" ]; then
    warning "DATABASE_URL not configured properly in .env"
    return 1
  fi
  
  # Try to connect to the database
  if command -v psql >/dev/null 2>&1; then
    # Extract connection details from DATABASE_URL
    DB_USER=$(echo "$DATABASE_URL" | sed -n 's/^postgres:\/\/\([^:]*\).*/\1/p')
    DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/^postgres:\/\/[^:]*:[^@]*@\([^:]*\).*/\1/p')
    DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/^postgres:\/\/[^:]*:[^@]*@[^:]*:\([0-9]*\).*/\1/p')
    DB_NAME=$(echo "$DATABASE_URL" | sed -n 's/^postgres:\/\/[^:]*:[^@]*@[^:]*:[0-9]*\/\([^?]*\).*/\1/p')
    
    # Check if we could parse the URL
    if [ -z "$DB_USER" ] || [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ] || [ -z "$DB_NAME" ]; then
      warning "Could not parse DATABASE_URL"
      return 1
    fi
    
    # Try to connect with psql
    if PGPASSWORD=$(echo "$DATABASE_URL" | sed -n 's/^postgres:\/\/[^:]*:\([^@]*\).*/\1/p') \
       psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" >/dev/null 2>&1; then
      info "Successfully connected to PostgreSQL database"
      return 0
    else
      warning "Failed to connect to PostgreSQL database"
      return 1
    fi
  else
    # If psql is not installed, try using Node.js to connect
    if command -v node >/dev/null 2>&1; then
      # Create a temporary Node.js script to test the connection
      cat > test_db_connection.js << EOL
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

pool.query('SELECT 1')
  .then(() => {
    console.log('Database connection successful');
    process.exit(0);
  })
  .catch(err => {
    console.error('Database connection failed:', err.message);
    process.exit(1);
  });
EOL
      
      # Run the Node.js script
      if node test_db_connection.js >/dev/null 2>&1; then
        rm test_db_connection.js
        info "Successfully connected to PostgreSQL database using Node.js"
        return 0
      else
        rm test_db_connection.js
        warning "Failed to connect to PostgreSQL database using Node.js"
        return 1
      fi
    else
      warning "Neither psql nor Node.js is available to test the database connection"
      return 1
    fi
  fi
}

# Test web server
test_web_server() {
  info "Testing web server..."
  
  # Source the .env file to get server details
  if [ -f ".env" ]; then
    source .env
  fi
  
  # Get server port
  SERVER_PORT=${PORT:-3000}
  
  # Check if server is running
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/; then
    info "Web server is running on port $SERVER_PORT"
    return 0
  else
    warning "Web server is not running on port $SERVER_PORT"
    return 1
  fi
}

# Test WebSocket connection
test_websocket() {
  info "Testing WebSocket connection..."
  
  # Source the .env file to get server details
  if [ -f ".env" ]; then
    source .env
  fi
  
  # Get server port
  SERVER_PORT=${PORT:-3000}
  
  # Create a temporary WebSocket test script
  cat > test_websocket.js << EOL
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:${SERVER_PORT}/ws');

ws.on('open', () => {
  console.log('WebSocket connection established');
  setTimeout(() => {
    ws.close();
    process.exit(0);
  }, 1000);
});

ws.on('error', (error) => {
  console.error('WebSocket connection failed:', error.message);
  process.exit(1);
});

// Set a timeout in case the connection hangs
setTimeout(() => {
  console.error('WebSocket connection timed out');
  process.exit(1);
}, 5000);
EOL
  
  # Run the WebSocket test script
  if node test_websocket.js >/dev/null 2>&1; then
    rm test_websocket.js
    info "WebSocket connection successful"
    return 0
  else
    rm test_websocket.js
    warning "WebSocket connection failed"
    return 1
  fi
}

# Test AI model integration
test_ai_integration() {
  info "Testing AI model integration..."
  
  # Create a temporary test script
  cat > test_ai_integration.js << EOL
const { generateDirectResponse, ModelType, DevTeamRole } = require('./server/ai/direct-integration');

async function testAIIntegration() {
  try {
    console.log('Testing AI model integration...');
    
    const prompt = 'Generate a simple hello world program in JavaScript';
    
    console.log('Sending test prompt to AI model...');
    const response = await generateDirectResponse(prompt, ModelType.HYBRID);
    
    if (response && response.length > 0) {
      console.log('AI model integration test passed');
      return true;
    } else {
      console.error('AI model integration test failed: Empty response');
      return false;
    }
  } catch (error) {
    console.error('AI model integration test failed:', error.message);
    return false;
  }
}

testAIIntegration()
  .then(success => process.exit(success ? 0 : 1))
  .catch(error => {
    console.error('Unexpected error:', error);
    process.exit(1);
  });
EOL
  
  # Run the AI integration test script
  if node test_ai_integration.js >/dev/null 2>&1; then
    rm test_ai_integration.js
    info "AI model integration successful"
    return 0
  else
    rm test_ai_integration.js
    warning "AI model integration failed"
    return 1
  fi
}

# Test authentication
test_authentication() {
  info "Testing authentication system..."
  
  # Source the .env file to get server details
  if [ -f ".env" ]; then
    source .env
  fi
  
  # Get server port
  SERVER_PORT=${PORT:-3000}
  
  # Create a temporary test script
  cat > test_authentication.js << EOL
const fetch = require('node-fetch');
const { v4: uuidv4 } = require('uuid');

async function testAuthentication() {
  try {
    // Create a unique test user
    const testUser = {
      username: \`test_user_\${uuidv4().split('-')[0]}\`,
      password: \`test_pass_\${uuidv4().split('-')[0]}\`,
      email: \`test_\${uuidv4().split('-')[0]}@example.com\`
    };
    
    console.log(\`Creating test user: \${testUser.username}\`);
    
    // Register test user
    const registerResponse = await fetch(\`http://localhost:${SERVER_PORT}/api/register\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testUser)
    });
    
    if (!registerResponse.ok) {
      const error = await registerResponse.text();
      console.error(\`Registration failed: \${error}\`);
      return false;
    }
    
    console.log('Registration successful');
    
    // Login with test user
    const loginResponse = await fetch(\`http://localhost:${SERVER_PORT}/api/login\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: testUser.username,
        password: testUser.password
      })
    });
    
    if (!loginResponse.ok) {
      const error = await loginResponse.text();
      console.error(\`Login failed: \${error}\`);
      return false;
    }
    
    console.log('Login successful');
    
    // Get session cookie
    const cookies = loginResponse.headers.get('set-cookie');
    
    // Get user info
    const userResponse = await fetch(\`http://localhost:${SERVER_PORT}/api/user\`, {
      headers: { Cookie: cookies }
    });
    
    if (!userResponse.ok) {
      const error = await userResponse.text();
      console.error(\`Getting user info failed: \${error}\`);
      return false;
    }
    
    const userData = await userResponse.json();
    console.log(\`Got user info: \${userData.username}\`);
    
    // Logout
    const logoutResponse = await fetch(\`http://localhost:${SERVER_PORT}/api/logout\`, {
      method: 'POST',
      headers: { Cookie: cookies }
    });
    
    if (!logoutResponse.ok) {
      const error = await logoutResponse.text();
      console.error(\`Logout failed: \${error}\`);
      return false;
    }
    
    console.log('Logout successful');
    
    // Authentication test passed
    return true;
  } catch (error) {
    console.error('Authentication test failed:', error.message);
    return false;
  }
}

testAuthentication()
  .then(success => process.exit(success ? 0 : 1))
  .catch(error => {
    console.error('Unexpected error:', error);
    process.exit(1);
  });
EOL
  
  # Run the authentication test script
  if node test_authentication.js >/dev/null 2>&1; then
    rm test_authentication.js
    info "Authentication system working correctly"
    return 0
  else
    rm test_authentication.js
    warning "Authentication system test failed"
    return 1
  fi
}

# Check security configuration
test_security() {
  info "Testing security configuration..."
  
  # Source the .env file to get server details
  if [ -f ".env" ]; then
    source .env
  fi
  
  # Get server port
  SERVER_PORT=${PORT:-3000}
  
  # Check for important security headers
  local security_headers=("X-Content-Type-Options" "X-Frame-Options" "X-XSS-Protection" "Content-Security-Policy")
  local missing_headers=()
  
  for header in "${security_headers[@]}"; do
    if ! curl -s -I "http://localhost:$SERVER_PORT/" | grep -i "$header" >/dev/null; then
      missing_headers+=("$header")
    fi
  done
  
  if [ ${#missing_headers[@]} -eq 0 ]; then
    info "All security headers are properly configured"
  else
    for missing in "${missing_headers[@]}"; do
      warning "Missing security header: $missing"
    done
  fi
  
  # Check session configuration
  if [ -z "$SESSION_SECRET" ] || [ "$SESSION_SECRET" = "replace-with-strong-secret-in-production" ]; then
    warning "SESSION_SECRET is not properly configured"
    return 1
  fi
  
  # Check JWT configuration
  if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "replace-with-strong-secret-in-production" ]; then
    warning "JWT_SECRET is not properly configured"
    return 1
  fi
  
  # If we got here with no errors, security config is good
  info "Security configuration passed basic checks"
  return 0
}

# Test system resources
test_system_resources() {
  info "Testing system resources..."
  
  # Check CPU cores
  CPU_CORES=$(nproc)
  if [ "$CPU_CORES" -lt 2 ]; then
    warning "CPU: Only $CPU_CORES core(s) available. Minimum recommended is 2 cores."
  else
    info "CPU: $CPU_CORES cores available."
  fi
  
  # Check available memory
  if command -v free >/dev/null 2>&1; then
    MEM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')
    MEM_AVAIL=$(free -m | awk '/^Mem:/{print $7}')
    
    if [ "$MEM_TOTAL" -lt 3072 ]; then
      warning "Memory: Only $MEM_TOTAL MB total. Minimum recommended is 4096 MB."
    else
      info "Memory: $MEM_TOTAL MB total memory available."
    fi
    
    if [ "$MEM_AVAIL" -lt 1024 ]; then
      warning "Memory: Only $MEM_AVAIL MB available memory. Minimum recommended available is 1024 MB."
    else
      info "Memory: $MEM_AVAIL MB available memory."
    fi
  else
    warning "Could not check memory, 'free' command not available."
  fi
  
  # Check disk space
  if command -v df >/dev/null 2>&1; then
    DISK_AVAIL=$(df -m . | awk 'NR==2 {print $4}')
    
    if [ "$DISK_AVAIL" -lt 10240 ]; then
      warning "Disk: Only $DISK_AVAIL MB available. Minimum recommended is 10240 MB (10 GB)."
    else
      info "Disk: $DISK_AVAIL MB available."
    fi
  else
    warning "Could not check disk space, 'df' command not available."
  fi
  
  # Check Node.js version
  if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node -v | cut -d 'v' -f 2)
    NODE_MAJOR_VERSION=$(echo "$NODE_VERSION" | cut -d '.' -f 1)
    
    if [ "$NODE_MAJOR_VERSION" -lt 18 ]; then
      warning "Node.js: Version $NODE_VERSION. Minimum recommended is 18.x."
    else
      info "Node.js: Version $NODE_VERSION."
    fi
  else
    warning "Node.js not found, cannot check version."
    return 1
  fi
  
  # Basic resource requirements met, return success
  return 0
}

# Main function
main() {
  section "Seren AI System - Production Test Suite"
  info "Running comprehensive tests on all system components"
  
  # Initialize array for failed tests
  failed_tests=()
  
  # Run the tests
  test_component "System Resources" test_system_resources
  test_component "Database Connection" test_database
  test_component "Web Server" test_web_server
  test_component "WebSocket Connection" test_websocket
  test_component "AI Model Integration" test_ai_integration
  test_component "Authentication System" test_authentication
  test_component "Security Configuration" test_security
  
  # Display test summary
  section "Test Summary"
  
  if [ ${#failed_tests[@]} -eq 0 ]; then
    success "All tests passed! The Seren AI System is ready for production."
    
    # Provide deployment instructions
    section "Next Steps"
    
    cat << EOL
Congratulations! Your Seren AI System is configured correctly and ready for production deployment.
Here are the next steps to complete the deployment:

1. Run the production setup script:
   ./setup-production.sh

2. Create the production deployment package:
   ./create-deployment-package.sh

3. Upload the deployment package to your VDS:
   scp seren-production-*.tar.gz user@your-vds-host:/tmp/

4. Extract and configure the deployment package on your VDS:
   tar -xzf seren-production-*.tar.gz -C /opt/seren
   cd /opt/seren
   cp .env.example .env
   nano .env  # Update with your actual configuration

5. Apply security hardening:
   ./security-hardening.sh

6. Start the application:
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup

For more information, refer to the README-PRODUCTION.md file.
EOL
    
    return 0
  else
    section "Failed Tests"
    
    echo "The following tests failed:"
    for test in "${failed_tests[@]}"; do
      echo -e "${RED}✗ $test${RESET}"
    done
    
    # Display troubleshooting instructions
    section "Troubleshooting"
    
    cat << EOL
Please fix the failed tests before deploying to production. Here are some troubleshooting tips:

1. Database Connection:
   - Check the DATABASE_URL in your .env file
   - Ensure PostgreSQL is running and accessible
   - Verify the database user has the correct permissions

2. Web Server:
   - Check if the server is running with \`npm run dev\` or \`pm2 status\`
   - Verify the correct port is being used
   - Check server logs for errors

3. WebSocket Connection:
   - Ensure the WebSocket server is configured correctly
   - Check for firewall or proxy issues that might block WebSocket connections

4. AI Model Integration:
   - Verify the Direct Integration module is properly configured
   - Check server logs for errors related to AI model initialization

5. Authentication System:
   - Check if the database tables for users exist
   - Verify session and JWT secrets are properly configured

6. Security Configuration:
   - Update the SESSION_SECRET and JWT_SECRET in your .env file
   - Configure security headers in your Express application

7. System Resources:
   - Ensure the VDS has sufficient CPU, memory, and disk resources

For more information, refer to the README-PRODUCTION.md file.
EOL
    
    return 1
  fi
}

# Run the main function
main