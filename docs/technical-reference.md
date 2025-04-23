# Technical Reference

## System Architecture

This document provides a technical reference for the advanced AI collaboration system architecture, covering implementation details, data flows, and component interactions.

## Core System Components

### 1. Backend Architecture

```
server/
├── index.ts            # Application entry point
├── routes.ts           # API route definitions
├── storage.ts          # Database interface
├── auth.ts             # Authentication system
├── db.ts               # Database connection
├── vite.ts             # Vite development server config
└── ai/                 # AI system components
    ├── index.ts        # AI module entry point
    ├── llama.ts        # Llama3 model handler
    ├── gemma.ts        # Gemma3 model handler
    ├── hybrid.ts       # Hybrid collaboration engine
    ├── reasoning.ts    # Neuro-symbolic reasoning engine
    ├── memory.ts       # AI memory system
    ├── execution.ts    # Code execution environment
    ├── conversation.ts # Model-to-model conversation system
    ├── knowledge-api.ts # Knowledge system API interface
    ├── knowledge-manager.ts # Knowledge management system
    └── resource-manager.ts # Adaptive resource management
```

### 2. Database Schema

The system uses a PostgreSQL database with the following schema structure:

```
shared/schema.ts        # Shared schema definitions

Tables:
- users                 # User accounts and authentication
- ai_memories           # Long-term AI memory storage
- ai_messages           # Conversation history
- ai_settings           # System configuration
- ai_knowledge_base     # Knowledge entries
- ai_knowledge_domains  # Knowledge categorization 
- ai_knowledge_relations # Relationships between knowledge
- ai_knowledge_tags     # Tags for knowledge entries
- ai_knowledge_to_domains # Knowledge-domain mapping
- ai_knowledge_to_tags  # Knowledge-tag mapping
- ai_knowledge_feedback # User feedback on knowledge
```

### 3. Client Architecture

```
client/
├── index.html          # HTML entry point
└── src/                # Client source code
    ├── main.tsx        # Application bootstrapping
    ├── App.tsx         # Root component
    ├── index.css       # Global styles
    ├── components/     # Reusable UI components
    ├── hooks/          # React hooks
    │   ├── use-auth.tsx # Authentication hook
    │   └── use-toast.ts # Notification system
    ├── lib/            # Utility functions
    │   ├── queryClient.ts # API client setup
    │   └── websocket.ts # WebSocket connection
    └── pages/          # Application pages
```

## Data Flow Architecture

### 1. Request Processing Flow

```
Client Request → API Gateway → Model Selection → Resource Allocation → 
Processing → Knowledge Enhancement → Response Formatting → Client Response
```

1. **Client Request**: Request arrives via REST API or WebSocket
2. **API Gateway**: Request validation, authentication, rate limiting
3. **Model Selection**: Determine processing mode and model(s) to use
4. **Resource Allocation**: Check available resources and allocate appropriately
5. **Processing**: Execute AI operations (inference, reasoning, etc.)
6. **Knowledge Enhancement**: Integrate relevant knowledge where appropriate
7. **Response Formatting**: Structure output according to client requirements
8. **Client Response**: Return formatted response to client

### 2. Knowledge Flow

```
Knowledge Sources → Extraction → Validation → Storage → Indexing → 
Retrieval → Enhancement → Application
```

1. **Knowledge Sources**: User input, model outputs, conversations
2. **Extraction**: Identifying valuable information for storage
3. **Validation**: Checking accuracy, uniqueness, and relevance
4. **Storage**: Persisting to database with metadata
5. **Indexing**: Creating efficient lookup structures
6. **Retrieval**: Finding relevant knowledge for current context
7. **Enhancement**: Integrating knowledge into prompts
8. **Application**: Using knowledge in model processing

### 3. Model Collaboration Flow

```
Input → Task Analysis → Mode Selection → Parallel Processing → 
Result Integration → Quality Assessment → Output
```

1. **Input**: User query or prompt
2. **Task Analysis**: Determining task type and requirements
3. **Mode Selection**: Choosing collaboration mode (collaborative/specialized/competitive)
4. **Parallel Processing**: Running models in parallel where appropriate
5. **Result Integration**: Combining outputs based on selected mode
6. **Quality Assessment**: Evaluating response quality with metrics
7. **Output**: Delivering final response with attribution metadata

## API Reference

### Authentication API

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/register` | POST | Register new user | `{ username, password, email? }` | User object |
| `/api/login` | POST | Authenticate user | `{ username, password }` | User object |
| `/api/logout` | POST | End user session | None | Success message |
| `/api/user` | GET | Get current user | None | User object |

### AI Models API

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/ai/qwen` | POST | Qwen model direct access | `{ prompt, options? }` | Generated text with metadata |
| `/api/ai/olympic` | POST | Olympic model direct access | `{ prompt, options? }` | Generated text with metadata |
| `/api/ai/hybrid` | POST | Hybrid collaboration engine | `{ prompt, options?, conversationId? }` | Generated text with collaboration metadata |
| `/api/ai/generate` | POST | Generic model access | `{ prompt, model, options?, enhanceWithKnowledge? }` | Generated text with metadata |

### Reasoning API

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/ai/reason` | POST | Perform reasoning | `{ input, operationSet?, maxSteps?, detailedOutput? }` | Reasoning result with steps |
| `/api/ai/reason/operations` | GET | Get available operation sets | None | List of operation sets |
| `/api/ai/reason/history` | GET | Get reasoning history | None | List of reasoning sessions |

### Knowledge API

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/knowledge` | POST | Add new knowledge | `{ content, source, domains?, importanceScore? }` | Knowledge entry |
| `/api/knowledge/user` | POST | Add user knowledge | `{ content, domains?, importanceScore? }` | Knowledge entry |
| `/api/knowledge/extract` | POST | Extract knowledge from conversation | `{ conversationId }` | Extracted knowledge entries |
| `/api/knowledge/retrieve` | GET | Get relevant knowledge | Query params: `query, domains?, limit?, minImportance?` | Knowledge entries |
| `/api/knowledge/similar` | GET | Find similar knowledge | Query params: `content, threshold?, limit?` | Similar knowledge entries |
| `/api/knowledge/domains` | GET | Get all domains | None | List of domains |
| `/api/knowledge/domains` | POST | Create new domain | `{ name, description? }` | Domain object |
| `/api/knowledge/domains/:domain` | GET | Get knowledge in domain | Query params: `limit?` | Knowledge entries |
| `/api/knowledge/:id/related` | GET | Get related knowledge | Query params: `limit?` | Related knowledge entries |
| `/api/knowledge/graph` | GET | Get knowledge graph | Query params: `centralDomain?, depth?, maxNodes?` | Graph data |
| `/api/knowledge/enhance-prompt` | POST | Enhance prompt with knowledge | `{ prompt, domains? }` | Enhanced prompt |

### Conversation API

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/ai/conversation` | POST | Start model conversation | `{ topic, userPrompt, mode? }` | Conversation ID |
| `/api/ai/conversation/:id` | GET | Get conversation | None | Conversation object |

### System API

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/ai/system/resources` | GET | Get resource info | None | Resource statistics |
| `/api/ai/status` | GET | Get system status | None | System status info |

## WebSocket API

The system supports real-time updates via WebSocket connections at `/ws`.

### Message Types

| Type | Direction | Payload | Description |
|------|-----------|---------|-------------|
| `ping` | Client → Server | None | Connection heartbeat |
| `pong` | Server → Client | `{ timestamp }` | Heartbeat response |
| `subscribe_conversation` | Client → Server | `{ conversationId }` | Subscribe to conversation updates |
| `conversation_update` | Server → Client | `{ conversation, isComplete }` | Conversation state update |
| `new-message` | Server → Client | `{ message }` | New message notification |
| `error` | Server → Client | `{ error }` | Error notification |

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `NODE_ENV` | Environment mode (development/production) | development |
| `PORT` | HTTP server port | 5000 |
| `SESSION_SECRET` | Secret for session encryption | Required |

### Resource Profiles

| Profile | RAM | Max Concurrent Requests | Max Context Size | Max Tokens | 
|---------|-----|--------------------------|-----------------|------------|
| minimal | 16GB | 2 | 4096 | 1024 |
| standard | 32GB | 5 | 8192 | 2048 |
| enhanced | 64GB | 10 | 16384 | 4096 |
| unlimited | >64GB | 30 | 32768 | 8192 |

## Performance Considerations

### Optimization Strategies

1. **Database Performance**
   - Indexing on frequently queried fields
   - Connection pooling for efficient database access
   - JSON/B data types for flexible schemaless data

2. **AI Processing Optimization**
   - Parallel model execution where possible
   - Response caching for repeated queries
   - Model quantization in resource-constrained environments

3. **Scaling Considerations**
   - Stateless API design for horizontal scaling
   - Resource-aware request handling
   - Graceful degradation under high load

### Benchmarks

| Operation | Average Response Time | p95 Response Time |
|-----------|----------------------|-------------------|
| Direct Model Query | 2s | 5s |
| Hybrid Collaborative | 5s | 10s |
| Knowledge Retrieval | 100ms | 350ms |
| Reasoning (simple) | 300ms | 800ms |
| Reasoning (complex) | 2s | 5s |

## Error Handling

### Error Response Format

```json
{
  "error": "Brief error description",
  "error_id": "error-1745291106220-203",
  "request_id": "request-1745291106220-203",
  "timestamp": "2025-04-22T03:05:11.000Z",
  "details": "Detailed error information"
}
```

### Common Error Codes

| HTTP Status | Description | Common Causes |
|-------------|-------------|---------------|
| 400 | Bad Request | Invalid input parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource does not exist |
| 429 | Too Many Requests | Rate limit exceeded or system at capacity |
| 500 | Internal Server Error | Unexpected server-side error |

## Security Implementation

1. **Authentication**
   - Password hashing using scrypt with per-user salts
   - Session-based authentication with secure cookies
   - CSRF protection for all state-changing operations

2. **Input Validation**
   - Zod schema validation for all API inputs
   - Content sanitization to prevent injection attacks
   - Request size limits to prevent DoS attacks

3. **Rate Limiting**
   - Per-IP and per-user rate limits on all endpoints
   - Increasing backoff for repeated failures
   - Separate pools for authenticated vs anonymous requests

4. **Data Protection**
   - TLS for all connections
   - Sensitive data encryption in database
   - Minimal exposure of internal system details

## Monitoring and Logging

The system implements structured logging with the following components:

1. **Request Logging**
   - Request IDs for traceability
   - Performance metrics for all operations
   - Input/output summaries for debugging

2. **Error Logging**
   - Detailed error information with stack traces
   - Error categorization for aggregation
   - Error IDs for correlation with client responses

3. **Performance Monitoring**
   - Resource utilization tracking
   - Response time metrics by endpoint
   - Database query performance

4. **Health Checks**
   - Internal state consistency checks
   - External dependency health monitoring
   - Automated recovery procedures where possible

## Deployment Architecture

The system can be deployed in several configurations:

1. **Single-Server Deployment**
   - All components on one server
   - Suitable for low to medium traffic
   - Simplest deployment model

2. **Multi-Server Deployment**
   - Separate API and worker servers
   - Horizontal scaling for higher traffic
   - Load balancer for request distribution

3. **Cloud-Native Deployment**
   - Containerized components (Docker)
   - Orchestration with Kubernetes
   - Autoscaling based on demand
   - Managed database services

## Development Workflow

1. **Local Development**
   - Run with `npm run dev`
   - Development server with hot reloading
   - Local database setup with migrations

2. **Testing**
   - Unit tests for core components
   - Integration tests for API endpoints
   - End-to-end tests for critical flows

3. **Deployment**
   - Build with `npm run build`
   - Production start with `npm start`
   - Environment-specific configuration via env vars

## Internal APIs and Interfaces

### Knowledge Manager Interface

```typescript
interface KnowledgeManager {
  addKnowledge(
    content: string,
    source: KnowledgeSource,
    domains: string[],
    userId?: number,
    metadata?: any,
    importanceScore?: number
  ): Promise<KnowledgeEntry>;
  
  addUserKnowledge(
    content: string,
    domains: string[],
    userId: number,
    metadata?: any,
    importanceScore?: number
  ): Promise<KnowledgeEntry>;
  
  extractKnowledgeFromConversation(
    conversationId: string,
    userId?: number
  ): Promise<KnowledgeEntry[]>;
  
  retrieveRelevantKnowledge(
    query: string,
    domains?: string[],
    limit?: number,
    minImportance?: number
  ): Promise<KnowledgeEntry[]>;
  
  // Additional methods...
}
```

### Resource Manager Interface

```typescript
interface ResourceManager {
  getProfile(): ResourceProfile;
  getLimits(): ResourceLimits;
  getSystemResources(): SystemResources;
  startRequest(): void;
  endRequest(): void;
  canProcessRequest(): boolean;
  getMaxContextSize(): number;
}

type ResourceProfile = 'minimal' | 'standard' | 'enhanced' | 'unlimited';

interface ResourceLimits {
  maxContextSize: number;
  maxTokens: number;
  maxConcurrentRequests: number;
  maxProcessingTime: number;
  enableParallelInference: boolean;
  useQuantization: boolean;
}

interface SystemResources {
  totalMemory: number;
  availableMemory: number;
  cpuCount: number;
  cpuUsage: number;
  activeRequests: number;
}
```

### Conversation Manager Interface

```typescript
interface ConversationManager {
  startConversation(
    topic: string,
    userPrompt: string,
    mode?: ConversationMode,
    userId?: number
  ): Promise<string>;
  
  getConversation(conversationId: string): ModelConversation | undefined;
  
  isConversationComplete(conversationId: string): boolean;
}

type ConversationMode = 'collaborative' | 'debate' | 'critical' | 'brainstorming';

interface ModelConversation {
  id: string;
  topic: string;
  mode: ConversationMode;
  turns: ConversationTurn[];
  startedAt: Date;
  endedAt?: Date;
  userPrompt: string;
  conclusion?: string;
}

interface ConversationTurn {
  model: 'qwen' | 'olympic';
  content: string;
  timestamp: Date;
  metadata?: {
    confidence?: number;
    questionCount?: number;
    criticalPoints?: number;
    suggestions?: number;
  };
}
```

## Dependencies and Third-Party Libraries

### Backend Dependencies

- **Express**: Web server framework
- **PostgreSQL/Neon**: Database
- **Drizzle ORM**: Database ORM
- **Zod**: Schema validation
- **Passport**: Authentication
- **ws**: WebSocket support

### Frontend Dependencies

- **React**: UI library
- **React Query**: Data fetching and state management
- **Wouter**: Routing
- **ShadCN/UI**: Component library
- **Tailwind CSS**: Styling