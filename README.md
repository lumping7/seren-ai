# Seren AI System

Seren is a cutting-edge hybrid AI development platform pioneering autonomous, adaptive, and secure intelligent system creation.

## Core Features

- **Multi-model AI architecture** - Leverages Qwen2.5-omni-7b and OlympicCoder-7B models
- **Advanced collaboration modes** - Collaborative, specialized, and competitive model interactions
- **Offline operation** - Fully locally hosted models for privacy and reliability
- **Quantum-enhanced security** - Secure inter-model communications
- **Modular neuro-symbolic reasoning engine** - For enhanced AI reasoning capabilities

## System Architecture

Seren is built with a modular architecture:

### AI Core
- **AI Engine** - Core orchestration of AI models and capabilities
- **Model Manager** - Loads and manages local AI models
- **Model Downloader** - Downloads required models from Hugging Face Hub
- **Model Communication** - Enables secure inter-model communication
- **API Server** - HTTP API access to AI capabilities

### Security
- **Quantum Encryption** - Simulated quantum-secure communications
- **Authentication** - API key-based access control

## Getting Started

### Prerequisites
- Python 3.8+ with required packages (see `install_requirements.sh`)
- ~30GB disk space for full models (or ~8GB for quantized models)
- 16GB+ RAM (32GB+ recommended)
- GPU with 8GB+ VRAM (optional but recommended)

### Installation

1. Clone the repository
2. Run the installation script:
   ```bash
   chmod +x install_requirements.sh
   ./install_requirements.sh
   ```
3. Download the required models:
   ```bash
   python -m ai_core.model_downloader --model all --quantized
   ```
   For full-precision models, omit the `--quantized` flag.

### Running the Server

Start the central server:
```bash
python -m ai_core.server
```

By default, this will start:
- HTTP API server on http://localhost:8000
- WebSocket server on port 8001 (when implemented)

## API Usage

The HTTP API provides the following endpoints:

- **`GET /health`** - Health check
- **`GET /api/status`** - System status
- **`POST /api/query`** - Process general queries
- **`POST /api/code/generate`** - Generate code
- **`POST /api/code/analyze`** - Analyze code
- **`POST /api/code/explain`** - Explain code
- **`POST /api/collaborative`** - Generate collaborative responses
- **`POST /api/specialized`** - Generate specialized responses
- **`POST /api/competitive`** - Generate competitive responses

Authentication is done via API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

The default API key is `dev-key` for development.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Seren AI System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐│
│  │               │    │               │    │               ││
│  │  HTTP API     │    │  WebSocket    │    │  CLI          ││
│  │  Server       │    │  Server       │    │  Interface    ││
│  │               │    │               │    │               ││
│  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘│
│          │                    │                    │        │
│          └──────────┬─────────┴──────────┬─────────┘        │
│                     │                    │                  │
│             ┌───────▼────────────────────▼───────┐          │
│             │                                    │          │
│             │            AI Engine               │          │
│             │                                    │          │
│             └───────┬────────────────────┬───────┘          │
│                     │                    │                  │
│        ┌────────────▼───────┐    ┌───────▼────────────┐     │
│        │                    │    │                    │     │
│        │   Model Manager    │    │  Communication     │     │
│        │                    │    │  System            │     │
│        └────────┬───────────┘    └────────┬───────────┘     │
│                 │                          │                 │
│        ┌────────▼───────────┐    ┌────────▼───────────┐     │
│        │                    │    │                    │     │
│        │   Qwen2.5-omni-7b  │    │  Quantum           │     │
│        │   OlympicCoder-7B  │    │  Encryption        │     │
│        │                    │    │                    │     │
│        └────────────────────┘    └────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## License

This project is proprietary and confidential. © 2025 Seren AI, All Rights Reserved.

## Acknowledgements

- The Qwen team for the Qwen2.5-omni-7b model
- The OlympicCoder-7B team for their specialized code model
- Hugging Face for hosting the models