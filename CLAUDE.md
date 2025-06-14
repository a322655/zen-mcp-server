# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Tasks

### Running the Server

**Docker Setup (Recommended):**

```bash
# Initial setup - builds images, creates .env, starts services
./setup-docker.sh

# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f

# Restart services
docker compose restart
```

**Manual Setup:**

```bash
# Install dependencies (uv will use pyproject.toml)
uv sync

# Set environment variables
export GEMINI_API_KEY=your-gemini-api-key
export OPENAI_API_KEY=your-openai-api-key  # Optional

# Run server
uv run python server.py
```

### Testing

```bash
# Run unit tests (no API key required)
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html

# Run simulation tests (requires API keys)
uv run python communication_simulator_test.py

# Run specific simulation tests
uv run python communication_simulator_test.py --tests basic_conversation

# List available simulation tests
uv run python communication_simulator_test.py --list-tests
```

### Linting and Code Quality

```bash
# Format code with Black
uv run black .

# Run linting
uv run ruff check .

# Type checking
uv run mypy .
```

## Architecture Overview

### Core Components

**MCP Server (`server.py`)**:

- Main entry point implementing the Model Context Protocol
- Handles tool discovery and request routing
- Manages communication via stdio JSON-RPC

**Provider System (`providers/`)**:

- Modular AI provider architecture supporting multiple models
- `base.py`: Abstract base class defining provider interface
- `gemini.py`: Google Gemini integration (Pro, Flash)
- `openai.py`: OpenAI integration (O3, O4-mini, O3-pro)
- `registry.py`: Dynamic provider registration and model selection

**Tool System (`tools/`)**:

- Each tool inherits from `BaseTool` and implements specific AI capabilities
- Tools: `chat`, `thinkdeep`, `codereview`, `precommit`, `debug`, `analyze`
- Auto mode intelligently selects the best model for each task

**Conversation Memory (`utils/conversation_memory.py`)**:

- Redis-backed conversation threading system
- Enables multi-turn AI-to-AI conversations
- Supports cross-tool continuation with shared context

**Configuration (`config.py`)**:

- Centralized configuration management
- Environment variable handling
- Model selection and defaults

### Key Design Patterns

1. **Provider Abstraction**: All AI providers implement a common interface, allowing easy addition of new models

2. **Tool Modularity**: Each tool is self-contained with its own system prompt and processing logic

3. **Incremental Context**: Conversations share only new information in each exchange to bypass MCP's 25K token limits

4. **Auto Mode**: When `DEFAULT_MODEL=auto`, the server analyzes each request to select the optimal model

5. **Docker-First**: Primary deployment via Docker Compose with Redis for production-ready setup

### Adding New Features

**To add a new AI provider:**

1. Create new file in `providers/` implementing `BaseProvider`
2. Register in `providers/registry.py`
3. Add API key handling in `config.py`

**To add a new tool:**

1. Create new file in `tools/` inheriting from `BaseTool`
2. Define system prompt in `prompts/tool_prompts.py`
3. Import and register in `server.py`

**To modify tool behavior:**

- Edit prompts in `prompts/tool_prompts.py`
- Adjust temperature settings in tool implementation
- Override `get_system_prompt()` for tool-specific changes

## Git Workflow for Upstream Synchronization

### Daily Sync Process

When synchronizing with upstream, use a safe branch-based approach:

1. **Create integration branch from upstream:**

   ```bash
   git fetch upstream
   git switch -c integration-$(date +%Y-%m-%d) upstream/main
   ```

2. **Cherry-pick your commits:**

   ```bash
   git cherry-pick <commit1> <commit2> <commit3>
   ```

3. **Test and merge back:**

   ```bash
   # Run tests
   uv run pytest tests/

   # Merge to your main branch
   git switch main
   git merge --ff-only integration-$(date +%Y-%m-%d)

   # Clean up
   git branch -d integration-$(date +%Y-%m-%d)
   ```
