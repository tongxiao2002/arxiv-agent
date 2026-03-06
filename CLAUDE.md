# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arxiv-Agent is an autonomous research assistant that automatically scans daily academic paper submissions from arXiv and Papers.cool, filters them based on user-defined research interests, and delivers personalized email digests. The system operates on a server with scheduled execution, running paper scanning at midnight (00:00) and delivering summaries at 9:00 AM (09:00) daily in configurable timezone (default UTC+8).

**Key Features**:
- Automated daily paper scanning from arXiv (non-CS) and Papers.cool (CS preferred)
- LLM-based relevance classification using configurable providers (OpenAI, Anthropic, local)
- LLM-based abstract summarization (2-3 sentence summaries)
- JSON file storage for paper metadata with monthly archiving
- Email notifications via python smtplib
- Robust error handling with exponential backoff retry
- Configuration via YAML file and `.env` for secrets

**Architecture**: Built on `langchain-ai/deepagents` framework with specialized agents (Scraper, Classifier, Emailer) coordinated by a supervisor agent.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core runtime and language |
| **langchain-ai/deepagents** | Agent framework foundation |
| **LangChain** | LLM abstraction and tool integration |
| **APScheduler** | Cron-style job scheduling with timezone support |
| **PyYAML** | Configuration file parsing |
| **python-dotenv** | Environment variable management |
| **Requests** | HTTP client for API calls |
| **BeautifulSoup4** | HTML parsing (if needed for scraping) |
| **pytz** | Timezone handling |
| **pytest** | Testing framework |
| **black/isort** | Code formatting and import sorting |

**Optional Dependencies**:
- **OpenAI Python Library**: For GPT models
- **Anthropic Python Library**: For Claude models

---

## Commands

```bash
# Install dependencies (after creating requirements.txt)
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Run the agent (main entry point)
python -m arxiv_agent.cli

# Run tests
pytest

# Format code
black arxiv_agent tests
isort arxiv_agent tests

# Type checking
mypy arxiv_agent

# Lint (if configured)
# flake8 arxiv_agent
```

---

## Project Structure

```
arxiv-agent/                          # Project root
├── arxiv_agent/                      # Main package
│   ├── __init__.py
│   ├── cli.py                       # CLI entry point
│   ├── config.py                    # Configuration management
│   ├── scheduler.py                 # APScheduler integration
│   ├── agents/                      # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                  # Base agent class
│   │   ├── scraper_agent.py         # Paper fetching agent
│   │   ├── classifier_agent.py      # LLM classification agent
│   │   ├── emailer_agent.py         # Email sending agent
│   │   └── supervisor.py            # Agent coordination
│   ├── sources/                     # Paper source implementations
│   │   ├── __init__.py
│   │   ├── base_source.py           # Abstract source class
│   │   ├── arxiv_source.py          # arXiv.org implementation
│   │   └── papers_cool_source.py    # Papers.cool implementation
│   ├── storage/                     # Data persistence
│   │   ├── __init__.py
│   │   ├── json_storage.py          # JSON file management
│   │   └── archiver.py              # Monthly archiving
│   ├── email/                       # Email functionality
│   │   ├── __init__.py
│   │   ├── sender.py                # Email service integration
│   │   └── templates.py             # Email templates
│   ├── llm/                         # LLM integration
│   │   ├── __init__.py
│   │   ├── classifier.py            # Relevance classification
│   │   └── summarizer.py            # Abstract summarization
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logging.py               # Logging configuration
│       ├── retry.py                 # Exponential backoff retry
│       └── timezone.py              # Timezone handling
├── config.yaml                      # User configuration (example: config.yaml.example)
├── .env.example                     # Example environment variables
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
├── pyproject.toml                   # Project metadata and build config
├── tests/                           # Test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_sources.py
│   └── ...
├── logs/                            # Daily log files (auto-created)
├── papers/                          # JSON paper storage (auto-created)
└── archive/                         # Monthly compressed archives (auto-created)
```

---

## Architecture

**Agent-Based Architecture**: Built on `langchain-ai/deepagents` framework with specialized agents:
1. **Scraper Agent**: Fetches papers from configured sources (arXiv/Papers.cool)
2. **Classifier Agent**: Uses LLM for relevance classification and abstract summarization
3. **Emailer Agent**: Sends formatted email notifications
4. **Supervisor Agent**: Coordinates agent execution and handles scheduling

**Scheduling**: Uses APScheduler for two separate jobs:
- **Scan Job (00:00)**: Fetches, classifies, summarizes, and stores papers
- **Email Job (09:00)**: Reads stored papers and sends email digest

**Data Flow**:
```
Sources (arXiv/Papers.cool) → Scraper Agent → Classifier Agent → JSON Storage → Emailer Agent → User Inbox
```

**Error Handling**: Exponential backoff retry for network operations, skip papers after 5 failures, continue processing other papers.

**Configuration**: YAML config file (`config.yaml`) for settings, `.env` file for API keys and secrets.

---

## Code Patterns

### Naming Conventions
- **Modules**: `snake_case.py` (e.g., `json_storage.py`, `scraper_agent.py`)
- **Classes**: `PascalCase` (e.g., `ArxivSource`, `ClassifierAgent`)
- **Functions/Methods**: `snake_case` (e.g., `fetch_papers`, `classify_relevance`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private members**: `_leading_underscore` (e.g., `_validate_config`)

### File Organization
- Each major component in its own module under appropriate directory
- Keep module files focused (<= 300 lines when possible)
- Use `__init__.py` files to expose public API
- Separate concerns: scraping logic in `sources/`, LLM logic in `llm/`, etc.

### Error Handling
- Use custom exception classes for domain-specific errors
- Implement exponential backoff with jitter for network operations
- Log errors with context but don't expose sensitive data
- Use `try/except` with specific exception types, not bare `except:`

### Type Hints
- Use Python type hints throughout codebase
- Run `mypy` as part of validation
- Define `TypedDict` or dataclasses for complex data structures

### Logging
- Use structured logging with context (paper IDs, job IDs, etc.)
- Log to both console and daily rotating files in `./logs/`
- Different log levels: INFO for normal operations, DEBUG for troubleshooting

---

## Testing

- **Run tests**: `pytest`
- **Test location**: `tests/` directory mirroring source structure
- **Pattern**: Unit tests for individual components, integration tests for agent coordination
- **Mock external dependencies**: Use `unittest.mock` for LLM APIs, email services, network calls
- **Test data**: Use fixtures for paper data, configuration, etc.

**Test Structure**:
```
tests/
├── conftest.py              # Shared fixtures
├── test_config.py          # Configuration tests
├── sources/                # Source implementation tests
│   ├── test_arxiv_source.py
│   └── test_papers_cool_source.py
├── agents/                 # Agent tests
│   ├── test_scraper_agent.py
│   ├── test_classifier_agent.py
│   └── test_emailer_agent.py
└── integration/            # Integration tests
    └── test_full_pipeline.py
```

---

## Validation

```bash
# Before committing, run:
black arxiv_agent tests    # Format code
isort arxiv_agent tests    # Sort imports
mypy arxiv_agent           # Type checking
pytest                     # Run tests
```

**Pre-commit hooks**: Configure `pre-commit` to run validation automatically.

---

## Key Files

| File | Purpose |
|------|---------|
| `arxiv_agent/cli.py` | Main CLI entry point, starts scheduled agent |
| `arxiv_agent/config.py` | Configuration loading and validation |
| `arxiv_agent/scheduler.py` | Job scheduling with APScheduler |
| `arxiv_agent/agents/supervisor.py` | Agent coordination and workflow |
| `config.yaml` | User configuration (topics, schedule, sources, etc.) |
| `.env` | API keys and secrets (LLM, email service) |
| `pyproject.toml` | Project metadata, dependencies, tool configuration |

---

## On-Demand Context

| Topic | File |
|-------|------|
| **Full PRD** | `.claude/PRD.md` |
| **Agent Framework** | `langchain-ai/deepagents` GitHub repo |
| **LangChain Integration** | LangChain documentation |
| **arXiv API** | `https://arxiv.org/help/api` |
| **APScheduler** | APScheduler documentation |

---

## Notes

- **Timezones**: Default to UTC+8 but configurable via `timezone` setting in config
- **Rate Limiting**: Respect source API rate limits, implement delays between requests
- **Cost Management**: LLM API calls are the primary cost driver, implement caching where possible
- **Data Retention**: Monthly archiving of JSON files, configurable retention period
- **Error Recovery**: System should recover from crashes without data loss
- **Secrets Management**: Never commit `.env` file, use `.env.example` as template
- **Logging**: Check `./logs/` directory for daily operation logs
- **First Run**: Copy `config.yaml.example` to `config.yaml` and `.env.example` to `.env` before first run