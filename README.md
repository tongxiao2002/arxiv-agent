# Arxiv-Agent

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated research assistant that scans daily academic papers from arXiv and Papers.cool, filters them based on your research interests, and delivers personalized email digests.**

## Features

- **Automated Daily Scanning**: Fetches papers at midnight (configurable timezone)
- **Intelligent Filtering**: Uses LLMs for relevance classification (OpenAI, Anthropic, or local models)
- **Concise Summaries**: Generates 2-3 sentence LLM summaries for quick scanning
- **Email Digests**: Sends formatted daily emails at 9:00 AM (configurable)
- **Multiple Sources**: Supports arXiv.org (non-CS) and Papers.cool (CS preferred)
- **Robust Operation**: Exponential backoff retry, error handling, daily logging
- **Extensible Architecture**: Built on LangChain DeepAgents framework

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/tongxiao/arxiv-agent.git
cd arxiv-agent

# Install dependencies
pip install -r requirements.txt

# For development, also install development dependencies
pip install -r requirements-dev.txt
```

### 2. Configuration

```bash
# Copy configuration templates
cp config.yaml.example config.yaml
cp .env.example .env

# Edit config.yaml with your preferences
# Edit .env with your API keys
```

**Example `config.yaml`** (see `config.yaml.example` for full options):

```yaml
agent:
  name: "arxiv-agent"
  timezone: "Asia/Shanghai"  # Default: UTC+8

sources:
  primary: "arxiv"
  arxiv:
    categories: ["cs", "physics", "math"]

topics:
  - "machine learning"
  - "deep learning"
  - "natural language processing"

schedule:
  scan_time: "00:00"
  email_time: "09:00"

llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"

email:
  service: "sendgrid"
  from_email: "arxiv-agent@example.com"
  to_emails: ["your-email@example.com"]
```

**Required API keys in `.env`**:

```bash
# Choose one LLM provider
OPENAI_API_KEY="sk-..."          # If using OpenAI
# ANTHROPIC_API_KEY="sk-ant-..."  # If using Anthropic
# DEEPSEEK_API_KEY="sk-..."       # If using DeepSeek

# Choose one email service
SENDGRID_API_KEY="SG..."         # If using SendGrid
# MAILGUN_API_KEY="key-..."       # If using Mailgun
# MAILGUN_DOMAIN="sandbox....mailgun.org"
```

### 3. Run the Agent

```bash
# Test configuration
python -m arxiv_agent.cli run-once --dry-run

# Start scheduled agent (runs in foreground)
python -m arxiv_agent.cli start

# Run one-time scan and email
python -m arxiv_agent.cli run-once

# Show version
python -m arxiv_agent.cli version
```

## Project Structure

```
arxiv-agent/
├── arxiv_agent/               # Source code
│   ├── agents/               # Agent implementations
│   ├── sources/              # Paper source integrations
│   ├── storage/              # Data persistence
│   ├── email/                # Email functionality
│   ├── llm/                  # LLM integration
│   └── utils/                # Utilities (logging, retry, etc.)
├── config.yaml.example       # Configuration template
├── .env.example              # Environment variables template
├── logs/                     # Daily log files (auto-created)
├── papers/                   # JSON paper storage (auto-created)
└── archive/                  # Monthly archives (auto-created)
```

## Configuration Details

### Agent Configuration

- **name**: Identifier for the agent (default: "arxiv-agent")
- **timezone**: Timezone for scheduling (default: "Asia/Shanghai")

### Sources Configuration

- **primary**: Primary source ("arxiv" or "papers_cool")
- **arxiv.categories**: arXiv categories to scan (e.g., ["cs", "physics", "math"])
- **papers_cool.categories**: Papers.cool categories (CS subfields)

### Topics Configuration

List of research topics for LLM classification. Papers are evaluated against these topics.

### Schedule Configuration

- **scan_time**: Time to fetch and classify papers (24-hour format)
- **email_time**: Time to send email digest (24-hour format)

### LLM Configuration

- **provider**: LLM provider ("openai", "anthropic", "local")
- **model**: Model name (e.g., "gpt-4-turbo-preview", "claude-3-opus")
- **classification_temperature**: Temperature for relevance classification (default: 0.1)
- **summarization_temperature**: Temperature for abstract summarization (default: 0.3)

### Email Configuration

- **service**: Email service ("sendgrid" or "mailgun")
- **from_email**: Sender email address
- **to_emails**: List of recipient email addresses
- **subject_template**: Email subject template (supports {date} placeholder)

### Storage Configuration

- **data_dir**: Directory for daily JSON files (default: "./papers")
- **archive_dir**: Directory for monthly archives (default: "./archive")
- **log_dir**: Directory for log files (default: "./logs")
- **retention_days**: Days to keep data before archiving (default: 30)

## Architecture

Arxiv-Agent uses an agent-based architecture built on the LangChain DeepAgents framework:

1. **Scraper Agent**: Fetches papers from configured sources
2. **Classifier Agent**: Uses LLM for relevance classification and summarization
3. **Emailer Agent**: Sends formatted email notifications
4. **Supervisor Agent**: Coordinates agent execution and handles scheduling

The system runs two separate scheduled jobs:
- **Scan Job (00:00)**: Fetches, classifies, summarizes, and stores papers
- **Email Job (09:00)**: Reads stored papers and sends email digest

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black arxiv_agent tests
isort arxiv_agent tests

# Type checking
mypy arxiv_agent
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest --cov=arxiv_agent --cov-report=term-missing
```

### Code Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints throughout
- Document public functions and classes
- Write unit tests for new functionality

### Project Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

## Deployment

### Running as a Service

For production deployment, consider running the agent as a system service:

```ini
# systemd service example (/etc/systemd/system/arxiv-agent.service)
[Unit]
Description=Arxiv-Agent Paper Discovery Service
After=network.target

[Service]
Type=simple
User=arxiv-agent
WorkingDirectory=/opt/arxiv-agent
ExecStart=/usr/bin/python -m arxiv_agent.cli start
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker (Future)

Docker support is planned for a future release.

## Troubleshooting

### Common Issues

1. **Configuration file not found**: Ensure `config.yaml` exists in the current directory
2. **API key errors**: Verify API keys in `.env` are correct and have necessary permissions
3. **Timezone issues**: Ensure timezone is valid (see [tz database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones))
4. **Email delivery failures**: Check email service API keys and sender/recipient addresses

### Logs

Check the `./logs/` directory for daily log files:
- `arxiv-agent.log`: Main log file with detailed information
- `arxiv-agent-daily.log`: Daily rotating log file

### Debug Mode

Set `LOG_LEVEL="DEBUG"` in `.env` for detailed logging.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain DeepAgents](https://github.com/langchain-ai/deepagents) - Agent framework foundation
- [arXiv API](https://arxiv.org/help/api) - Paper metadata
- [Papers.cool](https://papers.cool) - CS paper aggregation

## Support

- **Issues**: [GitHub Issues](https://github.com/tongxiao/arxiv-agent/issues)
- **Documentation**: This README and code comments
- **Email**: tongxiao@example.com