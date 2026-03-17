# Arxiv-Agent

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Arxiv-Agent scans daily arXiv papers, classifies them against your research topics with an OpenAI or Anthropic model, stores the enhanced results locally, and sends an SMTP digest on a schedule.

## Supported Today

- `sources.primary: arxiv`
- `llm.provider: openai` or `anthropic`
- SMTP email delivery
- Foreground scheduling with APScheduler
- Local JSON storage, archiving, and rotating logs

## Not Yet Implemented

- `sources.primary: papers_cool`
- `llm.provider: local`
- Non-SMTP delivery providers

Those options still exist in config examples as future-facing placeholders, but the CLI now fails fast if you select them.

## Quick Start

### 1. Install

```bash
git clone https://github.com/tongxiao/arxiv-agent.git
cd arxiv-agent
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

Minimal supported `config.yaml`:

```yaml
agent:
  timezone: "Asia/Shanghai"

sources:
  primary: "arxiv"
  arxiv:
    categories: ["cs", "stat"]
    max_papers: -1

topics:
  - "machine learning"
  - "language models"

schedule:
  scan_time: "00:00"
  email_time: "09:00"

llm:
  provider: "openai"
  model: "gpt-4o-mini"

email:
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  smtp_security: "starttls"
  smtp_username: "your-email@example.com"
  from_email: "your-email@example.com"
  to_emails: ["your-email@example.com"]
  subject_template: "Daily Papers Digest - {date}"

storage:
  data_dir: "./papers"
  archive_dir: "./archive"
  log_dir: "./logs"
  retention_days: 30

advanced:
  max_retries: 5
  retry_backoff_factor: 2.0
  request_timeout: 30
  log_level: "INFO"
```

Required `.env` values:

```bash
# Pick one LLM provider
OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="sk-ant-..."

# Required when email.smtp_username is set
SMTP_PASSWORD="app-password-or-smtp-password"
```

Optional `.env` overrides:

```bash
TZ="Asia/Shanghai"
LOG_LEVEL="DEBUG"
MAX_RETRIES="5"
RETRY_BACKOFF_FACTOR="2"
REQUEST_TIMEOUT="30"
```

### 3. Dry Run First

```bash
python -m arxiv_agent.cli run-once --dry-run --config config.yaml
```

This still performs scraping and LLM classification, but it does not send a real email.

One-off interval backfill examples:

```bash
python -m arxiv_agent.cli run-once --from 2026-03-10T08:30 --to 2026-03-10T18:00
python -m arxiv_agent.cli run-once --from 2026-03-10T08:30 --to 2026-03-12T18:00 --no-email
```

`--from` and `--to` are naive local datetimes interpreted in `agent.timezone`. Interval mode is supported only on `run-once`, is capped at 31 days, and keeps second-level precision by widening the arXiv query to GMT minute bounds and then strict-filtering locally.

### 4. Run

```bash
python -m arxiv_agent.cli start
python -m arxiv_agent.cli run-once
python -m arxiv_agent.cli version
```

`start` does not accept `--from`, `--to`, or `--no-email`; scheduled mode remains date-based.

## Runtime Notes

- Logs are written to `storage.log_dir` and honor `advanced.log_level`.
- `advanced.max_retries`, `advanced.retry_backoff_factor`, and `advanced.request_timeout` now apply to source fetches, LLM calls, and SMTP delivery retries.
- The app logs an effective runtime summary at startup without printing secrets.
- arXiv fetches in 100-paper pages; `sources.arxiv.max_papers: -1` means no per-category cap and positive values remain per-category caps.
- Re-running arXiv scraping for a day merges in only new paper identities and preserves papers already stored for that date.
- Re-running classification for a day skips papers that are already stored as enhanced results.
- Interval `run-once` writes back into the existing daily JSON files and may augment a day instead of replacing it.
- When interval mode spans multiple local days, classification and email execution happen per affected day.

## Configuration Reference

- `agent.timezone`: IANA timezone used for date selection and scheduling.
- `sources.primary`: Supported value is currently `arxiv`.
- `sources.arxiv.max_papers`: arXiv-only. Use `-1` for unlimited per-category paging, or a positive integer for a per-category cap.
- `topics`: At least one non-empty topic is required.
- `llm.provider`: Supported values are `openai` and `anthropic`.
- `llm.model`: Set this to a model available in your provider account.
- `email.subject_template`: Must include `{date}`.
- `storage.log_dir`: Directory used by the CLI log setup.
- `advanced.*`: Runtime retry, timeout, and log verbosity controls.

## Operations and Troubleshooting

- Operational guidance: [docs/operations.md](docs/operations.md)
- Troubleshooting guide: [docs/troubleshooting.md](docs/troubleshooting.md)

## Development

```bash
pytest
black arxiv_agent tests
isort arxiv_agent tests
mypy arxiv_agent
```

Useful targeted checks:

```bash
pytest tests/test_config.py -v
pytest tests/test_cli.py -v
pytest tests/integration/test_notification_pipeline.py -v
```
