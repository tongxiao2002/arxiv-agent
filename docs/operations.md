# Operations Guide

## Daily Workflow

- `python -m arxiv_agent.cli run-once --dry-run --config config.yaml` is the recommended pre-production check.
- `python -m arxiv_agent.cli start --config config.yaml` runs the scheduler in the foreground.
- The scheduler creates two daily jobs: scan/classify and email delivery.

## Logs

- Logs are written to `storage.log_dir`.
- The CLI respects `advanced.log_level` and the optional `LOG_LEVEL` environment override.
- Startup logs include timezone, source, provider, schedule, retry count, timeout, and storage paths.

## Storage and Retention

- Daily paper data is written to `storage.data_dir` as JSON.
- Old daily files are archived into `storage.archive_dir`.
- Archive rotation uses `storage.retention_days` to decide when to move old files.

## Deployment Checklist

- Confirm `config.yaml` validates with supported values only.
- Confirm exactly one LLM API key is present for the selected provider.
- Run a dry run and inspect the generated logs before enabling scheduled operation.
- Verify SMTP credentials with a real send before relying on the scheduler.
- Ensure the process manager restarts the app if the foreground scheduler exits.
