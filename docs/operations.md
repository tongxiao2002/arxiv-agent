# Operations Guide

## Daily Workflow

- `python -m arxiv_agent.cli run-once --dry-run --config config.yaml` is the recommended pre-production check.
- `python -m arxiv_agent.cli run-once --from 2026-03-10T08:30 --to 2026-03-11T09:00 --config config.yaml` runs a one-off local datetime backfill.
- Add `--no-email` to any `run-once` invocation when you want scrape/classify/storage only.
- `python -m arxiv_agent.cli start --config config.yaml` runs the scheduler in the foreground.
- The scheduler creates two daily jobs: scan/classify and email delivery.
- `start` does not support `--from`, `--to`, or `--no-email`; those flags are run-once only.

## Interval Semantics

- `--from` and `--to` must be naive ISO local datetimes such as `2026-03-10T08:30`.
- Interval timestamps are interpreted in `agent.timezone`.
- The interval is closed, so papers exactly at the start or end timestamp are kept.
- The maximum supported interval length is 31 days.
- arXiv queries are widened to GMT minute bounds, then strict app-side filtering preserves the exact requested local interval.
- arXiv requests page in 100-result chunks and wait 3 seconds between sequential page fetches.
- `sources.arxiv.max_papers` is a per-category cap for both daily runs and interval backfills; `-1` means unlimited.

## Logs

- Logs are written to `storage.log_dir`.
- The CLI respects `advanced.log_level` and the optional `LOG_LEVEL` environment override.
- Startup logs include timezone, source, provider, schedule, retry count, timeout, and storage paths.

## Storage and Retention

- Daily paper data is written to `storage.data_dir` as JSON.
- Daily arXiv reruns merge into the existing day file, keeping previously stored records when the same paper identity appears again.
- Interval `run-once` still writes only daily files; it merges matching paper IDs into existing files instead of creating a separate interval artifact.
- Partial-day interval runs preserve unrelated papers already stored for the same day.
- Old daily files are archived into `storage.archive_dir`.
- Archive rotation uses `storage.retention_days` to decide when to move old files.

## Email Delivery

- Standard `run-once` and `start` keep the existing single-day digest behavior.
- Interval `run-once` sends one daily digest per affected local day unless `--no-email` is set.
- If an affected day contains no relevant papers after classification, the existing no-papers email variant is used for that day.

## Deployment Checklist

- Confirm `config.yaml` validates with supported values only.
- Confirm exactly one LLM API key is present for the selected provider.
- Run a dry run and inspect the generated logs before enabling scheduled operation.
- Verify SMTP credentials with a real send before relying on the scheduler.
- Ensure the process manager restarts the app if the foreground scheduler exits.
