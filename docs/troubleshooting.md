# Troubleshooting

## Startup fails with "Runtime configuration validation failed"

Check the logged validation errors first. Common causes:

- `sources.primary` is set to `papers_cool`. Change it to `arxiv`.
- `llm.provider` is set to `local`. Change it to `openai` or `anthropic`.
- The required API key for the selected LLM provider is missing.
- `SMTP_PASSWORD` is missing for an authenticated SMTP configuration.

## Startup fails with configuration validation errors

Typical fixes:

- Use a valid IANA timezone such as `Asia/Shanghai` or `UTC`.
- Ensure `schedule.scan_time` and `schedule.email_time` use `HH:MM`.
- Make sure `email.subject_template` includes `{date}`.
- Make sure `advanced.max_retries` and `advanced.request_timeout` are positive values.

## LLM calls keep retrying and then fail

- Check the root-cause message logged after the retry wrapper fails.
- Verify the selected `llm.model` exists in your provider account.
- Increase `advanced.request_timeout` if the provider is slow.
- Reduce `sources.arxiv.max_papers` while testing to shorten runs.

## SMTP delivery fails

- Verify `email.smtp_host`, `email.smtp_port`, and `email.smtp_security`.
- Confirm `SMTP_PASSWORD` matches `email.smtp_username`.
- For Gmail or similar providers, use an app password instead of a normal login password.
- Retry failures include the underlying SMTP error in logs and surfaced exceptions.

## No email is sent because no papers are available

- Run `python -m arxiv_agent.cli run-once --dry-run --config config.yaml` first.
- Check that the scrape step stored a file under `storage.data_dir`.
- If a date was already processed, classification now skips papers already stored as enhanced records.
- If no relevant papers are found, Arxiv-Agent sends the "no papers" email variant instead of a digest.
