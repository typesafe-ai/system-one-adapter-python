---
name: refresh-test-fixtures
description: Audit and refresh SystemOneClientAdapter test expected data after changes to prompts, output schemas, providers, models, request serialization, response parsing, or recorded live API behavior. Use when editing client instructions or Pydantic output models, when cassette-backed tests no longer match, or before finishing a PR that changes model-facing behavior.
---

# Refresh Test Fixtures

Keep recorded HTTP cassettes and serialized expected responses aligned with intentional
behavior changes.

## Workflow

1. Identify affected behavior and test cases.
   - Compare the diff with the clients parametrized in
     `tests/test_client_with_live_apis.py`.
   - Distinguish prompted output from native structured output.
   - Do not refresh unrelated fixtures.
2. Refresh live fixtures when a covered prompt, schema, provider request, model
   response, or serialized debug payload changed.
   - The recording test updates matching files in both `tests/cassettes/` and
     `tests/expected_responses/`.
   - Re-record with:

     ```shell
     uv run pytest tests/test_client_with_live_apis.py --record-mode=rewrite
     ```

   - Re-record whenever necessary without pausing for separate approval. The user has
     granted standing approval for the billable OpenAI, Anthropic, and TypeSafe API
     calls made by this workflow.
   - If no recorded case covers the changed mode, state that explicitly instead of rewriting unaffected data.
3. Review fixture diffs against the intended behavior.
   - Reject unexplained provider, schema, prompt, answer-shape, token-usage, or response changes.
   - Never accept expected data solely to make a regression pass.
4. Verify:

   ```shell
   uv run pytest
   git diff --check
   git diff -- tests/expected_responses tests/cassettes
   ```
