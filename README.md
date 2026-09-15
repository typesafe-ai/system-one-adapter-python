# OpenSystemOne

OpenSystemOne evaluates TypeSafe questions through LLM APIs using PydanticAI. It
supports comparisons with TypeSafe and an LLM fallback for the evaluation API.

## Usage

Install with `uv add open-system-one`. This package uses
[`typesafe-sdk` 0.5.7](https://pypi.org/project/typesafe-sdk/0.5.7/).

```python
from open_system_one import OpenSystemOne
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

state = "This book was a delight to read."
questions = {
    "positive": Noul(instructions="The review is positive."),
    "stars": Score(instructions="Rate the review.", criteria=["Bad.", "Good."]),
    "genre": Choice(
        instructions="Which genre is this?",
        criteria={"fiction": "A story.", "nonfiction": "Facts."},
    ),
}

with TypeSafeClient() as typesafe_client:
    typesafe_response = typesafe_client.system_one(state, questions, model="speed_latest")
with OpenSystemOne(
    structured_outputs=True,
    llm_answer_mode="probabilities",
    normalize_probabilities=True,
    model="gpt-4o-mini",
) as open_system_one:
    llm_response = open_system_one.system_one(state=state, questions=questions)

print(typesafe_response.choices["genre"].choice)
print(llm_response.choices["genre"].choice)
print(llm_response.scores["stars"].probabilities[1])
print(llm_response.model_dump_json(indent=2))
```

Set the matching provider credential (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).
Only the reference `TypeSafeClient` needs `TYPESAFE_API_KEY`.

For async callers, `AsyncOpenSystemOne.system_one` matches the awaitable evaluation
interface of `typesafe_sdk.AsyncTypeSafeClient`:

```python
from open_system_one import AsyncOpenSystemOne

async def evaluate(state, questions):
    async with AsyncOpenSystemOne(
        structured_outputs=True,
        llm_answer_mode="probabilities",
        model="gpt-4o-mini",
    ) as client:
        return await client.system_one(state, questions)
```

## Migrating from typesafe-client

- Import `Noul`, `Score`, `Choice`, and `RetryPolicy` from `typesafe_sdk`.
- Call `system_one(state, questions, model=...)`; `model` is keyword-only and can
  also be set on `OpenSystemOne` at construction.
- Raw question dictionaries are accepted. Score criteria dictionaries must have
  consecutive integer keys starting at zero. This adapter continues to require
  at least two criteria for Choice and Score questions.
- Responses extend the SDK's `SystemOneResponse` and `Usage`, retaining `.debug`,
  cumulative usage, retry counts, latency, and `model_dump`/`model_dump_json`.
  Use `.nouls`, `.choices`, and `.scores` for typed answer views. Score probabilities
  and legends have integer keys in Python and string keys in JSON.
- SDK answers use tagged serialization; inspect their classes or typed views rather
  than an answer's `.type` attribute.
- Use `RetryPolicy(max_retries=...)` instead of `RetryConfig(max_attempts=...)`.
  `max_retries` counts retries after the first attempt. Retries remain disabled by
  default here; pass a policy on the client or override it on an individual call.
- Catch errors from `typesafe_sdk`. HTTP errors retain their status on `.status`,
  provider body on `.body`, and response headers on `.headers`. Connection and
  timeout failures inherit `TypeSafeAPIConnectionError`; malformed model output
  raises `TypeSafeAPIResponseValidationError`. All inherit `TypeSafeError` and
  terminal evaluation errors carry `.debug`.
- The adapter implements evaluation and context management without subclassing an
  SDK HTTP client. TypeSafe-specific transport options and the Models API are not
  part of its interface. `system_one_async` remains available on `OpenSystemOne`.
- SDK internal wire validation and retry builders are used to preserve their
  behavior; the dependency is constrained to `>=0.5.7,<0.6`.

# Replaying an LLM attempt

Every `llm_attempt` contains the native objects needed to call PydanticAI again. The
corresponding provider credential must be available in the environment.

```python
import asyncio

from pydantic_ai.models import infer_model

llm_attempt = response.debug["llm_attempts"][0]
model = infer_model(llm_attempt["debug_info"]["model_id"])
replayed_response = asyncio.run(
    model.request(
        llm_attempt["messages"],
        llm_attempt["model_settings"],
        llm_attempt["model_request_parameters"],
    )
)
```

Repeating a request does not guarantee identical nondeterministic model output.

# Provider schema compatibility

Treat the final provider HTTP schema as the compatibility contract. PydanticAI and
provider SDKs transform Pydantic's raw JSON Schema, and those transformations may
silently remove or rewrite unsupported keywords without an error or warning. When
changing output models or upgrading Pydantic, PydanticAI, or a provider SDK, inspect
the recorded provider request rather than relying only on `model_json_schema()`.
Pay particular attention to keywords beside `$ref`: their preservation can vary by
transformer. Keep behavior-critical context inside referenced definitions or concrete
properties when possible.

# Specification

- PydanticAI for queries
- Prompt construction
  - probability and discrete answer modes use distinct system instructions
  - documents are JSON-serialized inside `<document>` tags; embedded tag characters are escaped and document instructions are never followed
  - probability schemas define Noul truth probability, Choice option probability, and Score rubric-level probability semantics
  - prompted and native structured output use a pinned schema-instruction template owned by this package rather than PydanticAI's mutable default
- Structured output transport
  - `structured_outputs=False` requests plain-text JSON and does not use provider-native structured outputs or output tools
  - `structured_outputs=True` uses the provider's native structured-output mode without repeating the schema in the model instructions
- Answer mode
  - `llm_answer_mode="probabilities"` requests probability distributions
    - requests fixed-key probability objects for Choice and Score questions
    - requires every allowed label and forbids additional labels through Pydantic output validation
    - keeps each question on its probability-map definition and each criterion on its concrete probability property
    - maps the probability objects into TypeSafe answers with derived choice, score, and confidence fields
  - `llm_answer_mode="discrete"` maps the selected value to a probability distribution of all 0s except one value of 1.0
- Question validation
  - score and choice questions require at least two criteria
- Telemetry
  - `input_tokens` and `output_tokens` report the final successful model attempt
  - `input_tokens_total` and `output_tokens_total` aggregate every PydanticAI response received during the call, including malformed-structure attempts that later failed and attempts repeated after transient failures
  - `n_retries` counts retries of transient provider failures
  - `n_retries_malformed_structure` counts PydanticAI corrective retries for malformed output
  - `latency` is end-to-end request latency in seconds, including retries
- Debugging
  - `max_error` is the largest probability distribution-sum error
  - `invalid_probs` counts answers whose probability error exceeds `1e-6`
  - `probability_errors` maps invalid question IDs to their errors
  - `original_probabilities` contains LLM outputs changed by normalization and is omitted when empty
  - `retry_reasons` contains chronological `(category, message)` tuples for `provider_error` and `malformed_structure` retries; JSON serialization emits each tuple as a two-item array
  - `llm_attempts` contains one dictionary per PydanticAI model attempt
    - each dictionary contains native PydanticAI messages, model settings, and `ModelRequestParameters`; `model_dump(mode="json")` serializes them
    - request parameters preserve function tools, output tools, output mode, and the structured-output schema needed to reconstruct the call
    - `llm_response` contains the matching PydanticAI `ModelResponse`, or `None` when no response arrived
    - `debug_info` contains model, provider, finish-reason, and error metadata for that attempt
  - the messages and response round-trip through PydanticAI's `ModelMessagesTypeAdapter`
  - a public PydanticAI `model_request` hook captures the logical request immediately before the model call
  - these fields describe the provider-independent PydanticAI request, not the provider's final HTTP body
  - serialized messages retain timestamps, run IDs, and conversation IDs for complete debugging context
  - terminal exceptions expose the same diagnostics on `.debug`
    - failed malformed-output calls retain every raw response and corrective validation message
- Probability normalization
  - `normalize_probabilities=False` preserves LLM probabilities and only reports errors
  - `normalize_probabilities=True` renormalizes score and choice distributions
- Retry distinction
  - `retry` configures retries for transient connection, timeout, and retryable HTTP failures
  - `n_retry_malformed_structure` configures corrective retries for output that fails structural validation
  - the default status policy excludes authentication and bad-request errors; malformed output is handled by `n_retry_malformed_structure`
- Tests make real LLM and TypeSafe API calls, recorded as HTTP cassettes so replay is deterministic (vcrpy via pytest-recording)
  - cassettes are JSON and live in `tests/cassettes`, next to the tests
  - replay is the default and needs no credentials or network; the whole client stack runs against recorded provider traffic
  - re-record with `uv run pytest tests/test_client_with_live_apis.py --record-mode=rewrite`, which makes real billable calls and needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `TYPESAFE_API_KEY`
  - request/response credentials are stripped at record time (see `tests/conftest.py`); matching includes the request body because every call posts to the same endpoint
- Exception handling tests pass provider-shaped HTTP responses through the real provider SDK and PydanticAI stacks, checking SDK error classes, statuses, and bodies.
- Compatibility scope is documented in the migration section above.
