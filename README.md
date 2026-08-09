TypeSafeClientAdapter is a library that is a drop-in replacement for the TypeSafeClient and API, but using LLM APIs.

It's main uses cases are
 - Evaluating TypeSafe's API vs an LLM API for cost/speed/intelligence
 - A backup API in case TypeSafe goes down

## Usage

```python
from typesafe_client_adapter import TypeSafeClientAdapter
from typesafe_client import TypeSafeClient
from typesafe_client.api.models import (
    NoulQuestion,
    ScoreQuestion,
    ChoiceQuestion,
)

# TypeSafeClientAdapter has the same system_one interface as TypeSafeClient.
# Its constructor selects structured-output and answer modes.
typesafe_client_adapter = TypeSafeClientAdapter()
typesafe_client = TypeSafeClient()

document = "This book has been a delight to read! Looking forward to the next one!"

questions = {
    "positive": NoulQuestion(instructions="The book review is positive."),
    "stars": ScoreQuestion(
        instructions="Star rating for the book based on the review.",
        criteria=[
            "Horrendous. Unreadable garbage.",
            "Pretty bad, but theoretically readable.",
            "Acceptable, but just barely.",
            "Pretty good. Worth reading but not perfect.",
            "Transcendent and impactful. A must read.",
        ],
    ),
    "genre": ChoiceQuestion(
        instructions="Which genre this review is about.",
        criteria={
            "fiction": "A novel or short story.",
            "nonfiction": "A book based on facts, real events, or ideas.",
        },
    ),
}

typesafe_response = typesafe_client.system_one("speed_latest", document, questions)
llm_response = typesafe_client_adapter.system_one("gpt-4o-mini", document, questions)

# llm_response will have a nearly identical shape to typesafe_response
print(typesafe_response.model_dump_json(indent=2))
print(llm_response.model_dump_json(indent=2))
```

TypeSafe response:

```json
{
  "model": "speed_latest",
  "answers": {
    "positive": {
      "type": "noul",
      "noul": 0.98
    },
    "stars": {
      "type": "score",
      "score": 3.635,
      "confidence": 0.6958333333333333,
      "probabilities": {
        "0": 0.005,
        "1": 0.005,
        "2": 0.04,
        "3": 0.25,
        "4": 0.7
      }
    },
    "genre": {
      "type": "choice",
      "choice": "fiction",
      "confidence": 0.76,
      "probabilities": {
        "fiction": 0.88,
        "nonfiction": 0.12
      }
    }
  },
  "usage": {
    "input_tokens": 287,
    "output_tokens": 47
  }
}
```

LLM response:

```json
{
  "model": "gpt-4o-mini",
  "answers": {
    "positive": {
      "type": "noul",
      "noul": 0.96
    },
    "stars": {
      "type": "score",
      "score": 3.46,
      "confidence": 0.55,
      "probabilities": {
        "0": 0.01,
        "1": 0.02,
        "2": 0.07,
        "3": 0.3,
        "4": 0.6
      }
    },
    "genre": {
      "type": "choice",
      "choice": "fiction",
      "confidence": 0.64,
      "probabilities": {
        "fiction": 0.82,
        "nonfiction": 0.18
      }
    }
  },
  "usage": {
    "input_tokens": 356,
    "output_tokens": 91,
    "n_retries": 0,
    "n_retries_malformed_structure": 0,
    "latency": 0.74
  },
  "debug": {
    "max_error": 0.0,
    "invalid_probs": 0,
    "probability_errors": {},
    "llm_attempts": [
      {
        "messages": [
          {
            "parts": [
              {
                "content": "Document:\n\"This book was a delight to read.\"",
                "part_kind": "user-prompt"
              }
            ],
            "instructions": "Evaluate every question using only the supplied document.",
            "kind": "request",
            "metadata": null,
            "state": "complete"
          }
        ],
        "model_settings": {},
        "model_request_parameters": {
          "function_tools": [],
          "output_mode": "native",
          "output_object": {
            "json_schema": {"type": "object", "properties": {"answers": {}}}
          }
        },
        "llm_response": {
          "parts": [
            {
              "content": "{\"answers\": {...}}",
              "part_kind": "text"
            }
          ],
          "model_name": "gpt-4o-mini-2024-07-18",
          "kind": "response",
          "provider_name": "openai",
          "finish_reason": "stop",
          "state": "complete"
        },
        "debug_info": {
          "model_name": "gpt-4o-mini-2024-07-18",
          "provider": "openai",
          "finish_reason": "stop"
        }
      }
    ]
  }
}
```

# Specification

- PydanticAI for queries
- Structured output transport
  - `structured_outputs=False` requests plain-text JSON and does not use provider-native structured outputs or output tools
  - `structured_outputs=True` uses the provider's native structured-output mode
- By default, it requests probabilities, but there is also a mode that supports getting discrete values
  - it will then map the discrete value to a probability distribution of all 0s, except one element which is 1.0
- Telemetry
  - `input_tokens` and `output_tokens` aggregate every PydanticAI request made during the call, including malformed-structure retries and attempts that later failed and were retried, so a retried call is never under-billed
  - `n_retries` counts retries of transient provider failures
  - `n_retries_malformed_structure` counts PydanticAI corrective retries for malformed output
  - `latency` is end-to-end request latency in seconds, including retries
- Debugging
  - `max_error` is the largest probability distribution-sum error
  - `invalid_probs` counts answers whose probability error exceeds `1e-6`
  - `probability_errors` maps invalid question IDs to their errors
  - `original_probabilities` contains LLM outputs changed by normalization and is omitted when empty
  - `llm_attempts` contains one dictionary per PydanticAI model attempt
    - each dictionary contains serialized messages, model settings, and `ModelRequestParameters`
    - request parameters preserve function tools, output tools, output mode, and the structured-output schema needed to reconstruct the call
    - `llm_response` contains the matching serialized PydanticAI `ModelResponse`, or `null` when no response arrived
    - `debug_info` contains model, provider, finish-reason, and error metadata for that attempt
  - the messages and response round-trip through PydanticAI's `ModelMessagesTypeAdapter`
  - a public PydanticAI `model_request` hook captures the logical request immediately before the model call
  - these fields describe the provider-independent PydanticAI request, not the provider's final HTTP body
  - serialized messages retain timestamps, run IDs, and conversation IDs for complete debugging context
- Probability normalization
  - `normalize_probabilities=False` preserves LLM probabilities and only reports errors
  - `normalize_probabilities=True` renormalizes score and choice distributions
- Retry distinction
  - `retry` configures retries for transient connection, timeout, and retryable HTTP failures
  - `n_retry_malformed_structure` configures corrective retries for output that fails structural validation
  - authentication, context-window, and malformed-structure errors are not retried by `retry`
- Tests make real LLM and TypeSafe API calls, recorded as HTTP cassettes so replay is deterministic (vcrpy via pytest-recording)
  - cassettes are JSON and live in `tests/cassettes`, next to the tests
  - replay is the default and needs no credentials or network; the whole client stack runs against recorded provider traffic
  - re-record with `uv run pytest tests/test_live_apis.py --record-mode=rewrite`, which makes real billable calls and needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `TYPESAFE_API_KEY`
  - request/response credentials are stripped at record time (see `tests/conftest.py`); matching includes the request body because every call posts to the same endpoint
- Compatibility scope
  - `TypeSafeClientAdapter` subclasses `TypeSafeClient`
  - supports synchronous and asynchronous `system_one`, context management, `close`, and `aclose`
  - accepts the same documents and question models and returns the same response models
- `SystemOneResponse` includes `.model`, `.answers`, `.usage`, and `.debug`; each answer includes `.type`.
- Provider SDK exceptions are mapped onto reference-shaped error types: 
  - TypeSafeAuthError (bad key), TypeSafeTimeoutError (timeouts and connection failures) 
  - TypeSafeTokensExceededError (context window exceeded)
  - TypeSafeUnknownError (everything else, carrying the HTTP status_code)
  - All inherit from TypeSafeApiError

# Replaying an LLM attempt

Every `llm_attempt` can be deserialized and sent through PydanticAI again. The
recorded model ID is used by default, so the corresponding provider credential must
be available in the environment.

```python
import asyncio

from pydantic_ai.models import infer_model
from typesafe_client_adapter import deserialize_llm_attempt

llm_attempt = response.debug["llm_attempts"][0]
model = infer_model(llm_attempt["debug_info"]["model_id"])
replayed_response = asyncio.run(
    model.request(*deserialize_llm_attempt(llm_attempt))
)
```

Use the original model instance instead of `infer_model()` when the call used a
configured model or custom provider. Repeating a request does not guarantee identical
nondeterministic model output.

class TypeSafeClientAdapter(TypeSafeClient):
   def __init__(
      self, 
      structured_outputs: bool = False,
      llm_answer_mode: Literal['probabilities','discrete']="probabilities",
      normalize_probabilities: bool = False,
      n_retry_malformed_structure: int = 0,
      retry: RetryConfig = NoRetries(),
  ):
     pass
