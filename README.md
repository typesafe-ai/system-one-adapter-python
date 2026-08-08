OpenTypeSafe is a library that is a drop-in replacement for the TypeSafeClient and API, but using LLM APIs.

It's main uses cases are
 - Evaluating TypeSafe's API vs an LLM API for cost/speed/intelligence
 - A backup API in case TypeSafe goes down

## Usage

```python
from open_typesafe_client import OpenTypeSafeClient, LLMConfig
from typesafe_client import (
    NoulQuestion,
    ScoreQuestion,
    ChoiceQuestion,
    TypeSafeClient
)

# OpenTypeSafeClient has the same interface as TypeSafeClient and is meant to , except for one exta arg `llm_config`, which configures
# the LLM to use.
open_typesafe_client = OpenTypeSafeClient()
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

llm_response = open_typesafe_client.system_one("gpt-4o-mini", document, questions)
typesafe_response = typesafe_client.system_one(document, questions)

# llm_response will have a nearly identical shape to typesafe_response
print(llm_response)
print(typesafe_response)
```

# Specification

- PydanticAI for queries
- By default, it requests probabilities, but there is also a mode that supports getting discrete values
  - it will then map the discrete value to a probability distribution of all 0s, except one element which is 1.0
- Telemetry
  - returns telemetry in the response like input_tokens, output_tokens, n_retries, latency
- Tests make real LLM and TypeSafe API calls, but also caches the calls to make testing deterministic (see json_cache.py)
- Responses include .type, .answers, and .usage fields like the reference models.
- Provider SDK exceptions are mapped onto reference-shaped error types: 
  - TypeSafeAuthError (bad key), TypeSafeTimeoutError (timeouts and connection failures) 
  - TypeSafeTokensExceededError (context window exceeded)
  - TypeSafeUnknownError (everything else, carrying the HTTP status_code)
  - All inherit from TypeSafeApiError

class OpenTypeSafeClient
   def __init__(
      self, 
      structured_outputs: bool = False,
      noul_mode: Literal['probabilities','discrete']="probabilities",
      score_mode: Literal['probabilities','discrete']="probabilities",
      choice_mode: Literal['probabilities','discrete']="probabilities",
      retry: RetryConfig = NoRetries(),
  ):
     pass
   
   