# System One Compatibility Specification

Version: 1.0 draft

Status: Public draft for implementation and feedback

## Purpose

System One is a small, transport-neutral interface for evaluating a document against
a named collection of typed questions. A caller submits one document and any mixture
of boolean-like, choice, and ordinal score questions. The implementation returns one
typed answer for every question, including probabilities where the question has
multiple outcomes.

This specification publishes that contract independently of TypeSafe's hosted API,
the `typesafe-client` Python package, SystemOneClientAdapter, PydanticAI, and any
particular model provider. Authors are invited to build compatible clients, servers,
local model adapters, gateways, test doubles, and implementations in other languages.

The contract deliberately specifies the portable request, response, and behavioral
invariants. Provider selection, prompting, inference, billing, observability, and
deployment remain implementation concerns.

Machine-readable schemas accompany this document:

- [`spec/system-one-request-v1.schema.json`](spec/system-one-request-v1.schema.json)
- [`spec/system-one-response-v1.schema.json`](spec/system-one-response-v1.schema.json)

## Conformance language

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative.

An implementation can claim:

- **Request conformance** when it accepts every valid core request and applies the
  question semantics defined here.
- **Response conformance** when every successful response has the core shape and
  satisfies the answer invariants defined here.
- **System One v1 conformance** when it has both request and response conformance.
- **Python profile conformance** when it additionally provides the Python interface
  described below.
- **HTTP profile conformance** when it additionally provides the HTTP interface
  described below.

Implementations SHOULD state which optional extensions they emit. A compatibility
claim MUST identify deliberate deviations from the core contract.

## Core operation

The logical operation is:

```text
system_one(model, document, questions) -> response
```

- `model` selects an implementation-defined evaluation model or model profile.
- `document` is the untrusted data to evaluate.
- `questions` maps caller-defined question IDs to typed question definitions.
- `response.answers` maps those same IDs to typed answers.

The operation evaluates all questions against the same document. It MUST either
return a complete response or report an error; a successful partial response is not
part of the v1 contract.

Implementations MAY evaluate questions together or independently. They MUST preserve
question IDs exactly and MUST NOT interpret IDs as instructions.

## JSON data model

The wire representation uses JSON.

An **instruction value** is one of:

- a string;
- an array containing JSON values; or
- an object with string keys and JSON values.

Instruction values can therefore carry either natural language or structured data.
The top-level `document` MUST be an instruction value. A question instruction or Noul
criterion MAY additionally be `null` or omitted.

Numbers MUST be finite JSON numbers. Objects MUST have string keys. Duplicate object
keys are invalid even if a parser would otherwise keep one of them.

## Request

A request contains exactly three core fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `model` | string | yes | Opaque model or model-profile identifier |
| `document` | instruction value | yes | Untrusted data evaluated by every question |
| `questions` | object | yes | Non-empty map from question ID to question |

Example:

```json
{
  "model": "example-model",
  "document": {
    "title": "The Left Hand of Darkness",
    "review": "Ambitious, humane, and beautifully written."
  },
  "questions": {
    "positive": {
      "type": "noul",
      "instructions": "The review is positive."
    },
    "rating": {
      "type": "score",
      "instructions": "Rate the reviewer's overall assessment.",
      "criteria": [
        "Strongly negative",
        "Negative",
        "Mixed",
        "Positive",
        "Strongly positive"
      ]
    },
    "genre": {
      "type": "choice",
      "instructions": "Choose the best genre.",
      "criteria": {
        "fiction": "A novel or short story",
        "nonfiction": "A factual work"
      }
    }
  }
}
```

The `questions` object MUST contain at least one property. Each property name is an
opaque question ID chosen by the caller. IDs MUST be unique within the request, as
required for JSON object keys, and SHOULD be non-empty.

Core request objects do not define extension fields. Portable clients MUST NOT depend
on undeclared request fields. Implementations MAY accept private request extensions,
but SHOULD reject unknown fields by default so misspellings do not silently change an
evaluation.

## Questions

Every question is an object with a required `type` discriminator. Question order has
no semantic meaning.

### Noul question

A Noul question represents a yes/no question or a claim whose truth is uncertain.
"Noul" is the stable wire name for this question type.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `type` | literal `"noul"` | yes | Question discriminator |
| `instructions` | instruction value or `null` | no | Question or assertion to evaluate |
| `criteria` | Noul criteria or `null` | no | Optional definitions of true and false |

Noul criteria is an object with optional `true` and `false` fields. Each field is an
instruction value or `null`.

```json
{
  "type": "noul",
  "instructions": "The customer wants to cancel.",
  "criteria": {
    "true": "An explicit request to stop or cancel the service",
    "false": "A request to change, pause, or ask about the service"
  }
}
```

### Choice question

A Choice question selects one label from a closed set.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `type` | literal `"choice"` | yes | Question discriminator |
| `instructions` | instruction value or `null` | no | What should be classified |
| `criteria` | object | yes | Map from label to its definition |

`criteria` MUST contain at least two labels. Labels are opaque strings, MUST be unique,
and SHOULD be short and stable. Criterion values are instruction values or `null`.

```json
{
  "type": "choice",
  "instructions": "Route this request.",
  "criteria": {
    "sales": "Pricing, trials, and purchasing",
    "support": "Help with an existing account",
    "other": null
  }
}
```

### Score question

A Score question selects an ordinal rubric level. Array position defines the numeric
level, starting at zero.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `type` | literal `"score"` | yes | Question discriminator |
| `instructions` | instruction value or `null` | no | What should be scored |
| `criteria` | array | yes | Ordered rubric levels from `0` upward |

`criteria` MUST contain at least two instruction values. For `N` criteria, the valid
levels are the integers `0` through `N - 1`.

```json
{
  "type": "score",
  "instructions": "Assess urgency.",
  "criteria": [
    "No action needed",
    "Routine follow-up",
    "Prompt action",
    "Immediate action"
  ]
}
```

## Evaluation semantics

The document is data, not an instruction channel. An implementation MUST evaluate it
according to the caller's questions and MUST NOT follow instructions embedded in the
document that attempt to alter the evaluation, reveal secrets, or redefine the output.

For every question:

- The answer MUST be based only on the supplied document, question instructions,
  criteria, and the implementation's declared evaluation policy.
- The response MUST contain exactly one answer under the same question ID.
- The answer `type` MUST equal the corresponding question `type`.
- A probability MUST be between `0` and `1`, inclusive.
- A complete probability distribution SHOULD sum to `1` within an absolute tolerance
  of `0.000001`.

Implementations SHOULD preserve genuine uncertainty rather than forcing confidence.
Implementations that can emit a distribution outside the recommended sum tolerance
SHOULD provide a normalization option or diagnostics.

## Response

A successful response has three required core fields and can contain extensions:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `model` | string | yes | Model identifier actually used |
| `answers` | object | yes | Non-empty map from question ID to answer |
| `usage` | object | yes | Token usage known to the implementation |

Example:

```json
{
  "model": "example-model",
  "answers": {
    "positive": {
      "type": "noul",
      "noul": 0.98
    },
    "rating": {
      "type": "score",
      "score": 3.85,
      "confidence": 0.83,
      "probabilities": {
        "0": 0.0,
        "1": 0.01,
        "2": 0.04,
        "3": 0.04,
        "4": 0.91
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

`answers` MUST have exactly the same keys as the request's `questions`. Consumers
MUST match answers by question ID rather than object order.

Response objects are extensible. Consumers MUST ignore fields they do not understand
in the top-level response, answer objects, and usage object. Extensions MUST NOT
change the meaning of core fields.

## Answers

### Noul answer

```json
{
  "type": "noul",
  "noul": 0.98
}
```

- `type` MUST be `"noul"`.
- `noul` MUST be the probability that the answer is yes or that the assertion is true.
- `0` means no or false, `0.5` represents maximum binary uncertainty, and `1` means
  yes or true.

### Choice answer

```json
{
  "type": "choice",
  "choice": "fiction",
  "confidence": 0.76,
  "probabilities": {
    "fiction": 0.88,
    "nonfiction": 0.12
  }
}
```

- `type` MUST be `"choice"`.
- `probabilities` MUST contain every label from the question's `criteria` and MUST NOT
  contain any other label.
- `choice` MUST be one of the labels with the greatest probability. An implementation
  MAY choose any greatest-probability label when there is a tie.
- `confidence` MUST be a number from `0` to `1`, or `null` when unavailable. It is an
  implementation-defined summary of certainty and MUST NOT replace the distribution.

### Score answer

```json
{
  "type": "score",
  "score": 3.85,
  "confidence": 0.83,
  "probabilities": {
    "0": 0.0,
    "1": 0.01,
    "2": 0.04,
    "3": 0.04,
    "4": 0.91
  }
}
```

For a question with `N` criteria:

- `type` MUST be `"score"`.
- `probabilities` MUST contain exactly the string keys `"0"` through `"N - 1"`.
- `score` MUST be the expected rubric level after rescaling the probability weights
  to sum to `1`, and MUST therefore be between `0` and `N - 1`. If every weight is
  zero, each level receives weight `1 / N` for this calculation.
- `confidence` follows the same requirements as Choice confidence.

An implementation MAY add a `legend` extension mapping level keys back to criteria.

## Usage

The usage object has two portable fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input_tokens` | non-negative integer or `null` | no | Input tokens for the reported evaluation |
| `output_tokens` | non-negative integer or `null` | no | Output tokens for the reported evaluation |

Implementations SHOULD include both fields when the underlying model reports them and
MAY add provider-specific or aggregate usage fields. They MUST document whether token
counts cover only the final attempt or all attempts.

SystemOneClientAdapter currently adds:

- `input_tokens_total` and `output_tokens_total` across all attempts;
- `n_retries` for transient provider retries;
- `n_retries_malformed_structure` for output-validation retries; and
- `latency` for end-to-end seconds.

These are extensions, not requirements for core conformance.

## Errors

Invalid requests MUST fail rather than returning invented question IDs or partial
answers. Implementations SHOULD distinguish at least:

- invalid request or question definitions;
- authentication or authorization failure;
- timeout or connection failure;
- document or context-window size exceeded; and
- unknown provider or internal failure.

The TypeSafe HTTP profile represents an error as:

```json
{
  "detail": {
    "message": "Human-readable explanation",
    "error_type": "tokens_exceeded"
  }
}
```

Known `error_type` values are `authentication_error`, `timeout`, `tokens_exceeded`,
and `unknown`. Clients MUST tolerate an absent or unrecognized `error_type` and SHOULD
preserve the human-readable message.

Python profile implementations SHOULD map failures to `TypeSafeApiError` and its
public subclasses `TypeSafeAuthError`, `TypeSafeTimeoutError`,
`TypeSafeTokensExceededError`, and `TypeSafeUnknownError` when they depend on
`typesafe-client`.

## HTTP profile

The TypeSafe-compatible HTTP operation is:

```http
POST /v1/systemone HTTP/1.1
Content-Type: application/json
Authorization: Bearer <token>
```

The request body is a conforming System One request. A successful response uses a 2xx
status and a conforming System One response body. An error uses a non-2xx status and
SHOULD use the error shape above.

An implementation MAY use different authentication, host, timeouts, and rate limits.
It MUST document those choices. Implementations that advertise the HTTP profile MUST
retain the method, path, and JSON body contract.

## Python profile

A Python-compatible client provides synchronous and asynchronous operations with the
following effective signatures:

```python
def system_one(
    model: str,
    document: InstructionValue,
    questions: Mapping[str, Question],
) -> SystemOneResponse: ...

async def system_one_async(
    model: str,
    document: InstructionValue,
    questions: Mapping[str, Question],
) -> SystemOneResponse: ...
```

It also provides:

- `close()` and `aclose()` lifecycle methods;
- synchronous and asynchronous context-manager support;
- question and response objects compatible with `typesafe_client.api.models`; and
- dictionary question inputs using the wire shapes in this specification.

The synchronous and asynchronous operations MUST have the same request semantics,
response semantics, and error categories. Implementations MAY accept richer model
objects or provider-qualified model names as extensions.

## Discrete evaluation extensions

Some implementations offer a discrete evaluation mode. In that mode:

- a Noul selection maps `false` to `0.0` and `true` to `1.0`;
- a Choice selection maps to a one-hot probability distribution; and
- a Score selection maps to a one-hot distribution with `score` equal to the selected
  level.

Discrete mode changes how answers are produced, not their response shape.

## Security and privacy

- Implementations MUST treat the document as untrusted data and SHOULD isolate it
  from system or developer instructions used by an underlying model.
- Implementations MUST NOT expose credentials in responses, errors, logs, or debug
  extensions.
- Implementations SHOULD document retention and provider-subprocessing behavior.
- Callers SHOULD assume that optional debug data can contain the full document,
  question instructions, provider responses, and model metadata.
- Services SHOULD apply request-size, question-count, and rate limits appropriate to
  their deployment and report limit failures explicitly.

## Extensions and versioning

The v1 core is intentionally additive:

- Response producers MAY add fields.
- Response consumers MUST ignore unknown fields.
- Request producers MUST NOT send undeclared fields unless they have negotiated an
  implementation extension.
- A new optional field does not require a new major version.
- Removing, renaming, or changing the meaning or type of a core field requires a new
  major version.
- New question or answer types require a new specification version or an explicitly
  negotiated extension because v1 consumers dispatch on `type`.

The HTTP `/v1/systemone` path and the schema filenames identify the v1 compatibility
family. This draft can be clarified without changing behavior; incompatible changes
will be proposed as a new version.

## Implementation checklist

A conforming implementation should be able to answer yes to each item:

- Does it accept string, array, and object documents?
- Does it reject an empty question collection?
- Does it support Noul, Choice, and Score questions with the exact discriminators?
- Does it preserve every question ID without reinterpretation?
- Does a successful response contain exactly one correctly typed answer per question?
- Are Noul and outcome probabilities bounded by `0` and `1`?
- Do Choice and Score probability keys exactly match the declared outcomes?
- Is Choice derived from a maximum-probability label?
- Is Score the expected value of the level distribution?
- Does the document remain data even when it contains instruction-like text?
- Do sync and async entry points behave equivalently, when both are provided?
- Are unknown response extensions tolerated by consumers?
- Are authentication, timeout, size, validation, and unknown failures distinguishable?

Compatibility tests can validate structural conformance with the included JSON
Schemas. Cross-field invariants—matching question IDs, matching outcome labels,
probability sums, Choice argmax, and Score expected value—must also be checked because
JSON Schema cannot express them relative to a separate request.

## Feedback and compatible implementations

Issues and pull requests are welcome in this repository. Compatible implementations
are encouraged to link to this specification, state their conformance profiles and
extensions, and contribute interoperability fixtures or clarifications discovered
while implementing it.
