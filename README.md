# call me maybe

*This project has been created as part of the 42 curriculum by mel-bakh.*

---

## Description

A function calling tool that translates natural language prompts into structured
JSON function calls using a small LLM (Qwen3-0.6B) with constrained decoding.

Given a prompt like `"What is the sum of 40 and 2?"`, the system outputs:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 40, "b": 2}
}
```

Instead of answering the question directly, it identifies the right function
and extracts the correct arguments — bridging natural language and executable code.

---

## Instructions

**Requirements:** Python 3.10+, `uv`

**Install dependencies:**
```bash
uv sync
```

**Run:**
```bash
uv run python -m src
```

**Run with custom paths:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**Lint:**
```bash
make lint
```

---

## Algorithm Explanation

The core technique is **constrained decoding** — guiding the model token by token
instead of letting it generate freely.

**Steps:**

1. Build a system prompt listing all available functions and their parameters.
2. Force the output to start with `{"name": "` — the model never sees a blank slate.
3. For the function name field, compute which tokens are valid at each step by
   comparing what has been generated so far against all known function name token
   sequences. Only tokens that continue a valid name are allowed.
4. Once the name is complete, switch to free generation but restricted to a
   filtered vocabulary containing only JSON-safe characters.
5. Stop when the output ends with `}}`.

This guarantees **100% valid JSON** regardless of model confidence.

---

## Design Decisions

- **Pydantic for validation** — all input files are validated through `Function`
  and `Prompt` models before any processing begins, giving clear error messages
  on malformed input.
- **Prefix forcing** — injecting `{"name": "` as forced tokens means the model
  never has to decide how to start the JSON, removing a major failure point.
- **Vocabulary filtering** — restricting free generation to JSON-relevant characters
  prevents the model from generating markdown, prose, or other non-JSON content.
- **Null function** — `"null"` is treated as a valid function name in the
  constrained decoding phase, so the model can cleanly signal no match.

---

## Challenges Faced

- **Escape handling:** Regex parameters like `\d+` required special handling
  to survive JSON serialization without becoming `\\d+` in the output.
- **Tokenizer vocabulary:** Filtering the vocabulary to only JSON-safe tokens
  required understanding how Qwen3's tokenizer represents special characters.
- **Name token matching:** Function names can tokenize into multiple tokens,
  so the constrained decoding had to track partial matches across steps.



## Resources

- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained Decoding — Outlines library concepts](https://github.com/outlines-dev/outlines)
- [Pydantic documentation](https://docs.pydantic.dev)
- [uv documentation](https://docs.astral.sh/uv/)

**AI usage:** Claude was used to help add type hints and docstrings to the
existing code, fix flake8 violations, and review the constrained decoding logic
for correctness. All core algorithm design and implementation decisions were
made by the author.