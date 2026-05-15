import json

from llm_sdk import Small_LLM_Model

from .validation import Function


def get_best_next_id(
    logits: list[float],
    valid_vocab: dict[int, str],
) -> int:
    """Get the best next token id from valid tokens only.

    Args:
        logits: Raw scores from the model for each token in vocabulary.
        valid_vocab: Mapping of allowed token IDs to their string values.

    Returns:
        The token ID with the highest logit score among valid tokens.
    """
    best_score = float('-inf')
    best_id = -1

    for token_id in valid_vocab:
        if logits[token_id] > best_score:
            best_score = logits[token_id]
            best_id = token_id

    return best_id


def filter_vocab(id_to_token: dict[int, str]) -> dict[int, str]:
    """Keep only tokens relevant to JSON generation.

    Args:
        id_to_token: Full vocabulary mapping token_id -> token_string.

    Returns:
        Filtered dictionary with only JSON-relevant tokens.
    """
    valid_chars = set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '{}":,.-_! '
        '\n'
        '*+?()[]/\\\''
        'ĠĊ'
    )

    allowed: dict[int, str] = {}
    for token_id, token_str in id_to_token.items():
        if token_str and all(c in valid_chars for c in token_str):
            allowed[token_id] = token_str

    return allowed


def load_vocabulary(model: Small_LLM_Model) -> dict[int, str]:
    """Load the vocabulary from the tokenizer file.

    Args:
        model: The loaded Small_LLM_Model instance.

    Returns:
        A dictionary mapping token_id (int) -> token_string (str).

    Raises:
        RuntimeError: If the tokenizer file cannot be read or parsed.
    """
    vocab_path = model.get_path_to_tokenizer_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        vocab: dict[str, int] = (
            tokenizer_data.get("model", {}).get("vocab", {})
        )
    except OSError as e:
        raise RuntimeError(f"Failed to read tokenizer file: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in tokenizer file: {e}")

    return {token_id: token_str for token_str, token_id in vocab.items()}


def system_prompt_builder(functions: list[Function]) -> str:
    """Build the system prompt describing available functions to the LLM.

    Args:
        functions: List of validated Function objects.

    Returns:
        A formatted system prompt string.
    """
    header = (
        "You are an AI assistant that selects the correct function to call.\n"
        "You must ONLY return a JSON object.\n"
        "Do not explain anything.\n\n"
        "Rules:\n"
        "- Use ONLY the provided functions\n"
        "- Use exact function names\n"
        "- Use exact parameter names\n"
        "- Do not add extra fields\n"
        "- For regex patterns:\n"
        '    * "all numbers" or "digits" -> use "\\d+"\n'
        '    * "all vowels" -> use "[aeiouAEIOU]"\n'
        '    * "all letters" -> use "[a-zA-Z]"\n'
        '    * "a specific word" -> use the exact word\n'
        '    * "asterisks" or "star" -> use "*"\n'
        '    * "whitespace" -> use "\\s+"\n'
        "- For string values containing quotes, use \\\" to escape them\n"
        '    * Example: Say "hello" -> "Say \\"hello\\""\n'
        "- If no function matches, return:\n"
        '{"name": null, "parameters": {}}\n'
        "- Never use an unrelated function for a different task\n"
        "- If the user request does NOT match ANY available function,\n"
        "  return null. Never guess. Never force a function.\n"
        "- Do NOT use a function for an unrelated task\n"
        '- "weather", "jokes", "capital cities" are NOT in the available\n'
        "  functions -> return null\n"
        "- Only use the EXACT parameters defined for each function.\n\n"
        "Format:\n"
        "{\n"
        '  "name": "<function_name>",\n'
        '  "parameters": { ... }\n'
        "}\n\n"
        "Available functions:\n"
    )

    functions_str = ""
    for fn in functions:
        functions_str += f"\nFunction name: {fn.name}\n"
        functions_str += f"Description: {fn.description}\n"
        functions_str += "Parameters:\n"
        for name, param in fn.parameters.items():
            functions_str += f"\t- {name} ({param.type})\n"

    functions_str += "\nFunction name: null\n"
    functions_str += (
        "Description: Use this when NO available function "
        "matches the user request\n"
    )
    functions_str += "Parameters:\n"
    functions_str += "\t(none)\n"

    return header + functions_str


def get_valid_name_tokens(
    generated_name_ids: list[int],
    valid_name_ids: dict[str, list[int]],
) -> dict[int, str]:
    """Get valid next token IDs based on what has been generated so far.

    Compares the generated token sequence against all valid function name
    token sequences and returns the set of tokens that could come next.

    Args:
        generated_name_ids: Token IDs generated so far for the name field.
        valid_name_ids: Mapping of function names to their token ID lists.

    Returns:
        Dict of valid next token IDs (compatible with get_best_next_id).
    """
    valid_next: dict[int, str] = {}
    n = len(generated_name_ids)
    for _, ids in valid_name_ids.items():
        if ids[:n] == generated_name_ids and len(ids) > n:
            valid_next[ids[n]] = ""
    return valid_next
