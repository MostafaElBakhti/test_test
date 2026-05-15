import json
from typing import Any, Dict

from pydantic import BaseModel, ValidationError


class Parameter(BaseModel):
    """A single function parameter with its type."""

    type: str


class Returns(BaseModel):
    """The return type of a function."""

    type: str


class Function(BaseModel):
    """A callable function definition with metadata.

    Attributes:
        name: The exact function name used in output JSON.
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their Parameter objects.
        returns: The return type descriptor.
    """

    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Returns


class Prompt(BaseModel):
    """A single natural language prompt to process.

    Attributes:
        prompt: The raw user input string.
    """

    prompt: str


def _load_json(path: str, label: str) -> list[dict[str, Any]]:
    """Load and parse a JSON file into a list of dictionaries.

    Args:
        path: Path to the JSON file.
        label: Human-readable label used in error messages.

    Returns:
        Parsed JSON array as a list of dicts.

    Raises:
        RuntimeError: If the file cannot be read, is invalid JSON,
                      or is not a JSON array.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise RuntimeError(
            f"Error occurred while reading {path}: {e}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON format in {path}: {e}"
        )

    if not isinstance(data, list):
        raise RuntimeError(f"{label} file must be a JSON array")

    return data


def function_definition_check(path: str) -> list[Function]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        List of validated Function objects.

    Raises:
        RuntimeError: If loading or validation fails.
    """
    data = _load_json(path, "Function definitions")

    try:
        return [Function.model_validate(item) for item in data]
    except ValidationError as e:
        raise RuntimeError(
            f"Function definition validation error:\n{e}"
        )


def prompt_check(path: str) -> list[Prompt]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        List of validated Prompt objects.

    Raises:
        RuntimeError: If loading or validation fails.
    """
    data = _load_json(path, "Prompts")

    try:
        return [Prompt.model_validate(item) for item in data]
    except ValidationError as e:
        raise RuntimeError(f"Prompt validation error:\n{e}")
