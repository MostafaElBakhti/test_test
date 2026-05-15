import argparse
import json
import os
from argparse import Namespace

from llm_sdk import Small_LLM_Model

from .models.decoding import (
    filter_vocab,
    get_best_next_id,
    get_valid_name_tokens,
    load_vocabulary,
    system_prompt_builder,
)
from .models.validation import function_definition_check, prompt_check


def parse_arguments() -> Namespace:
    """Parse command-line arguments for the function calling pipeline.

    Returns:
        Namespace object containing parsed arguments.
    """
    parse = argparse.ArgumentParser(description="prompt to json")

    parse.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
    )
    parse.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parse.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
    )
    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
    )

    return parse.parse_args()


def main() -> None:
    """Run the function calling pipeline.

    Loads function definitions and prompts, runs constrained decoding
    with the LLM, and writes structured JSON results to the output file.
    """
    print("starting...")
    args = parse_arguments()

    functions = function_definition_check(args.functions_definition)
    if not functions:
        raise RuntimeError(
            f"function definitions file is empty, "
            f"please check {args.functions_definition}"
        )

    prompts = prompt_check(args.input)
    if not prompts:
        raise RuntimeError(
            f"prompt file is empty, please check {args.input}"
        )

    sys_prompt = system_prompt_builder(functions)

    print(f"loading model ... {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{args.model}': {e}")
    print("Model loaded successfully.")

    functions_names = [f.name for f in functions]
    print(f"Available functions: {functions_names}")

    valid_name_ids: dict[str, list[int]] = {}
    for name in functions_names:
        ids: list[int] = model.encode(name)[0].tolist()
        valid_name_ids[name] = ids
    valid_name_ids["null"] = model.encode("null")[0].tolist()

    id_to_token: dict[int, str] = load_vocabulary(model)
    valid_vocab: dict[int, str] = filter_vocab(id_to_token)

    results: list[dict] = []

    for p in prompts:
        if not p.prompt or not p.prompt.strip():
            results.append(
                {"prompt": p.prompt, "name": None, "parameters": {}}
            )
            continue

        full_prompt = f"{sys_prompt}\n\nUser: {p.prompt}\nAssistant:"
        full_prompt_ids: list[int] = model.encode(full_prompt)[0].tolist()

        forced = '{"name": "'
        forced_ids: list[int] = model.encode(forced)[0].tolist()
        all_generated: list[int] = list(forced_ids)

        generated_name_ids: list[int] = []
        name_complete = False

        for _ in range(100):
            current_input = full_prompt_ids + all_generated
            logits = model.get_logits_from_input_ids(current_input)

            if not name_complete:
                valid_next = get_valid_name_tokens(
                    generated_name_ids, valid_name_ids
                )
                next_id = get_best_next_id(logits, valid_next)
                generated_name_ids.append(next_id)
                all_generated.append(next_id)

                if generated_name_ids == valid_name_ids["null"]:
                    results.append(
                        {"prompt": p.prompt, "name": None, "parameters": {}}
                    )
                    break

                if generated_name_ids in list(valid_name_ids.values()):
                    closing_id: int = model.encode('"')[0].tolist()[0]
                    all_generated.append(closing_id)
                    name_complete = True
                    continue
            else:
                next_id = get_best_next_id(logits, valid_vocab)
                all_generated.append(next_id)

            generated_text: str = model.decode(all_generated)
            if generated_text.strip().endswith("}}"):
                try:
                    cleaned = (
                        generated_text.strip()
                        .replace('\\d', '\\\\d')
                        .replace('\\s', '\\\\s')
                    )
                    parsed = json.loads(cleaned)
                    function_def = next(
                        (f for f in functions if f.name == parsed["name"]),
                        None,
                    )
                    if function_def:
                        valid_params = function_def.parameters.keys()
                        parsed["parameters"] = {
                            k: v
                            for k, v in parsed["parameters"].items()
                            if k in valid_params
                        }
                    if parsed["name"] is None:
                        results.append(
                            {
                                "prompt": p.prompt,
                                "name": None,
                                "parameters": {},
                            }
                        )
                    else:
                        results.append(
                            {
                                "prompt": p.prompt,
                                "name": parsed["name"],
                                "parameters": parsed["parameters"],
                            }
                        )
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: failed to parse JSON "
                        f"for prompt '{p.prompt}': {e}"
                    )
                break
        else:
            print(
                f"Warning: max tokens reached for prompt '{p.prompt}'"
            )
            results.append(
                {"prompt": p.prompt, "name": None, "parameters": {}}
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            print(f"Results saved to {args.output}")
    except OSError as e:
        print(f"Failed to write results to {args.output}: {e}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"An error occurred: {e}")
