"""Generate rewrite or local decoherence attacks for JSON datasets."""

import argparse
import json
import os
import random

from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of records")
    return data


def save_json(data, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def rewrite_attack(data, model_name, base_url):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DS_DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set DS_DEEPSEEK_API_KEY before running rewrite mode")

    client = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    output = []
    for item in tqdm(data, desc="Rewrite attack"):
        text = str(item.get("text", ""))
        response = client.invoke(
            "Rewrite the following English text while preserving its meaning. "
            "Return only the rewritten text:\n\n" + text
        )
        updated = dict(item)
        updated["text"] = response.content.strip()
        output.append(updated)
    return output


def decohere_text(text, swap_threshold, rng):
    words = str(text).split()
    if len(words) <= swap_threshold:
        return str(text)
    index = rng.randrange(len(words) - 1)
    words[index], words[index + 1] = words[index + 1], words[index]
    return " ".join(words)


def decoherence_attack(data, swap_threshold, seed):
    rng = random.Random(seed)
    output = []
    for item in tqdm(data, desc="Decoherence attack"):
        updated = dict(item)
        updated["text"] = decohere_text(
            item.get("text", ""), swap_threshold=swap_threshold, rng=rng
        )
        output.append(updated)
    return output


def main(args):
    data = load_json(args.input_path)
    if args.mode == "rewrite":
        output = rewrite_attack(data, args.model, args.base_url)
    else:
        output = decoherence_attack(data, args.swap_threshold, args.seed)
    save_json(output, args.output_path)
    print(f"Saved {len(output)} records to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rewrite", "decoherence"), required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--swap_threshold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
