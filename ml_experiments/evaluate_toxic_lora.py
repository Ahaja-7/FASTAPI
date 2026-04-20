import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from peft import PeftModel
from transformers import AutoTokenizer

from train_toxic_lora_qwen import DEFAULT_MODEL, ROOT, build_prompt, load_base_model


DEFAULT_VALIDATION_CSV = r"\\wsl.localhost\Ubuntu-22.04\home\jekal\data\tvalidation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--data-file", default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--adapter-dir", default=str(ROOT / "outputs" / "toxic_qwen3_0_6b_lora"))
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--show-errors", type=int, default=10)
    return parser.parse_args()


def resolve_wsl_path(path: str) -> Path:
    if path.startswith(r"\\wsl.localhost\Ubuntu-22.04"):
        return Path(path.replace(r"\\wsl.localhost\Ubuntu-22.04", "").replace("\\", "/"))
    return Path(path)


def existing_data_file(path: Path) -> Path:
    if path.exists():
        return path

    fallback = path.with_name("validation.csv")
    if path.name == "tvalidation.csv" and fallback.exists():
        print(f"Data file not found: {path}")
        print(f"Using fallback instead: {fallback}")
        return fallback

    raise FileNotFoundError(f"Data file not found: {path}")


def normalize_label(value: str) -> str:
    value = str(value).strip()
    if value == "1":
        return "toxic"
    if value == "0":
        return "non-toxic"
    raise ValueError(f"Unsupported toxic label: {value!r}")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            comment = row["comment_text"].strip()
            if comment:
                rows.append({"comment_text": comment, "label": normalize_label(row["toxic"])})
    return rows


def parse_prediction(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if re.search(r"\bnon[- ]?toxic\b", text):
        return "non-toxic"
    if re.search(r"\btoxic\b", text):
        return "toxic"
    return "unknown"


def print_metrics(predictions: list[tuple[str, str]]) -> None:
    labels = ["non-toxic", "toxic"]
    predicted_labels = labels + ["unknown"]
    confusion = {(actual, predicted): 0 for actual in labels for predicted in predicted_labels}

    for actual, predicted in predictions:
        confusion[(actual, predicted)] = confusion.get((actual, predicted), 0) + 1

    correct = sum(1 for actual, predicted in predictions if actual == predicted)
    total = len(predictions)
    print("\nToxic LoRA Validation")
    print(f"validation_samples: {total}")
    print(f"accuracy: {correct / total if total else 0.0:.4f}")

    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in labels if other != label)
        fn = sum(confusion[(label, other)] for other in predicted_labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    print("confusion:")
    for actual in labels:
        values = ", ".join(f"pred_{predicted}={confusion[(actual, predicted)]}" for predicted in predicted_labels)
        print(f"  actual_{actual}: {values}")


def main() -> None:
    args = parse_args()
    data_file = existing_data_file(resolve_wsl_path(args.data_file))
    rows = load_rows(data_file)
    rows = rows[: args.eval_limit] if args.eval_limit else rows

    print(f"Loaded validation rows: {len(rows)} {dict(Counter(row['label'] for row in rows))}")
    print(f"Adapter dir: {args.adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _dtype = load_base_model(args.model_name)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    import torch

    predictions = []
    errors = []
    for index, row in enumerate(rows, start=1):
        inputs = tokenizer(build_prompt(row["comment_text"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        predicted = parse_prediction(generated_text)
        actual = row["label"]
        predictions.append((actual, predicted))

        if actual != predicted and len(errors) < args.show_errors:
            errors.append((index, actual, predicted, generated_text, row["comment_text"]))

        if index % 100 == 0:
            print(f"Evaluated {index}/{len(rows)}")

    print_metrics(predictions)

    if errors:
        print("\nSample errors:")
        for index, actual, predicted, generated_text, comment in errors:
            compact_comment = " ".join(comment.split())
            print(f"- row={index} actual={actual} predicted={predicted} generated={generated_text!r}")
            print(f"  comment={compact_comment[:300]!r}")


if __name__ == "__main__":
    main()
