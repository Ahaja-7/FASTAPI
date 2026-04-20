import argparse
import json
from json import JSONDecodeError
from pathlib import Path

from transformers import AutoTokenizer

from train_eval_sms_lora import DEFAULT_MODEL, ROOT, build_prompt, load_base_model, parse_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--test-split-file", default=str(ROOT / "data" / "sms_spam_test.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--show-errors", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    decoder = json.JSONDecoder()
    content = path.read_text(encoding="utf-8")
    rows = []
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        try:
            row, index = decoder.raw_decode(content, index)
        except JSONDecodeError:
            next_object = content.find("{", index + 1)
            if next_object == -1:
                break
            index = next_object
            continue
        rows.append(row)
    return rows


def print_metrics(predictions: list[tuple[str, str]]) -> None:
    labels = ["ham", "spam"]
    confusion = {(actual, predicted): 0 for actual in labels for predicted in labels + ["unknown"]}

    for actual, predicted in predictions:
        confusion[(actual, predicted)] = confusion.get((actual, predicted), 0) + 1

    correct = sum(1 for actual, predicted in predictions if actual == predicted)
    total = len(predictions)
    accuracy = correct / total if total else 0.0

    print("\nBase Qwen Evaluation")
    print(f"test_samples: {total}")
    print(f"accuracy: {accuracy:.4f}")

    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in labels if other != label)
        fn = sum(confusion[(label, other)] for other in labels + ["unknown"] if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    print("confusion:")
    for actual in labels:
        values = ", ".join(f"pred_{predicted}={confusion[(actual, predicted)]}" for predicted in labels + ["unknown"])
        print(f"  actual_{actual}: {values}")


def main() -> None:
    args = parse_args()
    test_rows = load_jsonl(Path(args.test_split_file))
    rows = test_rows[: args.eval_limit] if args.eval_limit else test_rows

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _dtype = load_base_model(args.model_name)
    model.eval()

    import torch

    predictions = []
    errors = []
    for row in rows:
        inputs = tokenizer(build_prompt(row["message"]), return_tensors="pt").to(model.device)
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
            errors.append((actual, predicted, generated_text, row["message"]))

    print_metrics(predictions)


    if errors:
        print("\nSample errors:")
        for actual, predicted, generated_text, message in errors:
            print(f"- actual={actual} predicted={predicted} generated={generated_text!r} sms={message!r}")


if __name__ == "__main__":
    main()
