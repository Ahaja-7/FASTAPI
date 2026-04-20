import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parent


def build_prompt(example: dict) -> str:
    user_parts = [example["instruction"].strip()]
    if example.get("input"):
        user_parts.append(f"참고 자료:\n{example['input'].strip()}")
    user_content = "\n\n".join(user_parts)

    return (
        "<|im_start|>system\n"
        "너는 한국어로 답하는 API 로그/RAG 분석 도우미다. "
        "제공된 근거만 사용하고, 근거가 부족하면 부족하다고 말한다. "
        "추론 과정은 출력하지 말고 최종 답변만 한국어로 작성한다."
        "<|im_end|>\n"
        f"<|im_start|>user\n/no_think\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output'].strip()}<|im_end|>"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=str(ROOT / "data" / "train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "qwen3_0_6b_lora"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.map(lambda row: {"text": build_prompt(row)})

    def tokenize(row: dict) -> dict:
        return tokenizer(
            row["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="epoch" if args.max_steps == -1 else "steps",
        save_steps=10,
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        report_to="none",
        remove_unused_columns=False,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
