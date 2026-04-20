# Qwen3-0.6B LoRA + RAG experiment

This folder is intentionally separate from the FastAPI app so you can test
fine-tuning and RAG before wiring it into an API endpoint.

## 1. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If PyTorch needs a CUDA-specific wheel, install the correct wheel from the
PyTorch site first, then run the command above.

## 2. Edit training data

Start with:

```text
ml_experiments/data/train.jsonl
```

Each row is one supervised example:

```json
{"instruction": "question", "input": "optional context", "output": "expected answer"}
```

Replace the sample rows with examples that match your domain and answer style.

## 3. Train LoRA adapter

```powershell
.\.venv\Scripts\python.exe ml_experiments\train_lora_qwen.py
```

Default output:

```text
ml_experiments/outputs/qwen3_0_6b_lora
```

For a quick CPU smoke test:

```powershell
.\.venv\Scripts\python.exe ml_experiments\train_lora_qwen.py --max-steps 1 --per-device-train-batch-size 1
```

## 4. Run RAG with base model

```powershell
.\.venv\Scripts\python.exe ml_experiments\rag_qwen_demo.py --question "timeout 오류가 많이 나는 원인은?"
```

To check retrieval only without loading Qwen:

```powershell
.\.venv\Scripts\python.exe ml_experiments\rag_qwen_demo.py --retrieval-only --question "timeout 오류가 많이 나는 원인은?"
```

## 5. Run RAG with LoRA adapter

```powershell
.\.venv\Scripts\python.exe ml_experiments\rag_qwen_demo.py --adapter-path ml_experiments\outputs\qwen3_0_6b_lora --question "timeout 오류가 많이 나는 원인은?"
```

The script prints retrieved documents and the model answer, so you can compare
base Qwen3-0.6B vs LoRA-tuned Qwen3-0.6B on the same RAG context.

## 6. Train and evaluate SMS spam LoRA

This reads:

```text
\\wsl.localhost\Ubuntu-22.04\home\jekal\data\spam.csv
```

It creates stratified train/test splits, trains only on the train split, then
evaluates the LoRA adapter on the test split.

```powershell
.\.venv\Scripts\python.exe ml_experiments\train_eval_sms_lora.py
```

Generated split files:

```text
ml_experiments\data\sms_spam_train.jsonl
ml_experiments\data\sms_spam_test.jsonl
```

Default adapter output:

```text
ml_experiments\outputs\sms_qwen3_0_6b_lora
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python.exe ml_experiments\train_eval_sms_lora.py --max-steps 1 --per-device-train-batch-size 1 --eval-limit 20
```

To create only the train/test split files without loading the model:

```powershell
.\.venv\Scripts\python.exe ml_experiments\train_eval_sms_lora.py --prepare-only
```

To evaluate an already trained adapter without training again:

```powershell
.\.venv\Scripts\python.exe ml_experiments\train_eval_sms_lora.py --skip-train
```

## Notes

- Default model: `Qwen/Qwen3-0.6B`
- Default embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Qwen3-0.6B is small enough to test locally, but training is still much faster
  on a CUDA GPU.
- The demo uses local FAISS instead of Qdrant to keep the experiment runnable
  without external services.
- The training script uses `transformers.Trainer` directly instead of `trl`
  because recent `trl` releases can fail on Windows cp949 locales while reading
  bundled UTF-8 templates.
