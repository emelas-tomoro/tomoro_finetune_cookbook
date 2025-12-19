## ar-finetune (tomoro_finetune_cookbook)

This repo packages the notebook workflows in `notebooks_sl/` into a local-installable Python CLI.

### Install (with `uv`)

From the repo root:

```bash
uv tool install .

# verify
ar-ft-make-conversations --help
ar-ft-tinker-finetune --help
ar-ft-tinker-infer --help
ar-ft-gpt-benchmark --help
```

For development / editable installs:

```bash
uv venv
uv pip install -e .
```

### Install from GitHub (no local clone)

Repo: [`emelas-tomoro/tomoro_finetune_cookbook`](https://github.com/emelas-tomoro/tomoro_finetune_cookbook)

Install directly from git (recommended for users):

```bash
uv tool install "git+https://github.com/emelas-tomoro/tomoro_finetune_cookbook.git"
```

Pin to a branch / tag / commit:

```bash
uv tool install "git+https://github.com/emelas-tomoro/tomoro_finetune_cookbook.git@main"
```

For private repos, use SSH:

```bash
uv tool install "git+ssh://git@github.com/emelas-tomoro/tomoro_finetune_cookbook.git@main"
```

### Commands

#### `ar-ft-make-conversations`
Build `training_data.json` (and optionally Tinker conversation JSONL).

```bash
ar-ft-make-conversations \
  --input data/finetuning/lora_train_emails.parquet \
  --system-prompt-path data/verification/system_prompt_spam.md \
  --training-json-out data/finetuning/training_data_spam.json \
  --tinker-jsonl-out data/finetuning/tinker_conversations_spam.jsonl
```

If you want the notebook-style balancing + train/test split:

```bash
ar-ft-make-conversations \
  --input /path/to/Enron.csv \
  --system-prompt-path data/verification/system_prompt_spam.md \
  --make-split \
  --train-out data/finetuning/lora_train_emails.parquet \
  --test-out data/finetuning/lora_test_emails.parquet \
  --training-json-out data/finetuning/training_data_spam.json \
  --tinker-jsonl-out data/finetuning/tinker_conversations_spam.jsonl
```

#### `ar-ft-tinker-finetune`
Convert `training_data.json` → Tinker JSONL and run supervised fine-tuning.

```bash
ar-ft-tinker-finetune \
  --training-json data/finetuning/training_data_spam.json \
  --conversations-jsonl data/finetuning/tinker_conversations_spam.jsonl \
  --log-path /tmp/tinker-examples/sl_ar_phishing_spam
```

Dry run (no training):

```bash
ar-ft-tinker-finetune \
  --training-json data/finetuning/training_data_spam.json \
  --conversations-jsonl data/finetuning/tinker_conversations_spam.jsonl \
  --dry-run
```

#### `ar-ft-tinker-infer`
Batch inference using either a remote Tinker sampler checkpoint (default) or a local LoRA adapter.

Remote Tinker sampler (reads the last `sampler_path` under `--log-path`):

```bash
ar-ft-tinker-infer \
  --input data/finetuning/lora_test_emails.parquet \
  --output data/verification/tinker_results_spam.parquet \
  --system-prompt-path data/verification/system_prompt_spam.md \
  --log-path /tmp/tinker-examples/sl_ar_phishing_spam
```

Local adapter mode:

```bash
ar-ft-tinker-infer \
  --input data/finetuning/lora_test_emails.parquet \
  --output data/verification/local_lora_results_spam.parquet \
  --system-prompt-path data/verification/system_prompt_spam.md \
  --use-local-adapter \
  --base-model-dir ../models/llama-3.1-8b \
  --adapter-dir ../adapters/lora_adapter_spam
```

#### `ar-ft-gpt-benchmark`
Runs a GPT benchmark over a dataset (requires `OPENAI_API_KEY`).

```bash
export OPENAI_API_KEY="..."

ar-ft-gpt-benchmark \
  --input data/finetuning/lora_test_emails.parquet \
  --output data/verification/gpt_results_spam.parquet \
  --system-prompt-path data/verification/system_prompt_spam.md
```

### Notes

- The CLIs expect datasets with `subject`, `body`, and `label` columns by default (configurable via `--*-col` args).
- `tinker-cookbook` is pulled via `[tool.uv.sources]` (git dependency).
