# tomoro_finetune_cookbook

This repo contains notebooks + a small Streamlit UI to:

- Build / select a dataset
- Fine-tune a model (via Tinker supervised fine-tuning)
- Run inference and compute metrics

## Quickstart

```bash
# from repo root
uv sync

# start the UI
uv run streamlit run streamlit_app.py
```

## Notes

- **Tinker training/inference** requires `TINKER_API_KEY` in your environment (or a `.env` file).
- The UI reuses the same file conventions as the notebooks under `notebooks_sl/`:
  - `data/finetuning/training_data_*.json`
  - `data/finetuning/tinker_conversations_*.jsonl`
  - logs under `/tmp/tinker-examples/...` with `checkpoints.jsonl`


