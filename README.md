# Cutting-Out-the-Middleman

This project contains a small NLP training stack for GPT-2. It includes supervised fine-tuning and direct preference optimization utilities, plus an example notebook that shows how the modules fit together.

## Paraphrase training for QQP

Use `src/run_paraphrase.py` for the paraphrase-only workflow.

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run a smoke test

```bash
PARAPHRASE_PROFILE=smoke python3 src/run_paraphrase.py
```

### Run the low-VRAM RTX 3050 profile

```bash
PARAPHRASE_PROFILE=rtx3050 python3 src/run_paraphrase.py
```

### Resume from the last saved checkpoint

```bash
PARAPHRASE_PROFILE=rtx3050 RESUME=1 python3 src/run_paraphrase.py
```

Checkpoints are saved under `checkpoints/paraphrase_<profile>/` and include:
- `sft_last.pt` / `sft_best.pt`
- `dpo_last.pt` / `dpo_best.pt`
- `status.json`
- `run_config.json`

Optional outputs:

```bash
PARAPHRASE_PROFILE=full RUN_TEST_PREDICTIONS=1 python3 src/run_paraphrase.py
PARAPHRASE_PROFILE=rtx3050 RUN_BETA_ABLATION=1 RUN_ERROR_ANALYSIS=1 python3 src/run_paraphrase.py
```

### Visualize smoke vs full checkpoint results

If you have separate checkpoint folders for smoke and full runs, generate comparison plots and findings with:

```bash
python3 src/visualize_paraphrase_runs.py \
  --smoke-dir checkpoints/paraphrase_smoke \
  --full-dir checkpoints/paraphrase_full \
  --output-dir Report/paraphrase_visualization
```

Generated outputs:
- `smoke_vs_full_visualization.png` (multi-panel chart)
- `summary_metrics.csv` (key metrics table)
- `findings.md` (analysis summary)

## How to run the example

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start Jupyter and open `example_usage.ipynb`.


```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook example_usage.ipynb
```
## fish

```fish
source venv/bin/activate.fish
python -m pip install -r requirements.txt
jupyter notebook example_usage.ipynb
```
