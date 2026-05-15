# Source files

This folder contains the three core Python modules used by the project.

## `model.py`

Brief: GPT-2 wrapper used for policy and reference models.

Run it:

```bash
python3 -m py_compile src/model.py
```

## `dpo_utils.py`

Brief: Utility functions for token log probabilities and DPO loss.

Run it:

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); import dpo_utils"
```

## `trainer.py`

Brief: Training loop for SFT and DPO.

Run it:

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); import trainer"
```

## Notes

- The modules are meant to be imported from notebooks or scripts.
- See the root `README.md` for the main project overview and example notebook instructions.

### Task 3: Sentiment Classification

- Notebook: [Task_3_Sentiment_Classification.ipynb](Task_3_Sentiment_Classification.ipynb)
- Purpose: Train and evaluate transformer-based sentiment classifiers on SST2 and SST datasets.

Quick run instructions (use the project venv so the notebook imports match the tested environment):

1. Activate the project environment and install dependencies:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

2. Start Jupyter and open the notebook:

```bash
python -m notebook src/Task_3_Sentiment_Classification.ipynb
```

3. Run the notebook cells sequentially. The notebook:
 - Loads SST2 via `pandas.read_parquet` (hf:// datasets path) as requested.
 - Loads SST by downloading and parsing the original Stanford source archive, because the current `datasets` version does not support the `stanfordnlp/sst` loading script.
 - Tokenizes and fine-tunes a baseline `distilbert-base-uncased` model and evaluates results.

Notes:
 - If `hf://` parquet access is unavailable, replace the SST2 loading cells with `load_dataset('stanfordnlp/sst2')`.
 - GPU is recommended; reduce `per_device_train_batch_size` if memory is limited.
- If imports fail inside Jupyter, make sure the kernel is the project venv Python at `venv/bin/python`.

