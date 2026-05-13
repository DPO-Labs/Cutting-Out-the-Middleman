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
