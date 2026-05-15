# Cutting-Out-the-Middleman

This project contains a small NLP training stack for GPT-2. It includes supervised fine-tuning and direct preference optimization utilities, plus an example notebook that shows how the modules fit together.


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