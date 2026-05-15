# Task 3: Sentiment Classification

## Overview

This directory contains implementation of **Task 3 - Sentiment Classification** using transformer-based models on two benchmark datasets.

## Notebook: `Task_3_Sentiment_Classification.ipynb`

### Brief
Comprehensive notebook for sentiment classification using DistilBERT, BERT, and RoBERTa on SST2 and SST datasets.

### Task Goal
Classify text sentiment as **Positive** or **Negative** (and fine-grained sentiment in SST).

**Example:**
- Input: "This movie was amazing and emotionally powerful."
- Output: Positive

### Datasets

#### 1. SST2 (Stanford Sentiment Treebank 2)
- **Type**: Binary classification (Positive/Negative)
- **Size**: 67,349 training samples
- **Task**: Sentence-level sentiment classification
- **Source**: https://huggingface.co/datasets/stanfordnlp/sst2

#### 2. SST (Stanford Sentiment Treebank - Full)
- **Type**: Fine-grained 5-class classification
- **Classes**: Very Negative, Negative, Neutral, Positive, Very Positive
- **Size**: Full parse trees with sub-sentence annotations
- **Source**: https://huggingface.co/datasets/stanfordnlp/sst

### Model Architecture

**Base Model**: DistilBERT (lightweight version of BERT)
- Reduced parameters for faster training
- Maintains good accuracy
- Suitable for CPU and GPU training

**Alternative Options:**
- BERT (larger, better accuracy)
- RoBERTa (improved pre-training)

### Notebook Sections (10 Parts)

1. **Import Required Libraries**
   - Load dependencies (transformers, datasets, torch, pandas, sklearn)
   - Set random seeds for reproducibility

2. **Load and Explore Dataset 1: SST2**
   - Load from Hugging Face
   - Analyze label distribution
   - Visualize text length statistics

3. **Load and Explore Dataset 2: SST (Full)**
   - Load full SST dataset
   - Explore parse tree structure
   - Analyze fine-grained sentiment labels

4. **Data Preprocessing**
   - Tokenization with pre-trained tokenizer
   - Padding and truncation (max_length=128)
   - PyTorch tensor conversion
   - Batched processing

5. **Model Training on SST2**
   - Initialize DistilBERT for binary classification
   - Configure training arguments
   - Fine-tune on SST2 training set
   - Validate on validation set

6. **Model Training on SST (Full)**
   - Convert continuous labels to 5 discrete classes
   - Initialize model for 5-class classification
   - Fine-tune on full SST dataset
   - Monitor training progress

7. **Model Evaluation on Test Sets**
   - Evaluate both models on test sets
   - Compute accuracy and F1-scores
   - Generate confusion matrices
   - Print classification reports

8. **Inference and Predictions**
   - Create text-classification pipelines
   - Make predictions on new text samples
   - Display confidence scores
   - Show sentiment for multiple examples

9. **Performance Comparison**
   - Compare SST2 vs SST models
   - Visualize performance metrics
   - Analyze differences between datasets

10. **Analysis and Insights**
    - Summary of results
    - Key observations
    - Possible improvements

### How to Run

#### Step 1: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

Or manually:
```bash
pip install datasets transformers torch pandas numpy matplotlib seaborn scikit-learn
```

#### Step 2: Launch Jupyter Notebook
```bash
# Using Jupyter
jupyter notebook src/Task_3_Sentiment_Classification.ipynb

# Or using Jupyter Lab
jupyter lab src/Task_3_Sentiment_Classification.ipynb

# Or using VS Code
code src/Task_3_Sentiment_Classification.ipynb
```

#### Step 3: Execute Cells
- Run cells sequentially from top to bottom
- The notebook automatically handles dataset downloads
- Training will progress through epochs with validation metrics
- Results and visualizations are displayed inline

### Expected Output

After running all cells, you will get:

✅ **Dataset Statistics**
- Label distribution charts
- Text length histograms
- Sample data exploration

✅ **Trained Models**
- SST2 binary sentiment classifier
- SST 5-class sentiment classifier
- Model checkpoints saved

✅ **Evaluation Results**
- Accuracy metrics for both datasets
- F1-scores
- Confusion matrices with heatmaps
- Detailed classification reports

✅ **Sentiment Predictions**
Examples of model predictions on new text:
```
Input: "This movie was absolutely amazing!"
SST2: Positive (99% confidence)
SST: Very Positive (98% confidence)

Input: "Terrible film, I hated it."
SST2: Negative (97% confidence)
SST: Very Negative (96% confidence)

Input: "It was okay, nothing special."
SST2: Positive (55% confidence)
SST: Neutral (78% confidence)
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | distilbert-base-uncased | Pre-trained model |
| Batch Size | 32 | Samples per batch |
| Epochs | 3 | Training iterations |
| Max Length | 128 | Max tokens per sequence |
| Learning Rate | 5e-5 | Default Adam rate |
| Warmup Steps | 500 | Linear warmup |
| Weight Decay | 0.01 | L2 regularization |

### Output Files and Directories

After running the notebook:
```
./results/sst2/
├── checkpoint-xxx/     # SST2 model checkpoints
└── pytorch_model.bin   # Final model

./results/sst/
├── checkpoint-xxx/     # SST model checkpoints
└── pytorch_model.bin   # Final model

./logs/
└── events.out.tfevents... # Training logs
```

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 10 GB (for datasets)

**Recommended:**
- GPU: NVIDIA (CUDA 11.x+)
- RAM: 16 GB
- Storage: 20 GB

### Troubleshooting

#### Out of Memory Error
Reduce batch size in training arguments:
```python
per_device_train_batch_size=16
per_device_eval_batch_size=16
```

#### Slow Dataset Download
Pre-download datasets:
```python
from datasets import load_dataset
load_dataset('stanfordnlp/sst2')
load_dataset('stanfordnlp/sst')
```

#### Missing Dependencies
Upgrade transformers:
```bash
pip install --upgrade datasets transformers torch
```

#### GPU Not Detected
Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Key Features

✅ **Comprehensive Data Exploration**
- Automatic loading from Hugging Face
- Statistical analysis
- Distribution visualizations

✅ **Robust Preprocessing**
- Efficient tokenization
- Padding/truncation
- Batched processing

✅ **Advanced Model Training**
- Fine-tuning pre-trained transformers
- Validation during training
- Checkpoint management

✅ **Detailed Evaluation**
- Multiple evaluation metrics
- Visual confusion matrices
- Per-class performance

✅ **Production-Ready Inference**
- Fast text-classification pipelines
- Confidence scores
- Real-time predictions

### Possible Improvements

1. **Model Enhancements**
   - Use larger models (BERT, RoBERTa)
   - Ensemble multiple models
   - Layer freezing and progressive unfreezing

2. **Hyperparameter Tuning**
   - Grid search or random search
   - Different learning rates and batch sizes
   - Longer training with early stopping

3. **Data Augmentation**
   - Back-translation
   - Paraphrase generation
   - Synonym replacement

4. **Advanced Techniques**
   - Attention visualization
   - Token importance analysis
   - Error analysis and correction

5. **Deployment**
   - Export models to ONNX
   - Create REST API with FastAPI
   - Deploy with Docker containers

### Next Steps

After completing this notebook:

1. Compare with other baseline models
2. Implement data augmentation
3. Try different hyperparameters
4. Create ensemble models
5. Add interpretability analysis
6. Deploy as a web service

### References

- SST2 Dataset: https://huggingface.co/datasets/stanfordnlp/sst2
- SST Dataset: https://huggingface.co/datasets/stanfordnlp/sst
- DistilBERT: https://huggingface.co/docs/transformers/model_doc/distilbert
- Transformers Library: https://huggingface.co/docs/transformers/
- PyTorch: https://pytorch.org/

### Notes

- The notebook includes all necessary explanations and comments
- Each section is self-contained and can be modified independently
- Models are automatically downloaded on first run
- GPU acceleration is automatically detected and used if available
- All visualizations are displayed inline in the notebook
