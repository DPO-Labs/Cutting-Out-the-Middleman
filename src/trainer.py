"""
Training Loop for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)

This module provides a unified Trainer class that handles both SFT and DPO training modes,
enabling seamless switching between standard language model fine-tuning and preference-based
learning. The trainer is designed to be modular and easily importable for use in Jupyter
notebooks by different team members working on different datasets.

Author: Backend Engineering Team
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional, Dict, Tuple, Union
from tqdm import tqdm

# Import the DPO loss function from our utils module
from dpo_utils import compute_dpo_loss


class Trainer:
    """
    A unified trainer class for both Supervised Fine-Tuning (SFT) and 
    Direct Preference Optimization (DPO) training modes.
    
    This trainer manages the training loop, handling forward passes, backward passes,
    and optimizer steps for both training methodologies. It provides a clean interface
    for training language models with different preference and supervision signals.
    
    Attributes:
        policy_model: The model being trained (requires_grad=True)
        ref_model: Frozen reference model for DPO (requires_grad=False)
        optimizer: AdamW optimizer for updating policy model parameters
        scaler: GradScaler for fp16 mixed precision on CUDA
        device: Device on which training occurs ('cuda' or 'cpu')
        mode: Training mode ('sft' or 'dpo')
        train_losses: List to track losses during training
        val_losses: List to track validation losses
    """
    
    def __init__(
        self,
        policy_model,
        ref_model: Optional[object] = None,
        learning_rate: float = 5e-5,
        device: str = 'cuda',
        beta: float = 0.1,
        padding_token_id: int = -100
    ):
        """
        Initialize the Trainer.
        
        Args:
            policy_model: The model to be trained (GPT2Wrapper or similar).
                Will have gradients enabled for all parameters.
            
            ref_model (optional): Frozen reference model for DPO.
                Only required if mode='dpo'.
                Should have all parameters with requires_grad=False.
                Default: None
            
            learning_rate (float): Learning rate for AdamW optimizer.
                Default: 5e-5 (common for fine-tuning)
            
            device (str): Device for training ('cuda' or 'cpu').
                Default: 'cuda'
            
            beta (float): Temperature parameter for DPO loss.
                Controls preference strength. Default: 0.1
            
            padding_token_id (int): Token ID for padding (usually -100).
                Default: -100
        
        Raises:
            ValueError: If device is not 'cuda' or 'cpu'.
        """
        # Validate device
        if device not in ['cuda', 'cpu']:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got '{device}'")
        
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.device = device
        self.beta = beta
        self.padding_token_id = padding_token_id
        
        # Move models to device
        self.policy_model.to(device)
        if self.ref_model is not None:
            self.ref_model.to(device)
        
        # Initialize AdamW optimizer
        # Only optimize policy model parameters
        self.optimizer = AdamW(
            self.policy_model.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0  # no weight decay for paraphrase SFT (per report)
        )
        
        # GradScaler for mixed precision on CUDA
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
        
        # Training metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
    
    def sft_train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Perform a single Supervised Fine-Tuning (SFT) training step.
        
        SFT trains the model using standard cross-entropy loss to predict the next token.
        This is standard language model fine-tuning where the model learns to imitate
        the behavior in the training data.
        
        Batch Format:
            The batch dictionary should contain:
            - 'input_ids': Token IDs of shape (batch_size, seq_length)
            - 'attention_mask': Binary attention mask of shape (batch_size, seq_length)
            - 'labels': Target token IDs for next-token prediction
                       (typically input_ids shifted right by one position)
        
        Loss Calculation:
            For each position in the sequence, the model predicts the next token.
            Cross-entropy loss is computed: CE = -log P(y_t | y_1,...,y_{t-1})
            
        Args:
            batch (dict): Batch of training data with keys:
                - 'input_ids': Token IDs (batch_size, seq_length)
                - 'attention_mask': Attention mask (batch_size, seq_length)
                - 'labels': Target labels (batch_size, seq_length)
        
        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing:
                - loss: Scalar cross-entropy loss
                - metrics (dict): Dictionary with training metrics:
                    - 'sft_loss': The cross-entropy loss value
                    - 'perplexity': exp(loss) - a measure of model uncertainty
        
        Example:
            >>> trainer = Trainer(policy_model, device='cuda')
            >>> batch = {
            ...     'input_ids': torch.randint(0, 50257, (4, 10)),
            ...     'attention_mask': torch.ones(4, 10),
            ...     'labels': torch.randint(0, 50257, (4, 10))
            ... }
            >>> loss, metrics = trainer.sft_train_step(batch)
            >>> print(f"Loss: {loss.item():.4f}")
        """
        # ====================================================================
        # Extract batch components
        # ====================================================================
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # ====================================================================
        # Forward pass through policy model
        # ====================================================================
        
        # Get logits from policy model
        # Shape: (batch_size, sequence_length, vocab_size)
        with torch.amp.autocast(device_type=self.device, enabled=(self.device == 'cuda')):
            logits = self.policy_model(input_ids, attention_mask)
        
        # ====================================================================
        # Compute cross-entropy loss
        # ====================================================================
        
        # Reshape logits and labels for cross-entropy computation
        # cross_entropy expects:
        # - logits: (batch_size * sequence_length, vocab_size)
        # - labels: (batch_size * sequence_length,)
        batch_size, seq_length = input_ids.shape
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)
        
        # Compute cross-entropy loss
        # By default, includes averaging over all positions
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.padding_token_id,  # Ignore padding tokens
            reduction='mean'
        )
        
        # ====================================================================
        # Compute metrics
        # ====================================================================
        
        # Perplexity: exp(loss)
        # Lower perplexity indicates better predictions
        perplexity = torch.exp(loss)
        
        metrics = {
            'sft_loss': loss.item(),
            'perplexity': perplexity.item(),
        }
        
        return loss, metrics
    
    def dpo_train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Perform a single Direct Preference Optimization (DPO) training step.
        
        DPO trains the model to prefer winner (preferred) sequences over loser
        (dispreferred) sequences without explicit reward modeling. This is done by
        directly optimizing the DPO objective using both the policy and reference models.
        
        Batch Format:
            The batch dictionary should contain pairs of completions:
            - 'winner_input_ids': Token IDs for preferred completions
            - 'winner_attention_mask': Attention mask for winners
            - 'loser_input_ids': Token IDs for dispreferred completions
            - 'loser_attention_mask': Attention mask for losers
            
            OR (if sequences contain both prompt and completion):
            - 'winner_input_ids': Full sequence (prompt + completion)
            - 'winner_labels': Target tokens (completion only, prompt is padding)
            - 'loser_input_ids': Full sequence (prompt + completion)
            - 'loser_labels': Target tokens (completion only, prompt is padding)
        
        Algorithm:
            1. Forward pass: Get logits from both policy and reference models
            2. Extract log probabilities for winner and loser sequences
            3. Compute preference margin using DPO formula
            4. Compute DPO loss using log-sigmoid
            5. Return loss and metrics
        
        Args:
            batch (dict): Batch of preference data with keys:
                - 'winner_input_ids': Token IDs for preferred sequences
                - 'winner_attention_mask': Attention mask for winners
                - 'loser_input_ids': Token IDs for dispreferred sequences
                - 'loser_attention_mask': Attention mask for losers
                
                AND one of:
                - 'winner_labels'/'loser_labels': Target token labels
                  OR auto-generate from input_ids (next token prediction)
        
        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing:
                - loss: Scalar DPO loss
                - metrics (dict): Dictionary containing:
                    - 'dpo_loss': The DPO loss value
                    - 'policy_logit_diff': Policy model's log prob difference
                    - 'margin': Preference margin before sigmoid
                    - 'accuracy': Fraction preferring winner over loser
        
        Raises:
            ValueError: If reference model is not provided.
            KeyError: If required batch keys are missing.
        
        Example:
            >>> trainer = Trainer(policy_model, ref_model, device='cuda')
            >>> batch = {
            ...     'winner_input_ids': torch.randint(0, 50257, (4, 10)),
            ...     'winner_attention_mask': torch.ones(4, 10),
            ...     'winner_labels': torch.randint(0, 50257, (4, 10)),
            ...     'loser_input_ids': torch.randint(0, 50257, (4, 10)),
            ...     'loser_attention_mask': torch.ones(4, 10),
            ...     'loser_labels': torch.randint(0, 50257, (4, 10)),
            ... }
            >>> loss, metrics = trainer.dpo_train_step(batch)
            >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        """
        # ====================================================================
        # Validation
        # ====================================================================
        
        if self.ref_model is None:
            raise ValueError(
                "Reference model required for DPO training. "
                "Provide ref_model when initializing Trainer."
            )
        
        # ====================================================================
        # Extract batch components
        # ====================================================================
        
        winner_input_ids = batch['winner_input_ids'].to(self.device)
        winner_attention_mask = batch.get(
            'winner_attention_mask',
            torch.ones_like(winner_input_ids)
        ).to(self.device)
        
        loser_input_ids = batch['loser_input_ids'].to(self.device)
        loser_attention_mask = batch.get(
            'loser_attention_mask',
            torch.ones_like(loser_input_ids)
        ).to(self.device)
        
        # Extract labels if provided, otherwise use input_ids (next token prediction)
        if 'winner_labels' in batch:
            winner_labels = batch['winner_labels'].to(self.device)
        else:
            # Shift input_ids right to create labels (next token prediction)
            winner_labels = winner_input_ids.clone()
        
        if 'loser_labels' in batch:
            loser_labels = batch['loser_labels'].to(self.device)
        else:
            loser_labels = loser_input_ids.clone()
        
        # ====================================================================
        # Forward pass: Policy model
        # ====================================================================
        
        # Get logits from policy model for both winner and loser sequences
        # Shape: (batch_size, sequence_length, vocab_size)
        policy_logits_winner = self.policy_model(
            winner_input_ids,
            winner_attention_mask
        )
        policy_logits_loser = self.policy_model(
            loser_input_ids,
            loser_attention_mask
        )
        
        # ====================================================================
        # Forward pass: Reference model (frozen, no gradients)
        # ====================================================================
        
        # Disable gradient computation for reference model
        with torch.no_grad():
            ref_logits_winner = self.ref_model(
                winner_input_ids,
                winner_attention_mask
            )
            ref_logits_loser = self.ref_model(
                loser_input_ids,
                loser_attention_mask
            )
        
        # ====================================================================
        # Compute DPO loss
        # ====================================================================
        
        # DPO loss for winner sequence
        loss_winner, metrics_winner = compute_dpo_loss(
            policy_logits=policy_logits_winner,
            ref_logits=ref_logits_winner,
            winner_labels=winner_labels,
            loser_labels=winner_labels,  # For single sequence, winner=winner, loser=winner
            beta=self.beta,
            padding_token_id=self.padding_token_id
        )
        
        # DPO loss for loser sequence
        loss_loser, metrics_loser = compute_dpo_loss(
            policy_logits=policy_logits_loser,
            ref_logits=ref_logits_loser,
            winner_labels=loser_labels,
            loser_labels=loser_labels,
            beta=self.beta,
            padding_token_id=self.padding_token_id
        )
        
        # Combined DPO loss (average of winner and loser)
        loss = (loss_winner + loss_loser) / 2
        
        # ====================================================================
        # Aggregate metrics
        # ====================================================================
        
        metrics = {
            'dpo_loss': loss.item(),
            'policy_logit_diff': (
                metrics_winner['policy_logit_diff'] +
                metrics_loser['policy_logit_diff']
            ) / 2,
            'margin': (
                metrics_winner['margin'] +
                metrics_loser['margin']
            ) / 2,
            'accuracy': (
                metrics_winner['accuracy'] +
                metrics_loser['accuracy']
            ) / 2,
        }
        
        return loss, metrics
    
    def train(
        self,
        epochs: int,
        dataloader,
        mode: str = 'sft',
        val_dataloader = None,
        gradient_accumulation_steps: int = 1
    ) -> Dict:
        """
        Full training loop for either SFT or DPO mode.
        
        This method handles the complete training process:
        - Iterates through epochs and batches
        - Performs forward and backward passes
        - Updates optimizer weights
        - Tracks and prints metrics
        - Optionally validates on a separate dataset
        
        Training Workflow:
            For each epoch:
                For each batch:
                    1. Forward pass (compute logits and loss)
                    2. Backward pass (compute gradients)
                    3. Accumulate gradients (if gradient_accumulation_steps > 1)
                    4. Update parameters (optimizer step)
                    5. Track metrics
                Print epoch summary
                Optionally validate
        
        Args:
            epochs (int): Number of training epochs.
            
            dataloader: PyTorch DataLoader with batches.
                Batches should be dictionaries with appropriate keys
                depending on mode ('sft' or 'dpo').
            
            mode (str): Training mode ('sft' or 'dpo').
                - 'sft': Supervised Fine-Tuning with cross-entropy loss
                - 'dpo': Direct Preference Optimization with preference learning
                Default: 'sft'
            
            val_dataloader (optional): Validation DataLoader.
                If provided, validation loss is computed after each epoch.
                Default: None
            
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
                before updating. Useful for simulating larger batch sizes.
                Default: 1 (update every batch)
        
        Returns:
            Dict: Training results containing:
                - 'train_losses': List of average losses per epoch
                - 'val_losses': List of validation losses per epoch (if val_dataloader provided)
                - 'train_metrics': List of metric dictionaries per epoch
        
        Raises:
            ValueError: If mode is not 'sft' or 'dpo'.
            RuntimeError: If DPO mode selected but ref_model not initialized.
        
        Example:
            >>> trainer = Trainer(policy_model, ref_model, device='cuda')
            >>> results = trainer.train(
            ...     epochs=3,
            ...     dataloader=train_dataloader,
            ...     mode='dpo',
            ...     val_dataloader=val_dataloader
            ... )
            >>> print(f"Final train loss: {results['train_losses'][-1]:.4f}")
        """
        # ====================================================================
        # Validation and Setup
        # ====================================================================
        
        if mode not in ['sft', 'dpo']:
            raise ValueError(f"Mode must be 'sft' or 'dpo', got '{mode}'")
        
        if mode == 'dpo' and self.ref_model is None:
            raise RuntimeError(
                "DPO mode requires reference model. "
                "Provide ref_model when initializing Trainer."
            )
        
        # Select training function based on mode
        train_step_fn = self.sft_train_step if mode == 'sft' else self.dpo_train_step
        
        print(f"\n{'='*70}")
        print(f"Starting {mode.upper()} Training")
        print(f"{'='*70}")
        print(f"Mode: {mode}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"{'='*70}\n")
        
        # ====================================================================
        # Training Loop
        # ====================================================================
        
        for epoch in range(epochs):
            # ================================================================
            # Training Phase
            # ================================================================
            
            self.policy_model.train()
            if self.ref_model is not None:
                self.ref_model.eval()
            
            epoch_losses = []
            epoch_metrics_list = []
            
            # Progress bar for training batches
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            for batch_idx, batch in enumerate(pbar):
                # Perform training step
                loss, metrics = train_step_fn(batch)
                
                # Backward pass with gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_losses.append(loss.item() * gradient_accumulation_steps)
                epoch_metrics_list.append(metrics)
                
                # Update parameters every gradient_accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update progress bar with current loss
                pbar.set_postfix({
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"
                })
            
            # ================================================================
            # Epoch Summary
            # ================================================================
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            self.train_metrics.append(epoch_metrics_list)
            
            # Aggregate metrics for display
            avg_metrics = {}
            for key in epoch_metrics_list[0].keys():
                avg_metrics[key] = sum(m[key] for m in epoch_metrics_list) / len(epoch_metrics_list)
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.6f}")
            for key, value in avg_metrics.items():
                print(f"  Average {key}: {value:.6f}")
            
            # ================================================================
            # Validation Phase (if val_dataloader provided)
            # ================================================================
            
            if val_dataloader is not None:
                self.policy_model.eval()
                
                val_losses = []
                with torch.no_grad():
                    val_pbar = tqdm(val_dataloader, desc="Validation", unit="batch")
                    for val_batch in val_pbar:
                        val_loss, _ = train_step_fn(val_batch)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                self.val_losses.append(avg_val_loss)
                print(f"  Validation Loss: {avg_val_loss:.6f}\n")
            else:
                print()
        
        # ====================================================================
        # Training Complete
        # ====================================================================
        
        print(f"{'='*70}")
        print(f"Training Complete!")
        print(f"Final Train Loss: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
        print(f"{'='*70}\n")
        
        # Return results
        results = {
            'train_losses': self.train_losses,
            'train_metrics': self.train_metrics,
        }
        
        if self.val_losses:
            results['val_losses'] = self.val_losses
        
        return results


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating the Trainer with synthetic data.
    This example shows how to use the Trainer for both SFT and DPO modes.
    """
    
    print("Trainer Example - Requires GPT2Wrapper from model.py")
    print("\nExample usage in Jupyter notebook:")
    print("""
    from src.model import GPT2Wrapper
    from src.trainer import Trainer
    
    # Initialize models
    policy_model = GPT2Wrapper('gpt2', device='cuda')
    ref_model = policy_model.get_reference_model()
    
    # Initialize trainer
    trainer = Trainer(
        policy_model=policy_model,
        ref_model=ref_model,
        learning_rate=5e-5,
        device='cuda',
        beta=0.1
    )
    
    # Train with SFT
    results = trainer.train(
        epochs=3,
        dataloader=train_dataloader,
        mode='sft'
    )
    
    # Or train with DPO
    results = trainer.train(
        epochs=3,
        dataloader=train_dataloader,
        mode='dpo',
        val_dataloader=val_dataloader
    )
    """)
