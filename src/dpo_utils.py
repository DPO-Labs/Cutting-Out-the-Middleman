"""
Direct Preference Optimization (DPO) Loss Implementation

This module implements the DPO loss function as described in Rafailov et al.,
enabling language models to learn directly from preference data without the need
for reward modeling.

The DPO loss aligns model behavior with human preferences by maximizing the
likelihood of preferred responses while minimizing the likelihood of dispreferred
responses, using only the frozen reference model for normalization.

Reference: Rafailov et al. "Direct Preference Optimization: Your Language Model
           is Secretly a Reward Model" (https://arxiv.org/abs/2305.18290)

Author: Backend Engineering Team
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def get_token_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    padding_token_id: int = -100
) -> torch.Tensor:
    """
    Extract the log probabilities of specific tokens from full vocabulary logits.
    
    This utility function converts vocabulary-level logits to log probabilities
    and extracts only the log probabilities for the specified label tokens.
    Padding tokens (typically marked with -100) are ignored during computation.
    
    Args:
        logits (torch.Tensor): Raw logits from the model of shape
            (batch_size, sequence_length, vocab_size).
            Each value is the model's raw output score for each token in the vocab.
        
        labels (torch.Tensor): Token IDs indicating which tokens we want log probs for.
            Shape: (batch_size, sequence_length).
            Values should be token IDs in range [0, vocab_size).
            Use padding_token_id (default -100) to mark tokens to ignore.
        
        padding_token_id (int): Token ID used to mark padding positions.
            Positions with this label will be set to log_prob = 0 (no loss).
            Default: -100 (standard PyTorch convention)
    
    Returns:
        torch.Tensor: Log probabilities of the specified tokens.
            Shape: (batch_size, sequence_length).
            For padding positions, returns 0.0 (no contribution to loss).
    
    Mathematical Details:
        - Convert logits to log probabilities using log_softmax
        - Gather log probs at positions specified by labels
        - Zero out contributions from padding tokens
    
    Example:
        >>> logits = torch.randn(2, 5, 50257)  # batch, seq_len, vocab
        >>> labels = torch.tensor([[10, 20, 30, -100, -100],
        ...                         [15, 25, 35, 45, -100]])
        >>> log_probs = get_token_log_probs(logits, labels)
        >>> print(log_probs.shape)  # (2, 5)
    
    Note:
        This function is highly optimized using torch.gather for efficiency.
    """
    # Convert logits to log probabilities using log-softmax
    # Numerical stability: log_softmax handles large values gracefully
    # Shape: (batch_size, sequence_length, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probabilities at positions specified by labels.
    # Padding labels such as -100 are not valid gather indices, so replace them
    # with a safe index first and then zero out those positions after gathering.
    batch_size, seq_length = labels.shape

    # Reshape for gathering: (batch_size * seq_length, vocab_size)
    log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))

    # Flatten labels: (batch_size * seq_length,)
    labels_flat = labels.reshape(-1)

    # Replace padding labels with a valid index before gather.
    # These positions will be masked out immediately after gathering.
    safe_labels_flat = labels_flat.clone()
    safe_labels_flat[safe_labels_flat == padding_token_id] = 0

    # Gather: for each position, get the log prob of the specified token
    # torch.gather is a highly optimized operation for this task
    # Result shape: (batch_size * seq_length,)
    gathered_log_probs = torch.gather(
        log_probs_flat,
        dim=1,
        index=safe_labels_flat.unsqueeze(1)
    ).squeeze(1)

    # Reshape back to original dimensions
    # Shape: (batch_size, sequence_length)
    gathered_log_probs = gathered_log_probs.reshape(batch_size, seq_length)

    # Zero out log probabilities for padding tokens
    # This ensures padding positions don't contribute to the loss
    mask = (labels != padding_token_id).float()
    gathered_log_probs = gathered_log_probs * mask
    
    return gathered_log_probs


def compute_dpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    winner_labels: torch.Tensor,
    loser_labels: torch.Tensor,
    beta: float = 0.1,
    padding_token_id: int = -100
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    DPO aligns language models with human preferences by directly learning from
    preference data without explicit reward modeling. The loss maximizes the
    log likelihood ratio between preferred (winner) and dispreferred (loser)
    responses, scaled by a temperature parameter beta.
    
    The DPO loss formula (Rafailov et al.):
    
    L_DPO = - log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))
    
    Where:
        - π_θ: Policy model (being trained)
        - π_ref: Reference model (frozen)
        - y_w: Winner (preferred) response
        - y_l: Loser (dispreferred) response
        - β: Temperature parameter controlling preference strength
        - σ: Sigmoid function
    
    Args:
        policy_logits (torch.Tensor): Raw logits from the policy model.
            Shape: (batch_size, sequence_length, vocab_size).
        
        ref_logits (torch.Tensor): Raw logits from the frozen reference model.
            Shape: (batch_size, sequence_length, vocab_size).
            IMPORTANT: Should come from model with requires_grad=False.
        
        winner_labels (torch.Tensor): Token IDs for the preferred response.
            Shape: (batch_size, sequence_length).
            Use padding_token_id (default -100) to mark padding positions.
        
        loser_labels (torch.Tensor): Token IDs for the dispreferred response.
            Shape: (batch_size, sequence_length).
            Use padding_token_id (default -100) to mark padding positions.
        
        beta (float): Temperature parameter controlling the strength of preference.
            - Larger β: Stronger preference for winner over loser
            - Smaller β: Softer preference learning
            Default: 0.1 (common value in DPO literature)
        
        padding_token_id (int): Token ID marking padding/ignored positions.
            Default: -100 (PyTorch standard)
    
    Returns:
        Tuple[torch.Tensor, dict]: A tuple containing:
            - loss (torch.Tensor): Scalar DPO loss (mean across batch and sequence)
            - metrics (dict): Dictionary containing useful training metrics:
                - 'policy_logit_diff': Mean log probability difference for policy
                - 'ref_logit_diff': Mean log probability difference for reference
                - 'margin': Mean raw margin before sigmoid
                - 'accuracy': Fraction of examples where margin > 0
                           (how often policy prefers winner over loser)
    
    Mathematical Steps:
        1. Extract log probs for winner tokens from both models
        2. Extract log probs for loser tokens from both models
        3. Compute log probability ratios: log(π_θ/π_ref)
        4. Compute preference margin: β * (policy_ratio - ref_ratio)
        5. Apply log sigmoid loss: - log σ(margin)
        6. Average over all non-padding positions
    
    Example:
        >>> policy_logits = torch.randn(2, 10, 50257)  # batch=2, seq_len=10
        >>> ref_logits = torch.randn(2, 10, 50257)
        >>> winner_labels = torch.randint(0, 50257, (2, 10))
        >>> loser_labels = torch.randint(0, 50257, (2, 10))
        >>> loss, metrics = compute_dpo_loss(policy_logits, ref_logits,
        ...                                   winner_labels, loser_labels,
        ...                                   beta=0.1)
        >>> print(f"Loss: {loss.item():.4f}")
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    Optimization Notes:
        - Uses torch.gather for efficient log probability extraction
        - All operations are vectorized; no Python loops
        - Numerically stable log_softmax for probability computation
        - Computes metrics in a single pass without additional forward passes
    
    Reference:
        Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Finn, C., & Durrett, G.
        (2023). Direct Preference Optimization: Your Language Model is Secretly a 
        Reward Model. arXiv preprint arXiv:2305.18290.
    """
    
    # ========================================================================
    # Step 1: Extract log probabilities for winner and loser tokens
    # ========================================================================
    
    # Get log probabilities of winner tokens from both models
    # Shape: (batch_size, sequence_length)
    policy_log_prob_winner = get_token_log_probs(
        policy_logits, winner_labels, padding_token_id
    )
    ref_log_prob_winner = get_token_log_probs(
        ref_logits, winner_labels, padding_token_id
    )
    
    # Get log probabilities of loser tokens from both models
    # Shape: (batch_size, sequence_length)
    policy_log_prob_loser = get_token_log_probs(
        policy_logits, loser_labels, padding_token_id
    )
    ref_log_prob_loser = get_token_log_probs(
        ref_logits, loser_labels, padding_token_id
    )
    
    # ========================================================================
    # Step 2: Compute log probability ratios: log(π_θ / π_ref)
    # ========================================================================
    
    # Policy model log probability ratios
    # log(π_θ(y_w|x)) - log(π_ref(y_w|x)) = log(π_θ(y_w|x) / π_ref(y_w|x))
    policy_log_ratio_winner = policy_log_prob_winner - ref_log_prob_winner
    policy_log_ratio_loser = policy_log_prob_loser - ref_log_prob_loser
    
    # Reference model log probability ratios
    # These should be close to zero, since we're dividing the model by itself
    ref_log_ratio_winner = ref_log_prob_winner - ref_log_prob_winner  # = 0
    ref_log_ratio_loser = ref_log_prob_loser - ref_log_prob_loser      # = 0
    
    # Note: The reference ratios are exactly zero; we could optimize by not computing them,
    # but we keep them explicit for clarity of the algorithm
    
    # ========================================================================
    # Step 3: Compute preference margin with temperature parameter β
    # ========================================================================
    
    # The margin represents how much the policy model prefers the winner over loser
    # relative to the reference model
    # margin = β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x))
    # Since reference ratios = 0:
    # margin = β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x))
    margin = beta * (policy_log_ratio_winner - policy_log_ratio_loser)
    
    # ========================================================================
    # Step 4: Compute DPO loss using log sigmoid
    # ========================================================================
    
    # DPO loss: - log σ(margin)
    # Where σ is the sigmoid function
    # PyTorch's F.logsigmoid is numerically stable and efficient
    # This computes: log(1 / (1 + exp(-margin)))
    log_sigmoid_margin = F.logsigmoid(margin)
    
    # DPO loss (negative log sigmoid)
    dpo_loss_per_token = -log_sigmoid_margin
    
    # ========================================================================
    # Step 5: Mask out padding tokens and compute mean loss
    # ========================================================================
    
    # Create mask for non-padding positions
    # A position is non-padding if both winner and loser labels are non-padding
    mask = (
        (winner_labels != padding_token_id).float() *
        (loser_labels != padding_token_id).float()
    )
    
    # Apply mask: zero out loss for padding positions
    masked_loss = dpo_loss_per_token * mask
    
    # Compute mean loss only over non-padding positions
    num_non_padding = mask.sum()
    if num_non_padding > 0:
        loss = masked_loss.sum() / num_non_padding
    else:
        # Handle case where all positions are padding (shouldn't happen in practice)
        loss = masked_loss.sum()
    
    # ========================================================================
    # Step 6: Compute training metrics for monitoring
    # ========================================================================
    
    # Policy log probability difference (should be positive for good training)
    policy_logit_diff = policy_log_ratio_winner - policy_log_ratio_loser
    policy_logit_diff_mean = (policy_logit_diff * mask).sum() / (num_non_padding + 1e-8)
    
    # Reference log probability difference (should be ~0)
    ref_logit_diff = ref_log_ratio_winner - ref_log_ratio_loser
    ref_logit_diff_mean = (ref_logit_diff * mask).sum() / (num_non_padding + 1e-8)
    
    # Raw margin before sigmoid
    margin_mean = (margin * mask).sum() / (num_non_padding + 1e-8)
    
    # Accuracy: fraction of examples where policy prefers winner over loser
    # (margin > 0 means policy prefers winner)
    accuracy = ((margin > 0).float() * mask).sum() / (num_non_padding + 1e-8)
    
    # Compile metrics dictionary
    metrics = {
        'policy_logit_diff': policy_logit_diff_mean.item(),
        'ref_logit_diff': ref_logit_diff_mean.item(),
        'margin': margin_mean.item(),
        'accuracy': accuracy.item(),
    }
    
    return loss, metrics


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating DPO loss computation with synthetic data.
    """
    
    print("=" * 70)
    print("Direct Preference Optimization (DPO) Loss Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # ========================================================================
    # Synthetic Data Setup
    # ========================================================================
    
    batch_size = 4
    seq_length = 8
    vocab_size = 50257
    
    print(f"\nSynthetic Data Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Vocabulary size: {vocab_size}")
    
    # Create synthetic logits (usually from model forward pass)
    policy_logits = torch.randn(batch_size, seq_length, vocab_size)
    ref_logits = torch.randn(batch_size, seq_length, vocab_size)
    
    # Create synthetic winner and loser labels
    winner_labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    loser_labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Add some padding (marked with -100)
    mask = torch.rand(batch_size, seq_length) > 0.8  # 20% padding
    winner_labels[mask] = -100
    loser_labels[mask] = -100
    
    # ========================================================================
    # Compute DPO Loss
    # ========================================================================
    
    print("\nComputing DPO loss...")
    loss, metrics = compute_dpo_loss(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        winner_labels=winner_labels,
        loser_labels=loser_labels,
        beta=0.1
    )
    
    # ========================================================================
    # Display Results
    # ========================================================================
    
    print(f"\nDPO Loss Results:")
    print(f"  Total Loss: {loss.item():.6f}")
    print(f"\nTraining Metrics:")
    print(f"  Policy Log Prob Difference: {metrics['policy_logit_diff']:.6f}")
    print(f"  Reference Log Prob Difference: {metrics['ref_logit_diff']:.6f}")
    print(f"  Margin (before sigmoid): {metrics['margin']:.6f}")
    print(f"  Accuracy (policy prefers winner): {metrics['accuracy']:.4f}")
    
    # ========================================================================
    # Demonstrate get_token_log_probs utility function
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Token Log Probability Extraction Example")
    print("=" * 70)
    
    logits = torch.randn(2, 5, 50257)
    labels = torch.tensor([
        [10, 20, 30, -100, -100],
        [15, 25, 35, 45, -100]
    ])
    
    log_probs = get_token_log_probs(logits, labels)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Log probabilities:\n{log_probs}")
    print(f"\nNote: Last column should be all zeros (padding tokens)")
    
    # ========================================================================
    # Test Different Beta Values
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DPO Loss with Different Beta Values")
    print("=" * 70)
    
    beta_values = [0.05, 0.1, 0.2, 0.5]
    
    for beta in beta_values:
        loss_beta, metrics_beta = compute_dpo_loss(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            winner_labels=winner_labels,
            loser_labels=loser_labels,
            beta=beta
        )
        print(f"\nβ = {beta}:")
        print(f"  Loss: {loss_beta.item():.6f}")
        print(f"  Accuracy: {metrics_beta['accuracy']:.4f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
