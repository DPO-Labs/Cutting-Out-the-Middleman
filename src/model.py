"""
GPT-2 Model Wrapper for Direct Preference Optimization (DPO) Training

This module provides a clean wrapper around the Hugging Face GPT-2 model
specifically designed for DPO training alongside SFT. The wrapper ensures
that raw vocabulary-level logits are returned without any additional
classification heads, as required by the DPO algorithm.

Author: Backend Engineering Team
"""

import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Tuple


class GPT2Wrapper:
    """
    A wrapper class for the GPT-2 model optimized for Direct Preference Optimization.
    
    This class encapsulates the GPT-2 model from Hugging Face transformers library
    and provides utilities for DPO training. Crucially, it returns raw vocabulary-level
    logits without any additional classification heads, which is required for DPO to
    work correctly alongside Supervised Fine-Tuning (SFT).
    
    Attributes:
        model: The underlying AutoModelForCausalLM instance.
        model_name: Name of the GPT-2 variant ('gpt2' or 'gpt2-large').
        device: Device on which the model runs (cpu or cuda).
    """
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        device: Optional[str] = None
    ):
        """
        Initialize the GPT-2 wrapper.
        
        Args:
            model_name (str): The model variant to load. Supported options:
                - 'gpt2': Base GPT-2 model (117M parameters)
                - 'gpt2-large': Large GPT-2 model (774M parameters)
                Default: 'gpt2'
            
            device (str, optional): Device to place the model on ('cuda' or 'cpu').
                If None, automatically selects CUDA if available, otherwise CPU.
                Default: None (auto-select)
        
        Raises:
            ValueError: If an unsupported model_name is provided.
            RuntimeError: If model loading fails.
        """
        # Validate model name
        supported_models = ['gpt2', 'gpt2-large']
        if model_name not in supported_models:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {supported_models}"
            )
        
        self.model_name = model_name
        
        # Auto-select device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load the model from Hugging Face
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}': {str(e)}"
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GPT-2 model.
        
        CRITICAL DESIGN CHOICE: This method returns raw vocabulary-level logits
        directly from the base GPT-2 model. NO additional classification head or
        output layer is applied. This is essential for DPO training to work correctly
        alongside SFT, as DPO requires direct access to the model's logits over
        the entire vocabulary.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, sequence_length).
                Typically produced by a tokenizer.
            
            attention_mask (torch.Tensor, optional): Binary mask indicating which tokens
                to attend to. Shape: (batch_size, sequence_length).
                - 1: tokens to attend to
                - 0: tokens to ignore (padding)
                Default: None (attend to all tokens)
        
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, sequence_length, vocab_size).
                Each value represents the model's raw output score for each token
                in the vocabulary at each sequence position.
        
        Example:
            >>> wrapper = GPT2Wrapper('gpt2')
            >>> input_ids = torch.tensor([[101, 2054, 2003, 102]])
            >>> logits = wrapper.forward(input_ids)
            >>> print(logits.shape)  # (1, 4, 50257)
        """
        # Move inputs to the same device as the model
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract and return raw logits (no additional processing)
        # Shape: (batch_size, sequence_length, vocab_size)
        logits = outputs.logits
        
        return logits
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make the wrapper callable. Delegates to forward method.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Attention mask of shape 
                (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, sequence_length, vocab_size).
        
        Example:
            >>> wrapper = GPT2Wrapper('gpt2')
            >>> logits = wrapper(input_ids)  # Calls __call__ -> forward
        """
        return self.forward(input_ids, attention_mask)
    
    def get_reference_model(self) -> 'GPT2Wrapper':
        """
        Create a frozen copy of the model to serve as the reference policy.
        
        In DPO training, we need a frozen reference model (π_ref) that does not
        update during training. This method creates a deep copy of the current
        model and disables gradient computation for all parameters.
        
        The reference model is essential for:
        - Computing the reference logits during DPO loss calculation
        - Maintaining a fixed baseline for preference learning
        - Preventing the reference policy from changing during training
        
        Returns:
            GPT2Wrapper: A new GPT2Wrapper instance with a frozen model copy.
                All parameters have requires_grad=False.
        
        Example:
            >>> policy_model = GPT2Wrapper('gpt2')
            >>> reference_model = policy_model.get_reference_model()
            >>> # reference_model.model is frozen and won't update during training
        
        Note:
            The returned model is completely independent from the original.
            Changes to the policy model will not affect the reference model.
        """
        # Create a deep copy of the wrapper
        ref_wrapper = copy.deepcopy(self)
        
        # Freeze all parameters in the reference model
        # This prevents any gradient computation during DPO training
        for param in ref_wrapper.model.parameters():
            param.requires_grad = False
        
        return ref_wrapper
    
    def get_model(self) -> AutoModelForCausalLM:
        """
        Get direct access to the underlying Hugging Face model.
        
        Returns:
            AutoModelForCausalLM: The underlying GPT-2 model instance.
        
        Note:
            Use this method only when you need direct access to the model
            for advanced operations. Most common use cases should use the
            wrapper methods (forward, get_reference_model, etc.).
        """
        return self.model
    
    def get_model_name(self) -> str:
        """
        Get the name of the loaded model variant.
        
        Returns:
            str: The model name ('gpt2' or 'gpt2-large').
        """
        return self.model_name
    
    def get_device(self) -> str:
        """
        Get the device the model is running on.
        
        Returns:
            str: Device name ('cuda' or 'cpu').
        """
        return self.device
    
    def to(self, device: str) -> 'GPT2Wrapper':
        """
        Move the model to a different device.
        
        Args:
            device (str): Target device ('cuda' or 'cpu').
        
        Returns:
            GPT2Wrapper: Self, for method chaining.
        
        Example:
            >>> wrapper = GPT2Wrapper('gpt2', device='cpu')
            >>> wrapper.to('cuda')  # Move to GPU
        """
        self.model.to(device)
        self.device = device
        return self
    
    def eval(self) -> 'GPT2Wrapper':
        """
        Set the model to evaluation mode.
        
        In evaluation mode:
        - Dropout layers are disabled
        - Batch normalization uses running statistics
        - No gradients are computed
        
        Returns:
            GPT2Wrapper: Self, for method chaining.
        
        Example:
            >>> wrapper = GPT2Wrapper('gpt2')
            >>> wrapper.eval()  # Switch to evaluation mode
        """
        self.model.eval()
        return self
    
    def train(self) -> 'GPT2Wrapper':
        """
        Set the model to training mode.
        
        In training mode:
        - Dropout layers are active
        - Batch normalization computes statistics
        - Gradients are computed (if parameters have requires_grad=True)
        
        Returns:
            GPT2Wrapper: Self, for method chaining.
        
        Example:
            >>> wrapper = GPT2Wrapper('gpt2')
            >>> wrapper.train()  # Switch to training mode
        """
        self.model.train()
        return self


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the GPT2Wrapper for DPO training.
    """
    
    # Initialize the wrapper with the base GPT-2 model
    print("Initializing GPT-2 wrapper...")
    wrapper = GPT2Wrapper(model_name='gpt2')
    print(f"Model loaded on device: {wrapper.get_device()}")
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Get logits from the policy model
    print("\nGetting logits from policy model...")
    policy_logits = wrapper(input_ids, attention_mask)
    print(f"Policy logits shape: {policy_logits.shape}")  # (2, 10, 50257)
    
    # Create a frozen reference model for DPO training
    print("\nCreating frozen reference model...")
    ref_wrapper = wrapper.get_reference_model()
    ref_logits = ref_wrapper(input_ids, attention_mask)
    print(f"Reference logits shape: {ref_logits.shape}")  # (2, 10, 50257)
    
    # Verify that reference model is frozen
    print("\nVerifying reference model is frozen...")
    frozen_count = sum(1 for p in ref_wrapper.model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in ref_wrapper.model.parameters())
    print(f"Frozen parameters: {frozen_count}/{total_count}")
    
    print("\nExample completed successfully!")
