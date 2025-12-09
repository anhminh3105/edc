"""
Local LLM Manager for HuggingFace models with optional 4-bit/8-bit quantization.

This module provides a singleton-based model manager for loading and running
local LLMs as an alternative to OpenAI API calls. Optimized for GPUs with
limited VRAM (e.g., Tesla P100 with 16GB).

Supported models:
- mistralai/Mistral-7B-Instruct-v0.3
- Qwen/Qwen2.5-7B-Instruct
- microsoft/Phi-3.5-mini-instruct

Environment variables:
- USE_LOCAL_LLM: Set to "true" to enable local LLM mode
- LOCAL_LLM_MODEL: HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
- LOCAL_EMBEDDER_MODEL: Embedder model ID (default: BAAI/bge-small-en-v1.5)
- LOCAL_LLM_QUANTIZE: Quantization level - "4bit", "8bit", or "none" (default: 4bit)
"""

import os
import logging
import gc
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model configurations
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUANTIZATION = "4bit"


class LocalLLMManager:
    """
    Singleton manager for local LLM and embedder models.
    
    Handles model loading with quantization support and provides a chat
    completion interface compatible with the OpenAI API signature.
    """
    
    _instance = None
    _llm_model = None
    _llm_tokenizer = None
    _embedder_model = None
    _current_llm_name = None
    _current_embedder_name = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "LocalLLMManager":
        """Get the singleton instance of LocalLLMManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @staticmethod
    def is_local_mode_enabled() -> bool:
        """Check if local LLM mode is enabled via environment variable."""
        return os.environ.get("USE_LOCAL_LLM", "").lower() == "true"
    
    @staticmethod
    def get_quantization_config() -> Optional[BitsAndBytesConfig]:
        """
        Create a BitsAndBytesConfig based on the LOCAL_LLM_QUANTIZE environment variable.
        
        Returns:
            BitsAndBytesConfig for 4-bit or 8-bit quantization, or None for no quantization.
        """
        quantize = os.environ.get("LOCAL_LLM_QUANTIZE", DEFAULT_QUANTIZATION).lower()
        
        if quantize == "4bit":
            logger.info("Using 4-bit quantization for LLM")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantize == "8bit":
            logger.info("Using 8-bit quantization for LLM")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            logger.info("Loading LLM without quantization")
            return None
    
    def get_llm_model_name(self) -> str:
        """Get the LLM model name from environment or default."""
        return os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    def get_embedder_model_name(self) -> str:
        """Get the embedder model name from environment or default."""
        return os.environ.get("LOCAL_EMBEDDER_MODEL", DEFAULT_EMBEDDER_MODEL)
    
    def load_llm(self, model_name: Optional[str] = None, force_reload: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the LLM model and tokenizer with quantization support.
        
        Args:
            model_name: HuggingFace model ID. If None, uses LOCAL_LLM_MODEL env var.
            force_reload: If True, reload the model even if already loaded.
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name is None:
            model_name = self.get_llm_model_name()
        
        # Return cached model if available and not forcing reload
        if (
            self._llm_model is not None 
            and self._llm_tokenizer is not None 
            and self._current_llm_name == model_name 
            and not force_reload
        ):
            logger.info(f"Reusing cached LLM: {model_name}")
            return self._llm_model, self._llm_tokenizer
        
        # Free existing model if switching
        if self._llm_model is not None:
            self.free_llm()
        
        logger.info(f"Loading LLM: {model_name}")
        
        quantization_config = self.get_quantization_config()
        
        # Load tokenizer
        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not set
        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
        
        # Load model with quantization
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self._current_llm_name = model_name
        logger.info(f"LLM loaded successfully: {model_name}")
        
        return self._llm_model, self._llm_tokenizer
    
    def load_embedder(self, model_name: Optional[str] = None, force_reload: bool = False) -> SentenceTransformer:
        """
        Load the embedder model (SentenceTransformer).
        
        Args:
            model_name: HuggingFace model ID. If None, uses LOCAL_EMBEDDER_MODEL env var.
            force_reload: If True, reload the model even if already loaded.
            
        Returns:
            SentenceTransformer model
        """
        if model_name is None:
            model_name = self.get_embedder_model_name()
        
        # Return cached model if available and not forcing reload
        if (
            self._embedder_model is not None 
            and self._current_embedder_name == model_name 
            and not force_reload
        ):
            logger.info(f"Reusing cached embedder: {model_name}")
            return self._embedder_model
        
        # Free existing model if switching
        if self._embedder_model is not None:
            self.free_embedder()
        
        logger.info(f"Loading embedder: {model_name}")
        
        self._embedder_model = SentenceTransformer(model_name, trust_remote_code=True)
        self._current_embedder_name = model_name
        
        logger.info(f"Embedder loaded successfully: {model_name}")
        
        return self._embedder_model
    
    def free_llm(self):
        """Free the LLM model from memory."""
        if self._llm_model is not None:
            logger.info(f"Freeing LLM: {self._current_llm_name}")
            try:
                self._llm_model.cpu()
                del self._llm_model
            except Exception as e:
                logger.warning(f"Error freeing LLM model: {e}")
            self._llm_model = None
        
        if self._llm_tokenizer is not None:
            del self._llm_tokenizer
            self._llm_tokenizer = None
        
        self._current_llm_name = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def free_embedder(self):
        """Free the embedder model from memory."""
        if self._embedder_model is not None:
            logger.info(f"Freeing embedder: {self._current_embedder_name}")
            try:
                del self._embedder_model
            except Exception as e:
                logger.warning(f"Error freeing embedder model: {e}")
            self._embedder_model = None
        
        self._current_embedder_name = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def free_all(self):
        """Free all loaded models from memory."""
        self.free_llm()
        self.free_embedder()
    
    def chat_completion(
        self,
        system_prompt: Optional[str],
        history: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Generate a chat completion using the local LLM.
        
        This method provides an interface compatible with openai_chat_completion.
        
        Args:
            system_prompt: System prompt for the conversation (can be None)
            history: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            model_name: Optional model name override
            
        Returns:
            Generated text response
        """
        model, tokenizer = self.load_llm(model_name)
        
        # Build messages list
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(model.device)
        
        # Configure generation
        generation_config = GenerationConfig(
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            generation = model.generate(
                **model_inputs,
                generation_config=generation_config,
            )
        
        # Decode only the generated part
        generated_ids = generation[:, model_inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        logger.debug(f"Local LLM prompt:\n{prompt}\nResponse: {generated_text}")
        
        return generated_text


def get_local_llm_manager() -> LocalLLMManager:
    """Convenience function to get the LocalLLMManager singleton."""
    return LocalLLMManager.get_instance()


def local_chat_completion(
    system_prompt: Optional[str],
    history: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """
    Convenience function for local chat completion.
    
    This function provides a drop-in replacement for openai_chat_completion.
    
    Args:
        system_prompt: System prompt for the conversation (can be None)
        history: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
    """
    manager = get_local_llm_manager()
    return manager.chat_completion(
        system_prompt=system_prompt,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
    )

