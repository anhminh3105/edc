#!/bin/bash
# =============================================================================
# Local LLM Configuration Script
# =============================================================================
# This script configures environment variables for running EDC with local
# HuggingFace models instead of OpenAI API.
#
# Usage: source export_local_llm.sh
#
# Optimized for Tesla P100 GPU with 16GB VRAM using 4-bit quantization.
# =============================================================================

# Enable local LLM mode
export USE_LOCAL_LLM=true

# =============================================================================
# LLM Model Configuration
# =============================================================================
# Recommended models for P100 (16GB VRAM) with 4-bit quantization:
#   - mistralai/Mistral-7B-Instruct-v0.3  (~5GB VRAM) - Default, excellent quality
#   - Qwen/Qwen2.5-7B-Instruct            (~5GB VRAM) - Excellent quality
#   - microsoft/Phi-3.5-mini-instruct     (~3GB VRAM) - Good quality, smaller
#
# For smaller VRAM requirements (no quantization needed):
#   - Qwen/Qwen2.5-3B-Instruct            (~6GB VRAM)
#   - microsoft/Phi-3-mini-4k-instruct    (~8GB VRAM)
# =============================================================================
export LOCAL_LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# =============================================================================
# Embedder Model Configuration
# =============================================================================
# Lightweight embedder models (replacing e5-mistral-7b):
#   - BAAI/bge-small-en-v1.5     (~130MB) - Default, good quality
#   - BAAI/bge-base-en-v1.5      (~440MB) - Better quality
#   - sentence-transformers/all-MiniLM-L6-v2  (~80MB) - Fastest
# =============================================================================
export LOCAL_EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"

# =============================================================================
# Quantization Configuration
# =============================================================================
# Options:
#   - 4bit  : 4-bit quantization via bitsandbytes (recommended for P100)
#   - 8bit  : 8-bit quantization via bitsandbytes
#   - none  : No quantization (requires more VRAM)
# =============================================================================
export LOCAL_LLM_QUANTIZE="4bit"

# =============================================================================
# Optional: HuggingFace Cache Directory
# =============================================================================
# Uncomment and modify to specify a custom cache directory for model downloads
# export HF_HOME="/path/to/your/huggingface/cache"
# export TRANSFORMERS_CACHE="/path/to/your/transformers/cache"

# =============================================================================
# Print Configuration Summary
# =============================================================================
echo "=============================================="
echo "Local LLM Configuration Loaded"
echo "=============================================="
echo "USE_LOCAL_LLM:       $USE_LOCAL_LLM"
echo "LOCAL_LLM_MODEL:     $LOCAL_LLM_MODEL"
echo "LOCAL_EMBEDDER_MODEL: $LOCAL_EMBEDDER_MODEL"
echo "LOCAL_LLM_QUANTIZE:  $LOCAL_LLM_QUANTIZE"
echo "=============================================="
echo ""
echo "To run EDC with local models:"
echo "  python run.py --input_text_file_path ./datasets/example.txt"
echo ""

