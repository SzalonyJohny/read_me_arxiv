"""Paper Reader: arXiv paper -> clean two-pane reader -> priced, accepted narration."""

import os

# Must be set before CUDA initializes: lets PyTorch grow memory segments instead of fragmenting,
# which matters for the large batched Qwen3-TTS generations (see tts/qwen_engine.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
