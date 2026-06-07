#!/usr/bin/env python3
"""
Construct the trained AVClassifier inside the Docker container.

The backbone weights are loaded from the HuggingFace cache (mounted
into the container by TIRA via --mount-hf-model). The custom head and
conditioning embeddings come from a state_dict bundled in the image.

If the bundled checkpoint contains the full state dict (including
backbone), it overrides the HF-loaded backbone weights — that's fine,
the backbone is reloaded into the model regardless. If it contains
only the trainable subset (top layers + head + conditioning), the
frozen backbone layers stay at their HF-pretrained values.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Import the model definition. Adjust the import path if your file lives elsewhere.
from av_classifier import AVClassifier


# ───────────────────────────────────────────────────────────────────────
# Configuration — must match training-time values.
# ───────────────────────────────────────────────────────────────────────

# Backbone HF identifier. Mounted into the container via --mount-hf-model.
BACKBONE_HF_NAME = os.environ.get(
	"BACKBONE_HF_NAME", "microsoft/deberta-v3-large"
)

# Path inside the container where the trained state_dict is bundled.
CHECKPOINT_PATH = Path(os.environ.get(
	"CHECKPOINT_PATH", "/opt/model/av_classifier.pt"
))

# Architecture hyperparameters — must match the training-time AVClassifier.
ARCH_CONFIG = dict(
	num_unfrozen           = 2,
	dropout                = 0.1,
	use_segment_sim        = False,     # OFF at inference (no labels)
	# Length conditioning
	use_length_features    = True,
	length_bucket_emb_dim  = 32,
	# Source / domain / language conditioning
	use_source_features    = True,
	num_sources            = 5,        # = len(SOURCE_TO_ID) at training time
	source_emb_dim         = 32,
	use_domain_features    = True,
	num_domains            = 5,         # easy/fanfic/hard/medium/wiki
	domain_emb_dim         = 32,
	use_language_features  = True,
	num_languages          = 2,         # en/diff
	language_emb_dim       = 32,
)



def load_av_classifier(device: torch.device | None = None):
	"""Build the AVClassifier, load its trained weights, return (model, tokenizer)."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if not CHECKPOINT_PATH.exists():
		raise FileNotFoundError(
			f"Trained checkpoint not found at {CHECKPOINT_PATH}. "
			f"Set CHECKPOINT_PATH env var or copy the file to that location."
		)

	# Load tokenizer + backbone from the HuggingFace cache (no internet).
	# dtype is set explicitly to FP32 to avoid the partial-FP16 mismatch
	# with the trainable head we ran into during training. The parameter
	# was renamed from `torch_dtype` to `dtype` in transformers 4.46;
	# we fall back to the old name for older versions.
	# fix_mistral_regex=True silences a misleading warning that fires for
	# many tokenizers (DeBERTa-v3 included) but only actually matters for
	# Mistral's BPE regex pre-tokenization — DeBERTa uses SentencePiece,
	# so the flag is a no-op here. Tolerated cross-version via try/except.
	try:
		tokenizer = AutoTokenizer.from_pretrained(
			BACKBONE_HF_NAME, use_fast=True, fix_mistral_regex=True,
		)
	except TypeError:
		tokenizer = AutoTokenizer.from_pretrained(BACKBONE_HF_NAME, use_fast=True)
	try:
		backbone = AutoModel.from_pretrained(BACKBONE_HF_NAME, dtype=torch.float32)
	except TypeError:
		backbone = AutoModel.from_pretrained(BACKBONE_HF_NAME, torch_dtype=torch.float32)

	# Fix up tokenizer-side IDs that some configs leave unset. DebertaV2Config
	# doesn't even declare sep_token_id / cls_token_id as Python attributes
	# (direct access raises AttributeError, not returns None), so we use
	# getattr with a default and setattr to assign. Matches validate_model.py.
	for attr in ("sep_token_id", "cls_token_id", "eos_token_id"):
		if getattr(backbone.config, attr, None) is None:
			tok_val = getattr(tokenizer, attr, None)
			if tok_val is not None:
				setattr(backbone.config, attr, tok_val)

	# Build the classifier shell.
	model = AVClassifier(backbone_model=backbone, **ARCH_CONFIG)

	# Load trained weights. strict=False so the load works whether the
	# checkpoint stored the full state dict or only trainable parameters.
	state = torch.load(CHECKPOINT_PATH, map_location="cpu")
	result = model.load_state_dict(state, strict=False)

	# Sanity check: complain if there are *unexpected* keys (real bug),
	# but missing keys are tolerated (they correspond to the frozen
	# backbone params that came from HF in the first place).
	if result.unexpected_keys:
		print(
			f"WARNING: {len(result.unexpected_keys)} unexpected keys in "
			f"checkpoint (first 5: {result.unexpected_keys[:5]}). "
			f"These will be ignored."
		)
	print(
		f"Loaded checkpoint from {CHECKPOINT_PATH}. "
		f"Trainable params in optimizer: "
		f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
	)

	model.to(device).eval()
	return model, tokenizer
