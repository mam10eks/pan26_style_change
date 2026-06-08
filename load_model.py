#!/usr/bin/env python3
"""
Construct the trained AVClassifier and load its weights.

The backbone (HuggingFace) is downloaded on first use into the standard
HF cache (``~/.cache/huggingface``). The custom head + conditioning
embeddings come from a state_dict downloaded separately via
``download_weights.py`` from a Google Drive link supplied by the repo
owner.
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

# Backbone HF identifier. Override via the BACKBONE_HF_NAME environment
# variable. Default matches the model used to train the bundled weights.
BACKBONE_HF_NAME = os.environ.get(
	"BACKBONE_HF_NAME", "microsoft/deberta-v3-large"
)

# Path to the trained state_dict. Default is repo-relative; override via
# CHECKPOINT_PATH if the weights live elsewhere.
CHECKPOINT_PATH = Path(os.environ.get(
	"CHECKPOINT_PATH", "weights/av_classifier.pt"
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
			f"Trained checkpoint not found at: {CHECKPOINT_PATH}\n"
			f"\n"
			f"To download from Google Drive, run:\n"
			f"    python download_weights.py --url <GOOGLE_DRIVE_LINK>\n"
			f"\n"
			f"Or set CHECKPOINT_PATH to an existing file:\n"
			f"    CHECKPOINT_PATH=/path/to/weights.pt python pan26_runner.py ...\n"
		)

	# Load tokenizer + backbone. First use will download from HuggingFace
	# (~280 MB for mdeberta-v3-base, ~720 MB for deberta-v3-large) and
	# cache under ~/.cache/huggingface for subsequent runs.
	# dtype is set explicitly to FP32 to avoid the partial-FP16 mismatch
	# with the trainable head we ran into during training. The parameter
	# was renamed from `torch_dtype` to `dtype` in transformers 4.46;
	# fix_mistral_regex=True silences a misleading warning that fires
	# for SentencePiece tokenizers (where the Mistral BPE bug doesn't
	# apply). Both flags are wrapped in try/except for cross-version use.
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
	# doesn't declare sep_token_id / cls_token_id as Python attributes
	# (direct access raises), so we use getattr with a default.
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

	# Sanity check: warn on unexpected keys (real bug if same arch as training,
	# expected if backbone was swapped). Missing keys are tolerated — they
	# correspond to frozen backbone params loaded from HF cache.
	if result.unexpected_keys:
		print(
			f"WARNING: {len(result.unexpected_keys)} unexpected keys in "
			f"checkpoint (first 5: {result.unexpected_keys[:5]}). "
			f"These will be ignored. This is expected if the backbone was "
			f"changed since training; investigate otherwise."
		)
	non_backbone_missing = [
		k for k in result.missing_keys if not k.startswith("backbone.")
	]
	if non_backbone_missing:
		print(
			f"WARNING: {len(non_backbone_missing)} non-backbone parameters "
			f"missing from checkpoint (first 5: {non_backbone_missing[:5]}). "
			f"These will use random initialization."
		)

	n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(
		f"Loaded checkpoint from {CHECKPOINT_PATH}. "
		f"Trainable params: {n_trainable:,}"
	)

	model.to(device).eval()
	return model, tokenizer