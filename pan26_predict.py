#!/usr/bin/env python3
"""
Run a trained pairwise style-change classifier on a PAN 2026 input folder
and write predictions in the PAN evaluator format.

Two thresholds
--------------
- ``threshold``         (default 0.5): used to produce the FINAL output
                        labels. ``label=1`` (change) when P_change >= threshold.
                        Also used inside ``refine_by_evidence`` for
                        provisional run detection.

- ``merging_threshold`` (default = threshold): used ONLY inside
                        ``refine_by_concat`` to decide whether two
                        adjacent sentences belong to the same provisional
                        same-author run (i.e. whether to concatenate them
                        into a single side of the test pair). Two
                        sentences are treated as same-author for concat
                        purposes when ``P_change < merging_threshold``.
                        Set this LOWER than ``threshold`` (e.g. 0.2) to
                        be conservative about concat-merging — important
                        because a wrong merge in concat mode literally
                        glues different-author text together, corrupting
                        the input. Has NO effect in evidence-window mode
                        (where every query is an independent single-pair
                        test, so the aggregation handles wrong-run noise
                        symmetrically).

Cross-pair evidence aggregation
-------------------------------
For every potential boundary (s_i, s_{i+1}) the model produces one
initial probability. Two optional refinement strategies use the
provisional predictions to identify same-author runs around the
boundary, then collect more evidence from those runs:

  evidence-window (--evidence-window > 0)
      Tests additional single-sentence pairs: (s_j, s_{i+1}) for prior
      in-run sentences j, and/or (s_i, s_k) for next in-run sentences k.
      Aggregates all probabilities (original + extras) via
      --evidence-aggregate (mean / max / min / median / logodds).

  concat-evidence (--concat-left-budget > 0)
      Builds maximal-context single pairs by concatenating same-author
      sentences on each side of the boundary, up to a token budget, then
      tests the merged pair. With --concat-shared-budget set, uses a
      single total budget split between the two sides instead of
      independent per-side budgets.

The two modes are mutually exclusive — pick one per inference run.

Conditioning
------------
If the model was trained with conditioning embeddings, pass the integer
IDs via --source-id / --domain-id / --language-id. They are forwarded to
the model as constant tensors on every batch.

Example:
    python pan26_predict.py \\
        -i /data/pan26/medium/test \\
        -o /preds/medium \\
        -m ./checkpoints/best \\
        --max-length 250 --batch-size 32 \\
        --domain-id 3 --language-id 0 \\
        --evidence-window 5 --evidence-direction both \\
        --threshold 0.5 --merging-threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# --------------------------------------------------------------------------- #
# Sentence chunking for long inputs
# --------------------------------------------------------------------------- #


def chunk_text(text: str, tokenizer, max_tokens: int, stride: int) -> List[str]:
	"""Split text into overlapping max_tokens-sized windows; pass through if short."""
	if max_tokens <= 0:
		return [text]
	if stride >= max_tokens:
		raise ValueError("--stride must be strictly smaller than --max-tokens")

	try:
		enc = tokenizer(
			text,
			add_special_tokens=False,
			return_offsets_mapping=True,
			truncation=False,
		)
		input_ids = enc["input_ids"]
		offsets = enc["offset_mapping"]
		use_offsets = True
	except (TypeError, NotImplementedError, ValueError):
		input_ids = tokenizer.encode(text, add_special_tokens=False)
		offsets = None
		use_offsets = False

	if len(input_ids) <= max_tokens:
		return [text]

	step = max_tokens - stride
	chunks: List[str] = []
	i, n = 0, len(input_ids)
	while i < n:
		end = min(i + max_tokens, n)
		if use_offsets:
			start_char = offsets[i][0]
			end_char = offsets[end - 1][1]
			chunk = (
				text[start_char:end_char] if end_char > start_char
				else tokenizer.decode(input_ids[i:end], skip_special_tokens=True)
			)
		else:
			chunk = tokenizer.decode(input_ids[i:end], skip_special_tokens=True)
		chunk = chunk.strip()
		if chunk:
			chunks.append(chunk)
		if end == n:
			break
		i += step

	deduped: List[str] = []
	for c in chunks:
		if not deduped or deduped[-1] != c:
			deduped.append(c)
	return deduped or [text]


# --------------------------------------------------------------------------- #
# Inference dataset & collator
# --------------------------------------------------------------------------- #


class InferencePairDataset(Dataset):
	"""Tokenises (s1, s2) into [CLS] s1 [SEP] s2 [SEP], mirroring training."""

	def __init__(
		self,
		pairs,
		tokenizer,
		max_length_per_side: int,
		max_total_length: Optional[int] = None,
	):
		self.pairs = pairs
		self.tokenizer = tokenizer
		self.max_length_per_side = max_length_per_side

		if max_total_length is None:
			mml = getattr(tokenizer, "model_max_length", None)
			if mml is None or mml > 1_000_000:
				mml = 512
			max_total_length = int(mml)
		self.max_content_length = max(2, max_total_length - 3)

		cls_id = tokenizer.cls_token_id
		sep_id = tokenizer.sep_token_id
		if cls_id is None or sep_id is None:
			raise ValueError(
				"tokenizer.cls_token_id / sep_token_id is None; cannot build "
				"a [CLS] s1 [SEP] s2 [SEP] sequence."
			)
		vocab_size = getattr(tokenizer, "vocab_size", None)
		if vocab_size is not None:
			for name, tid in [("cls", cls_id), ("sep", sep_id)]:
				if tid >= vocab_size or tid < 0:
					raise ValueError(
						f"{name}_token_id ({tid}) is outside tokenizer "
						f"vocab_size ({vocab_size})."
					)
		self._cls_id = int(cls_id)
		self._sep_id = int(sep_id)

	def __len__(self) -> int:
		return len(self.pairs)

	def __getitem__(self, idx: int):
		s1, s2 = self.pairs[idx]
		tok = self.tokenizer
		tokens1 = tok(
			s1, truncation=True, max_length=self.max_length_per_side,
			add_special_tokens=False,
		)["input_ids"]
		tokens2 = tok(
			s2, truncation=True, max_length=self.max_length_per_side,
			add_special_tokens=False,
		)["input_ids"]

		budget = self.max_content_length
		total = len(tokens1) + len(tokens2)
		if total > budget:
			n1 = len(tokens1); n2 = len(tokens2)
			keep1 = max(1, int(round(budget * n1 / max(total, 1))))
			keep2 = max(1, budget - keep1)
			if keep1 + keep2 > budget:
				if keep1 >= keep2:
					keep1 = budget - keep2
				else:
					keep2 = budget - keep1
			tokens1 = tokens1[-keep1:]
			tokens2 = tokens2[:keep2]

		input_ids = (
			[self._cls_id] + tokens1 + [self._sep_id]
			+ tokens2 + [self._sep_id]
		)
		attention_mask = [1] * len(input_ids)
		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_mask, dtype=torch.long),
		}


def make_collator(tokenizer):
	"""Pad-only collator. Tokenisation already happened in the Dataset."""
	pad_id = tokenizer.pad_token_id
	if pad_id is None:
		pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

	def collate(batch):
		max_len = max(b["input_ids"].size(0) for b in batch)
		bsz = len(batch)
		input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
		attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
		for i, b in enumerate(batch):
			n = b["input_ids"].size(0)
			input_ids[i, :n] = b["input_ids"]
			attention_mask[i, :n] = b["attention_mask"]
		return {"input_ids": input_ids, "attention_mask": attention_mask}

	return collate


# --------------------------------------------------------------------------- #
# Conditioning helpers
# --------------------------------------------------------------------------- #


def _attach_conditioning(
	batch: dict,
	device: torch.device,
	*,
	language_id: Optional[int],
	source_id:   Optional[int],
	domain_id:   Optional[int],
) -> dict:
	"""Add constant-per-batch conditioning tensors when provided."""
	bsz = batch["input_ids"].size(0)
	for key, value in (
		("language_ids", language_id),
		("source_ids",   source_id),
		("domain_ids",   domain_id),
	):
		if value is not None:
			batch[key] = torch.full(
				(bsz,), int(value), dtype=torch.long, device=device,
			)
	return batch


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


# Match anything between "problem-" and ".txt", not just digits, to handle
# string IDs (e.g. zero-padded or alphanumeric) the same way the official
# verifier does (os.path.basename(file)[8:-4]).
PROBLEM_RE = re.compile(r"problem-(.+)\.txt$")


def _problem_sort_key(p: Path):
	"""Sort problem files by integer ID if possible, otherwise by string."""
	m = PROBLEM_RE.search(p.name)
	if m is None:
		return (1, p.name)
	pid = m.group(1)
	try:
		return (0, int(pid))
	except ValueError:
		return (1, pid)


def read_sentences(problem_file: Path) -> List[str]:
	"""Read sentences for model input — filters empty lines.

	Note: this may produce a different count than the official verifier's
	``raw.count("\\n") + 1`` if the file has blank lines or a trailing
	newline. The output-length reconciliation happens in run_prediction,
	which pads or truncates predictions to match the verifier's count.
	"""
	with open(problem_file, "r", newline="", encoding="utf-8") as f:
		raw = f.read()
	return [line for line in raw.splitlines() if line.strip()]


def count_chunks_verifier_style(problem_file: Path) -> int:
	"""Mirror the PAN verifier: ``open(path, 'r', newline='')`` then
	``read().count('\\n') + 1``. This is the count of sentences the
	verifier will use to validate the output length.
	"""
	with open(problem_file, "r", newline="", encoding="utf-8") as f:
		raw = f.read()
	return raw.count("\n") + 1


@torch.inference_mode()
def predict_probs(
	pairs: List[Tuple[str, str]],
	model,
	tokenizer,
	device: torch.device,
	batch_size: int,
	max_length_per_side: int,
	language_id: Optional[int] = None,
	source_id:   Optional[int] = None,
	domain_id:   Optional[int] = None,
) -> torch.Tensor:
	"""Return a 1-D tensor of P(label=1) for each input pair."""
	if not pairs:
		return torch.empty(0)

	loader = DataLoader(
		InferencePairDataset(pairs, tokenizer, max_length_per_side),
		batch_size=batch_size,
		shuffle=False,
		collate_fn=make_collator(tokenizer),
	)

	probs_chunks: List[torch.Tensor] = []
	for batch in loader:
		batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
		batch = _attach_conditioning(
			batch, device,
			language_id=language_id, source_id=source_id, domain_id=domain_id,
		)
		out = model(**batch)
		if isinstance(out, dict):
			logits = out["logits"]
		elif hasattr(out, "logits"):
			logits = out.logits
		else:
			logits = out[0] if isinstance(out, (tuple, list)) else out
		if logits.shape[-1] == 1:                # single-logit / BCE head
			p1 = torch.sigmoid(logits.squeeze(-1))
		else:                                    # 2-class softmax head
			p1 = torch.softmax(logits, dim=-1)[:, 1]
		probs_chunks.append(p1.detach().cpu())
	return torch.cat(probs_chunks, dim=0)


def _probs_to_merge_preds(probs: Iterable[float], merging_threshold: float) -> List[int]:
	"""Convert P(change) to provisional boundary predictions for run detection.

	merge_preds[k] = 1  → BOUNDARY between s_k and s_{k+1}  (do NOT merge)
	merge_preds[k] = 0  → SAME-author        (merge into same run)

	Uses ``merging_threshold`` rather than the output threshold, so refinement
	logic can be conservative about which sentences it treats as same-author.
	"""
	return [1 if p >= merging_threshold else 0 for p in probs]


# --------------------------------------------------------------------------- #
# Cross-pair evidence aggregation: extra single-pair queries within runs
# --------------------------------------------------------------------------- #


def _aggregate_probs(probs: List[float], method: str) -> float:
	"""Combine multiple P(change) estimates into one."""
	if not probs:
		raise ValueError("Cannot aggregate an empty list of probabilities.")
	if method == "mean":
		return sum(probs) / len(probs)
	if method == "max":
		return max(probs)
	if method == "min":
		return min(probs)
	if method == "median":
		s = sorted(probs)
		n = len(s)
		return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
	if method == "logodds":
		eps = 1e-6
		log_odds = 0.0
		for p in probs:
			p_clipped = min(max(p, eps), 1.0 - eps)
			log_odds += math.log(p_clipped / (1.0 - p_clipped))
		return 1.0 / (1.0 + math.exp(-log_odds))
	raise ValueError(f"Unknown aggregate method: {method}")


@torch.inference_mode()
def refine_by_evidence(
	sentences: List[str],
	initial_probs: List[float],
	model,
	tokenizer,
	device: torch.device,
	batch_size: int,
	max_length_per_side: int,
	window: int,
	aggregate: str,
	threshold: float,
	max_iterations: int,
	direction: str = "backward",
	language_id: Optional[int] = None,
	source_id:   Optional[int] = None,
	domain_id:   Optional[int] = None,
) -> List[float]:
	"""Refine per-pair P(change) by collecting more single-pair evidence.

	Same-author runs are determined by the OUTPUT ``threshold``, not by a
	stricter merging threshold: each evidence query is an independent
	single-pair test that contributes valid evidence in both directions
	via the aggregation step, so wrongly-merged neighbors don't corrupt
	the result the way they would for concat-evidence.
	"""
	n_sent = len(sentences)
	n_pairs = n_sent - 1
	if n_pairs <= 0 or window <= 0:
		return list(initial_probs)
	if direction not in ("backward", "forward", "both"):
		raise ValueError(f"Unknown direction: {direction}")

	do_back = direction in ("backward", "both")
	do_fwd  = direction in ("forward",  "both")
	probs = list(initial_probs)

	for _it in range(max_iterations):
		preds = _probs_to_merge_preds(probs, threshold)

		# First sentence index of the run containing s_k.
		run_start = [0] * n_sent
		for k in range(1, n_sent):
			run_start[k] = k if preds[k - 1] == 1 else run_start[k - 1]
		# First sentence index AFTER the run containing s_k (exclusive end).
		run_end = [n_sent] * n_sent
		for k in range(n_sent - 2, -1, -1):
			run_end[k] = k + 1 if preds[k] == 1 else run_end[k + 1]

		extra_pairs: List[Tuple[str, str]] = []
		extra_owner: List[int] = []
		for i in range(n_pairs):
			if do_back:
				j_lo = max(run_start[i], i - window)
				for j in range(j_lo, i):
					extra_pairs.append((sentences[j], sentences[i + 1]))
					extra_owner.append(i)
			if do_fwd:
				k_hi = min(run_end[i + 1], i + 1 + window + 1)
				for k in range(i + 2, k_hi):
					extra_pairs.append((sentences[i], sentences[k]))
					extra_owner.append(i)

		if not extra_pairs:
			break

		extra_probs = predict_probs(
			extra_pairs, model, tokenizer, device, batch_size,
			max_length_per_side,
			language_id=language_id, source_id=source_id, domain_id=domain_id,
		).tolist()

		by_pair: dict = defaultdict(list)
		for p, owner in zip(extra_probs, extra_owner):
			by_pair[owner].append(p)

		new_probs = [
			_aggregate_probs([initial_probs[i]] + by_pair.get(i, []), aggregate)
			for i in range(n_pairs)
		]

		new_preds = _probs_to_merge_preds(new_probs, threshold)
		converged = new_preds == preds
		probs = new_probs
		if converged:
			break

	return probs


# --------------------------------------------------------------------------- #
# Cross-pair evidence aggregation: maximal-context single pair via concat
# --------------------------------------------------------------------------- #


def _build_concat_side(
	sentences: List[str],
	token_cache: List[List[int]],
	anchor_idx: int,
	run_lo: int,
	run_hi: int,
	budget: int,
	tokenizer,
	side: str,
) -> str:
	"""Concatenate sentences inside a same-author run up to a token budget."""
	if side == "left":
		order = range(anchor_idx, run_lo - 1, -1)
		clip_start = True
	elif side == "right":
		order = range(anchor_idx, run_hi + 1)
		clip_start = False
	else:
		raise ValueError(f"side must be 'left' or 'right', got {side!r}")

	kept_indices: List[int] = []
	kept_total = 0
	truncated_at_idx: Optional[int] = None
	truncated_text: Optional[str] = None

	for j in order:
		n_j = len(token_cache[j])
		if kept_indices and kept_total + n_j > budget:
			remaining = budget - kept_total
			if remaining > 0:
				ids = (
					token_cache[j][-remaining:] if clip_start
					else token_cache[j][:remaining]
				)
				txt = tokenizer.decode(ids, skip_special_tokens=True).strip()
				if txt:
					kept_indices.append(j)
					truncated_at_idx = j
					truncated_text = txt
					kept_total += remaining
			break
		kept_indices.append(j)
		kept_total += n_j
		if kept_total >= budget:
			break

	ordered = sorted(kept_indices)
	parts = [
		truncated_text if (j == truncated_at_idx) else sentences[j]
		for j in ordered
	]
	return " ".join(p for p in parts if p)


@torch.inference_mode()
def refine_by_concat(
	sentences: List[str],
	initial_probs: List[float],
	model,
	tokenizer,
	device: torch.device,
	batch_size: int,
	left_budget: int,
	right_budget: int,
	merging_threshold: float,
	max_iterations: int,
	direction: str = "both",
	language_id: Optional[int] = None,
	source_id:   Optional[int] = None,
	domain_id:   Optional[int] = None,
	shared_budget: int = 0,
	shared_anchor: str = "auto",
	shared_min_per_side: int = 16,
) -> List[float]:
	"""Re-test each potential boundary with a maximal-context single pair.

	Same-author runs are determined by ``merging_threshold`` (same convention
	as refine_by_evidence).
	"""
	n_sent = len(sentences)
	n_pairs = n_sent - 1
	if n_pairs <= 0:
		return list(initial_probs)
	if left_budget <= 0 or right_budget <= 0:
		raise ValueError("left_budget and right_budget must be positive")
	if direction not in ("left", "right", "both"):
		raise ValueError(f"direction must be 'left'/'right'/'both', got {direction!r}")
	if shared_budget < 0:
		raise ValueError("shared_budget must be >= 0 (0 disables shared-budget mode)")
	if shared_anchor not in ("auto", "left", "right"):
		raise ValueError(f"shared_anchor must be 'auto'/'left'/'right'")
	if shared_budget > 0 and shared_min_per_side * 2 > shared_budget:
		raise ValueError(
			f"shared_min_per_side ({shared_min_per_side}) is too large for "
			f"shared_budget ({shared_budget})"
		)

	grow_left  = direction in ("left",  "both")
	grow_right = direction in ("right", "both")

	token_cache: List[List[int]] = [
		tokenizer(s, add_special_tokens=False, truncation=False)["input_ids"]
		for s in sentences
	]

	def grow_one_side(
		anchor_idx: int, run_lo: int, run_hi: int, budget: int, side: str,
		do_grow: bool,
	) -> str:
		if do_grow:
			return _build_concat_side(
				sentences, token_cache,
				anchor_idx=anchor_idx, run_lo=run_lo, run_hi=run_hi,
				budget=budget, tokenizer=tokenizer, side=side,
			)
		ids = (
			token_cache[anchor_idx][-budget:] if side == "left"
			else token_cache[anchor_idx][:budget]
		)
		return tokenizer.decode(ids, skip_special_tokens=True).strip()

	probs = list(initial_probs)
	prev_preds: Optional[List[int]] = None

	for _it in range(max_iterations):
		preds = _probs_to_merge_preds(probs, merging_threshold)
		if preds == prev_preds:
			break
		prev_preds = preds

		run_start = [0] * n_sent
		for k in range(1, n_sent):
			run_start[k] = k if preds[k - 1] == 1 else run_start[k - 1]
		run_end = [n_sent - 1] * n_sent
		for k in range(n_sent - 2, -1, -1):
			run_end[k] = k if preds[k] == 1 else run_end[k + 1]

		test_pairs: List[Tuple[str, str]] = []
		for i in range(n_pairs):
			l_run_lo, l_run_hi = run_start[i], i
			r_run_lo, r_run_hi = i + 1, run_end[i + 1]

			if shared_budget == 0:
				left_text = grow_one_side(
					i, l_run_lo, l_run_hi, left_budget, "left", grow_left,
				)
				right_text = grow_one_side(
					i + 1, r_run_lo, r_run_hi, right_budget, "right", grow_right,
				)
			else:
				if shared_anchor == "auto":
					n_left_anchor  = len(token_cache[i])
					n_right_anchor = len(token_cache[i + 1])
					anchor = "left" if n_left_anchor <= n_right_anchor else "right"
				else:
					anchor = shared_anchor

				if anchor == "right":
					right_text = grow_one_side(
						i + 1, r_run_lo, r_run_hi, right_budget, "right",
						grow_right,
					)
					used = len(tokenizer(
						right_text, add_special_tokens=False, truncation=False,
					)["input_ids"])
					left_alloc = max(shared_min_per_side, shared_budget - used)
					left_text = grow_one_side(
						i, l_run_lo, l_run_hi, left_alloc, "left", grow_left,
					)
				else:
					left_text = grow_one_side(
						i, l_run_lo, l_run_hi, left_budget, "left", grow_left,
					)
					used = len(tokenizer(
						left_text, add_special_tokens=False, truncation=False,
					)["input_ids"])
					right_alloc = max(shared_min_per_side, shared_budget - used)
					right_text = grow_one_side(
						i + 1, r_run_lo, r_run_hi, right_alloc, "right",
						grow_right,
					)

			if not left_text:
				left_text = sentences[i]
			if not right_text:
				right_text = sentences[i + 1]
			test_pairs.append((left_text, right_text))

		per_side_cap = (
			shared_budget if shared_budget > 0
			else max(left_budget, right_budget)
		)
		new_probs = predict_probs(
			test_pairs, model, tokenizer, device, batch_size,
			max_length_per_side=per_side_cap,
			language_id=language_id, source_id=source_id, domain_id=domain_id,
		).tolist()
		probs = new_probs

	return probs


# --------------------------------------------------------------------------- #
# Per-problem prediction
# --------------------------------------------------------------------------- #


def predict_problem(
	sentences: List[str],
	model,
	tokenizer,
	device: torch.device,
	batch_size: int,
	max_length_per_side: int,
	chunk_max_tokens: int,
	chunk_stride: int,
	threshold: float,
	merging_threshold: float,
	aggregate: str,
	invert_labels: bool,
	# Conditioning
	language_id: Optional[int] = None,
	source_id:   Optional[int] = None,
	domain_id:   Optional[int] = None,
	# Evidence-window
	evidence_window: int = 0,
	evidence_iterations: int = 1,
	evidence_aggregate: str = "mean",
	evidence_direction: str = "backward",
	# Concat-evidence
	concat_left_budget: int = 0,
	concat_right_budget: int = 0,
	concat_iterations: int = 2,
	concat_direction: str = "both",
	concat_shared_budget: int = 0,
	concat_shared_anchor: str = "auto",
	concat_shared_min_per_side: int = 16,
) -> List[int]:
	"""Return a list of binary predictions, one per consecutive-sentence pair."""
	n_pairs_expected = max(0, len(sentences) - 1)
	if n_pairs_expected == 0:
		return []

	sentence_chunks: List[List[str]] = [
		chunk_text(s, tokenizer, chunk_max_tokens, chunk_stride)
		for s in sentences
	]

	flat_pairs: List[Tuple[str, str]] = []
	owner: List[int] = []
	for i in range(n_pairs_expected):
		for c1 in sentence_chunks[i]:
			for c2 in sentence_chunks[i + 1]:
				flat_pairs.append((c1, c2))
				owner.append(i)

	flat_probs = predict_probs(
		flat_pairs, model, tokenizer, device, batch_size,
		max_length_per_side,
		language_id=language_id, source_id=source_id, domain_id=domain_id,
	)

	# Chunk aggregation: combine per-chunk-pair probs into per-original-pair.
	if aggregate == "mean":
		agg = torch.zeros(n_pairs_expected)
		counts = torch.zeros(n_pairs_expected)
		for p, idx in zip(flat_probs.tolist(), owner):
			agg[idx] += p
			counts[idx] += 1
		per_pair_prob = agg / counts.clamp(min=1)
	elif aggregate == "max":
		per_pair_prob = torch.full((n_pairs_expected,), -1.0)
		for p, idx in zip(flat_probs.tolist(), owner):
			if p > per_pair_prob[idx]:
				per_pair_prob[idx] = p
	else:
		raise ValueError(f"Unknown --aggregate value: {aggregate}")

	initial_probs = per_pair_prob.tolist()

	# Cross-pair evidence aggregation. Concat uses `merging_threshold`
	# (conservative, because wrong merges contaminate the input text).
	# Evidence-window uses `threshold` (no contamination risk — each
	# query is an independent single-pair test).
	if concat_left_budget > 0:
		final_probs = refine_by_concat(
			sentences=sentences,
			initial_probs=initial_probs,
			model=model,
			tokenizer=tokenizer,
			device=device,
			batch_size=batch_size,
			left_budget=concat_left_budget,
			right_budget=concat_right_budget or max_length_per_side,
			merging_threshold=merging_threshold,
			max_iterations=concat_iterations,
			direction=concat_direction,
			language_id=language_id, source_id=source_id, domain_id=domain_id,
			shared_budget=concat_shared_budget,
			shared_anchor=concat_shared_anchor,
			shared_min_per_side=concat_shared_min_per_side,
		)
	elif evidence_window > 0:
		final_probs = refine_by_evidence(
			sentences=sentences,
			initial_probs=initial_probs,
			model=model,
			tokenizer=tokenizer,
			device=device,
			batch_size=batch_size,
			max_length_per_side=max_length_per_side,
			window=evidence_window,
			aggregate=evidence_aggregate,
			threshold=threshold,
			max_iterations=evidence_iterations,
			direction=evidence_direction,
			language_id=language_id, source_id=source_id, domain_id=domain_id,
		)
	else:
		final_probs = initial_probs

	# Final output uses the (looser) `threshold`, not `merging_threshold`.
	preds = [1 if p >= threshold else 0 for p in final_probs]
	if invert_labels:
		preds = [1 - p for p in preds]
	return preds


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def _resolve_device(device: Optional[str]) -> "torch.device":
	if device:
		return torch.device(device)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def run_prediction(
	input_dir,
	output_dir,
	model,
	tokenizer=None,
	*,
	# Inference shape
	max_length: int = 250,
	max_tokens: int = 0,
	stride: int = 64,
	batch_size: int = 32,
	threshold: float = 0.5,
	merging_threshold: Optional[float] = None,
	aggregate: str = "mean",
	invert_labels: bool = False,
	# Conditioning
	language_id: Optional[int] = None,
	source_id:   Optional[int] = None,
	domain_id:   Optional[int] = None,
	# Evidence-window
	evidence_window: int = 0,
	evidence_iterations: int = 1,
	evidence_aggregate: str = "mean",
	evidence_direction: str = "backward",
	# Concat-evidence
	concat_left_budget: int = 0,
	concat_right_budget: int = 0,
	concat_iterations: int = 2,
	concat_direction: str = "both",
	concat_shared_budget: int = 0,
	concat_shared_anchor: str = "auto",
	concat_shared_min_per_side: int = 16,
	# Runtime
	device: Optional[str] = None,
	show_progress: bool = True,
	verbose: bool = True,
) -> dict:
	"""Notebook-friendly entry point. Returns summary stats dict.

	``merging_threshold`` defaults to ``threshold`` when not specified
	(preserves single-threshold behavior). Set it lower than threshold
	(e.g. 0.2 with threshold 0.5) for conservative run detection.
	"""
	if concat_left_budget > 0 and evidence_window > 0:
		raise ValueError(
			"concat_left_budget and evidence_window are mutually exclusive. "
			"Pick one cross-pair evidence strategy per inference run."
		)
	if merging_threshold is None:
		merging_threshold = threshold
	if not (0.0 <= merging_threshold <= 1.0):
		raise ValueError(f"merging_threshold must be in [0,1], got {merging_threshold}")
	if not (0.0 <= threshold <= 1.0):
		raise ValueError(f"threshold must be in [0,1], got {threshold}")

	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	resolved_device = _resolve_device(device)
	if verbose:
		print(f"Using device: {resolved_device}")

	if isinstance(model, (str, Path)):
		model_path = str(model)
		if tokenizer is None:
			tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
		model = AutoModelForSequenceClassification.from_pretrained(model_path)
		model.to(resolved_device).eval()
	else:
		if tokenizer is None:
			raise ValueError(
				"When passing a pre-loaded model, also pass its tokenizer."
			)
		try:
			model.to(resolved_device)
		except Exception:
			pass
		try:
			model.eval()
		except Exception:
			pass

	if verbose:
		cond_parts = []
		if language_id is not None: cond_parts.append(f"language_id={language_id}")
		if source_id   is not None: cond_parts.append(f"source_id={source_id}")
		if domain_id   is not None: cond_parts.append(f"domain_id={domain_id}")
		cond_str = (" | " + ", ".join(cond_parts)) if cond_parts else " | no conditioning"
		print(
			f"Tokenisation: per-side max_length={max_length}{cond_str}"
		)
		# merging_threshold only matters in concat-evidence mode — in
		# evidence-window mode and in raw single-pair mode, only the
		# output `threshold` is used.
		if concat_left_budget > 0:
			thr_note = (
				f"Thresholds: output={threshold}, merging={merging_threshold}"
				+ ("  (same as output)" if merging_threshold == threshold
				   else "  (stricter — conservative concat-merging)")
			)
		else:
			thr_note = (
				f"Threshold: output={threshold}  "
				f"(merging_threshold ignored — only used in concat-evidence mode)"
			)
		print(thr_note)
		if evidence_window > 0:
			print(
				f"Evidence-window: ON  (window={evidence_window}, "
				f"direction={evidence_direction}, "
				f"aggregate={evidence_aggregate}, "
				f"iterations={evidence_iterations})"
			)
		elif concat_left_budget > 0:
			mode_extra = ""
			if concat_shared_budget > 0:
				mode_extra = (
					f", SHARED_TOTAL={concat_shared_budget}, "
					f"anchor={concat_shared_anchor}, "
					f"min_per_side={concat_shared_min_per_side}"
				)
			print(
				f"Concat-evidence: ON  (direction={concat_direction}, "
				f"left_budget={concat_left_budget}, "
				f"right_budget={concat_right_budget or max_length}, "
				f"iterations={concat_iterations}{mode_extra})"
			)
		else:
			print("Cross-pair evidence aggregation: OFF (raw single-pair predictions)")

	problem_files = sorted(
		input_dir.glob("problem-*.txt"),
		key=_problem_sort_key,
	)
	if not problem_files:
		raise FileNotFoundError(f"No problem-*.txt files found in {input_dir}")

	iterator = (
		tqdm(problem_files, desc="Predicting", unit="doc")
		if show_progress else problem_files
	)

	n_changes_total = 0
	n_pairs_total = 0
	n_padded = 0
	n_truncated = 0
	for problem_file in iterator:
		match = PROBLEM_RE.search(problem_file.name)
		if match is None:
			print(f"WARNING: skipping non-matching filename: {problem_file.name}")
			continue
		pid = match.group(1)        # KEEP AS STRING — verifier uses string IDs
		sentences = read_sentences(problem_file)
		preds = predict_problem(
			sentences=sentences,
			model=model,
			tokenizer=tokenizer,
			device=resolved_device,
			batch_size=batch_size,
			max_length_per_side=max_length,
			chunk_max_tokens=max_tokens,
			chunk_stride=stride,
			threshold=threshold,
			merging_threshold=merging_threshold,
			aggregate=aggregate,
			invert_labels=invert_labels,
			language_id=language_id, source_id=source_id, domain_id=domain_id,
			evidence_window=evidence_window,
			evidence_iterations=evidence_iterations,
			evidence_aggregate=evidence_aggregate,
			evidence_direction=evidence_direction,
			concat_left_budget=concat_left_budget,
			concat_right_budget=concat_right_budget,
			concat_iterations=concat_iterations,
			concat_direction=concat_direction,
			concat_shared_budget=concat_shared_budget,
			concat_shared_anchor=concat_shared_anchor,
			concat_shared_min_per_side=concat_shared_min_per_side,
		)

		# Reconcile output length with the verifier's expectation. The PAN
		# verifier counts chunks as raw.count("\n") + 1, which differs from
		# splitlines() when the file has trailing newlines or blank lines.
		# We pad with 0 (no change) if our prediction array is too short,
		# truncate if too long. Padding with 0 is conservative — a phantom
		# "empty" sentence pair can't carry a style change in any meaningful
		# sense.
		expected_n_pairs = max(0, count_chunks_verifier_style(problem_file) - 1)
		if len(preds) < expected_n_pairs:
			preds = preds + [0] * (expected_n_pairs - len(preds))
			n_padded += 1
		elif len(preds) > expected_n_pairs:
			preds = preds[:expected_n_pairs]
			n_truncated += 1

		out_path = output_dir / f"solution-problem-{pid}.json"
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump({"changes": preds}, f)

		n_pairs_total += len(preds)
		n_changes_total += sum(preds)
		if show_progress and hasattr(iterator, "set_postfix"):
			iterator.set_postfix(pairs=n_pairs_total, changes=n_changes_total)

	summary = {
		"n_problems": len(problem_files),
		"n_pairs": n_pairs_total,
		"n_changes": n_changes_total,
		"change_rate": n_changes_total / max(n_pairs_total, 1),
		"output_dir": str(output_dir),
		"n_padded": n_padded,
		"n_truncated": n_truncated,
	}
	if verbose:
		print(
			f"Done. Wrote {summary['n_problems']} solution files to "
			f"{summary['output_dir']}\n"
			f"Total predicted change rate: "
			f"{summary['n_changes']}/{summary['n_pairs']} = "
			f"{summary['change_rate']:.3f}"
		)
		if n_padded or n_truncated:
			print(
				f"Length reconciliation: {n_padded} files padded with 0s, "
				f"{n_truncated} files truncated to match verifier's "
				f"raw.count('\\n')+1 sentence count."
			)
	return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
	parser = argparse.ArgumentParser(
		description="Generate PAN 2026 style-change predictions."
	)
	parser.add_argument("-i", "--input", required=True, type=Path)
	parser.add_argument("-o", "--output", required=True, type=Path)
	parser.add_argument("-m", "--model", required=True, type=str)
	parser.add_argument("-t", "--tokenizer", default=None, type=str)
	parser.add_argument("--max-length", type=int, default=250)
	parser.add_argument("--max-tokens", type=int, default=0)
	parser.add_argument("--stride", type=int, default=64)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--threshold", type=float, default=0.5,
		help="Final output threshold. P_change >= threshold => predict change (label=1).")
	parser.add_argument("--merging-threshold", type=float, default=None,
		help="Threshold used INSIDE concat-evidence refinement to decide "
		     "same-author runs (P_change < merging-threshold means treat as "
		     "same author for concat purposes). Defaults to --threshold. "
		     "Set lower (e.g. 0.2 with --threshold 0.5) for conservative "
		     "concat merging — sentences whose change probability sits "
		     "between the two thresholds are neither treated as boundaries "
		     "in the output nor as same-author for concat-evidence. "
		     "IGNORED in evidence-window mode (which uses the output "
		     "threshold for run detection) and in raw single-pair mode.")
	parser.add_argument("--aggregate", choices=["mean", "max"], default="mean")
	parser.add_argument("--invert-labels", action="store_true")

	# Conditioning
	parser.add_argument("--language-id", type=int, default=None)
	parser.add_argument("--source-id", type=int, default=None)
	parser.add_argument("--domain-id", type=int, default=None)

	# Evidence-window
	parser.add_argument("--evidence-window", type=int, default=0)
	parser.add_argument("--evidence-aggregate",
		choices=["mean", "max", "min", "median", "logodds"], default="mean")
	parser.add_argument("--evidence-iterations", type=int, default=1)
	parser.add_argument("--evidence-direction",
		choices=["backward", "forward", "both"], default="backward")

	# Concat-evidence
	parser.add_argument("--concat-left-budget", type=int, default=0)
	parser.add_argument("--concat-right-budget", type=int, default=0)
	parser.add_argument("--concat-iterations", type=int, default=2)
	parser.add_argument("--concat-direction",
		choices=["left", "right", "both"], default="both")
	parser.add_argument("--concat-shared-budget", type=int, default=0)
	parser.add_argument("--concat-shared-anchor",
		choices=["auto", "left", "right"], default="auto")
	parser.add_argument("--concat-shared-min-per-side", type=int, default=16)

	parser.add_argument("--device", default=None)
	args = parser.parse_args(list(argv) if argv is not None else None)

	tokenizer = AutoTokenizer.from_pretrained(
		args.tokenizer or args.model, use_fast=True
	)
	model = AutoModelForSequenceClassification.from_pretrained(args.model)

	run_prediction(
		input_dir=args.input,
		output_dir=args.output,
		model=model,
		tokenizer=tokenizer,
		max_length=args.max_length,
		max_tokens=args.max_tokens,
		stride=args.stride,
		batch_size=args.batch_size,
		threshold=args.threshold,
		merging_threshold=args.merging_threshold,
		aggregate=args.aggregate,
		invert_labels=args.invert_labels,
		language_id=args.language_id,
		source_id=args.source_id,
		domain_id=args.domain_id,
		evidence_window=args.evidence_window,
		evidence_iterations=args.evidence_iterations,
		evidence_aggregate=args.evidence_aggregate,
		evidence_direction=args.evidence_direction,
		concat_left_budget=args.concat_left_budget,
		concat_right_budget=args.concat_right_budget,
		concat_iterations=args.concat_iterations,
		concat_direction=args.concat_direction,
		concat_shared_budget=args.concat_shared_budget,
		concat_shared_anchor=args.concat_shared_anchor,
		concat_shared_min_per_side=args.concat_shared_min_per_side,
		device=args.device,
		show_progress=True,
		verbose=True,
	)


if __name__ == "__main__":
	main()
