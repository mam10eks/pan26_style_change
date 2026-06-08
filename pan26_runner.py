#!/usr/bin/env python3
"""
TIRA entry point for PAN 2026 Multi-Author Writing Style Analysis.

Invoked by TIRA as:
    pan26_runner.py -i INPUT-DIRECTORY -o OUTPUT-DIRECTORY

INPUT-DIRECTORY may either:
  (a) contain problem-*.txt files directly (single dataset — usually 'test'
      of one of easy/medium/hard); detect which by substring in the path.
  (b) contain easy/, medium/, hard/ subfolders (each with problem-*.txt);
      process each one and write predictions under matching subfolders of
      OUTPUT-DIRECTORY.

The model + tokenizer are loaded once and reused across all subfolders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from load_model     import load_av_classifier
from pan26_predict  import run_prediction


# ───────────────────────────────────────────────────────────────────────
# Training-time mappings. Must match the values used during training.
# ───────────────────────────────────────────────────────────────────────

# LANGUAGE_TO_ID = {
# 	"de": 0, "en": 1, "es": 2, "fr": 3, "id": 4,
# 	"it": 5, "nl": 6, "pt": 7, "diff": 8,
# }
# DOMAIN_TO_ID = {
# 	"easy": 0, "fanfic": 1, "hard": 2, "medium": 3, "wiki": 4,
# }
DOMAIN_TO_ID = {'easy': 0, 'fanfic': 1, 'hard': 2, 'medium': 3, 'wiki': 4}
LANGUAGE_TO_ID =  {'en': 0, 'diff': 1}


# ───────────────────────────────────────────────────────────────────────
# Inference defaults — same values used for validation runs.
# Adjust here if a different inference config is needed for the test set.
# ───────────────────────────────────────────────────────────────────────

INFERENCE_DEFAULTS = dict(
	max_length=250,
	batch_size=32,
	threshold=0.5,
	merging_threshold=0.5,
	# Concat-evidence with shared budget; fits a 512-token model exactly.
	concat_left_budget=493,
	concat_right_budget=493,
	concat_iterations=1,
	concat_direction="both",
	concat_shared_budget=509,
	concat_shared_anchor="auto",
	concat_shared_min_per_side=16,
)


# ───────────────────────────────────────────────────────────────────────
# Path detection
# ───────────────────────────────────────────────────────────────────────

def detect_dataset_from_path(path: Path) -> Optional[str]:
	"""Return 'easy' / 'medium' / 'hard' if the path contains that name."""
	parts = [p.lower() for p in path.parts]
	# Check parts first (most reliable), then full string fallback.
	for name in ("easy", "medium", "hard"):
		if name in parts:
			return name
	p_lower = str(path).lower()
	for name in ("easy", "medium", "hard"):
		if name in p_lower:
			return name
	return None


# Subdirectory name priority when multiple candidates exist at a given depth.
# Inside a difficulty folder, prefer 'test' (TIRA final eval), then validation
# variants, then train, then anything else. Avoids picking 'train' when both
# train and test happen to be mounted in the same image.
_SUBDIR_PRIORITY = {"test": 0, "validation": 1, "val": 1, "dev": 1, "train": 2}


def _subdir_sort_key(p: Path) -> tuple:
	return (_SUBDIR_PRIORITY.get(p.name.lower(), 9), p.name.lower())


def _has_problem_files(d: Path) -> bool:
	return any(d.glob("problem-*.txt"))


def _find_problem_dir(root: Path) -> Optional[Path]:
	"""Search ``root`` and up to two levels below for a directory that
	contains problem-*.txt files. Returns the first match in this order:

	    depth 0: root/problem-*.txt
	    depth 1: root/<sub>/problem-*.txt
	    depth 2: root/<sub>/<subsub>/problem-*.txt

	Within each depth, subdirectories are visited in this order:
	test > validation/val/dev > train > others (alphabetical). This
	handles the canonical TIRA layouts:

	    <difficulty>/problem-*.txt              (legacy)
	    <difficulty>/test/problem-*.txt         (current)
	    <difficulty>/train/problem-*.txt        (smoke-test)
	    <difficulty>/<x>/<y>/problem-*.txt      (defensive fallback)
	"""
	if not root.is_dir():
		return None
	if _has_problem_files(root):
		return root
	subdirs = sorted(
		(p for p in root.iterdir() if p.is_dir()),
		key=_subdir_sort_key,
	)
	# Depth 1
	for sub in subdirs:
		if _has_problem_files(sub):
			return sub
	# Depth 2
	for sub in subdirs:
		for subsub in sorted(
			(p for p in sub.iterdir() if p.is_dir()),
			key=_subdir_sort_key,
		):
			if _has_problem_files(subsub):
				return subsub
	return None


def find_dataset_subdirs(input_dir: Path) -> dict[str, Path]:
	"""Return {'easy': path, 'medium': path, 'hard': path} mapping each
	discovered difficulty to the directory actually containing its
	problem-*.txt files (which may be nested up to two levels deep)."""
	found = {}
	for name in ("easy", "medium", "hard"):
		difficulty_root = input_dir / name
		if not difficulty_root.is_dir():
			continue
		problem_dir = _find_problem_dir(difficulty_root)
		if problem_dir is not None:
			found[name] = problem_dir
	return found


# ───────────────────────────────────────────────────────────────────────
# Prediction dispatch
# ───────────────────────────────────────────────────────────────────────

def run_single_dataset(
	input_dir: Path,
	output_dir: Path,
	dataset_name: Optional[str],
	model,
	tokenizer,
) -> dict:
	"""Run prediction for one input folder containing problem-*.txt files."""
	output_dir.mkdir(parents=True, exist_ok=True)

	if dataset_name is None:
		print(
			f"WARNING: could not detect dataset (easy/medium/hard) for "
			f"input {input_dir}. Falling back to zero-vector domain "
			f"conditioning.",
			file=sys.stderr,
		)
		domain_id = None
	else:
		domain_id = DOMAIN_TO_ID[dataset_name]

	# PAN 2026 is English-only.
	language_id = LANGUAGE_TO_ID["en"]

	print(
		f"\n=== Processing {dataset_name or 'unknown'} "
		f"({input_dir} -> {output_dir}) ===",
		flush=True,
	)
	summary = run_prediction(
		input_dir=input_dir,
		output_dir=output_dir,
		model=model,
		tokenizer=tokenizer,
		language_id=language_id,
		domain_id=domain_id,
		show_progress=True,
		verbose=True,
		**INFERENCE_DEFAULTS,
	)
	return summary


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="PAN 2026 style change detection — TIRA entry point.",
	)
	parser.add_argument(
		"-i", "--input", required=True, type=Path,
		help="Input directory. Supports several layouts: "
		     "(a) problem-*.txt directly under INPUT (or one or two levels "
		     "below); (b) easy/medium/hard subdirectories, each containing "
		     "problem-*.txt files directly OR under a train/test/validation "
		     "subdirectory (the canonical TIRA layout).",
	)
	parser.add_argument(
		"-o", "--output", required=True, type=Path,
		help="Output directory for solution-problem-*.json files.",
	)
	parser.add_argument(
		"--dataset", choices=["easy", "medium", "hard"], default=None,
		help="Override path-based dataset detection (advanced).",
	)
	args = parser.parse_args(argv)

	input_dir = args.input.resolve()
	output_dir = args.output.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not input_dir.is_dir():
		print(f"ERROR: input directory does not exist: {input_dir}",
		      file=sys.stderr)
		return 1

	# Load model once, reuse across all dataset processings.
	print("Loading model + tokenizer...", flush=True)
	model, tokenizer = load_av_classifier()
	print("Model ready.", flush=True)

	# ── Case (b): subfolders for easy/medium/hard ──────────────────────
	subdirs = find_dataset_subdirs(input_dir)
	if subdirs:
		print(f"Detected {len(subdirs)} dataset subdirs:", flush=True)
		for ds_name in sorted(subdirs.keys()):
			rel = subdirs[ds_name].relative_to(input_dir)
			print(f"  {ds_name:<7s} -> {input_dir.name}/{rel}", flush=True)
		all_summaries = {}
		for ds_name in sorted(subdirs.keys()):
			ds_input  = subdirs[ds_name]
			ds_output = output_dir / ds_name
			summary = run_single_dataset(
				ds_input, ds_output, ds_name, model, tokenizer,
			)
			all_summaries[ds_name] = summary
		print("\n=== Done. Summary per dataset ===")
		for ds_name, summary in all_summaries.items():
			print(
				f"  {ds_name:<7s}: {summary['n_pairs']} pairs, "
				f"{summary['n_changes']} predicted changes "
				f"(rate={summary['change_rate']:.3f})"
			)
		return 0

	# ── Case (a): problem-*.txt somewhere under input_dir (up to 2 levels) ─
	problem_dir = _find_problem_dir(input_dir)
	if problem_dir is not None:
		if problem_dir != input_dir:
			rel = problem_dir.relative_to(input_dir)
			print(f"Found problem files at {input_dir.name}/{rel}", flush=True)
		# Try detection from the actual problem_dir path (catches 'easy' in
		# nested paths like <root>/easy/test). Fall back to input_dir.
		ds_name = (
			args.dataset
			or detect_dataset_from_path(problem_dir)
			or detect_dataset_from_path(input_dir)
		)
		summary = run_single_dataset(
			problem_dir, output_dir, ds_name, model, tokenizer,
		)
		print(
			f"\nDone. {summary['n_pairs']} pairs, "
			f"{summary['n_changes']} predicted changes "
			f"(rate={summary['change_rate']:.3f})"
		)
		return 0

	print(
		f"ERROR: no problem-*.txt files found anywhere under {input_dir} "
		f"(searched at depth 0, 1, and 2). Expected one of:\n"
		f"  - {input_dir}/problem-*.txt\n"
		f"  - {input_dir}/<train|test|validation>/problem-*.txt\n"
		f"  - {input_dir}/{{easy,medium,hard}}/<train|test|validation>/problem-*.txt",
		file=sys.stderr,
	)
	return 1


if __name__ == "__main__":
	sys.exit(main())
