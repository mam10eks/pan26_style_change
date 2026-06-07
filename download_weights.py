#!/usr/bin/env python3
"""Download the trained AVClassifier weights from a shared Google Drive link.

Usage
-----
    # Pass the link on the command line:
    python download_weights.py --url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"

    # Or via environment variable:
    GDRIVE_URL="<link>" python download_weights.py

    # Or pass a bare file ID:
    python download_weights.py --url FILE_ID

The default output location is ``./weights/av_classifier.pt``, matching the
default ``CHECKPOINT_PATH`` in ``load_model.py``. The repo owner shares the
Google Drive link separately (e.g., in the README, an email, or a private
channel) — it is not committed to the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
	import gdown
except ImportError:
	print(
		"ERROR: 'gdown' is not installed. Install it with:\n"
		"    pip install gdown",
		file=sys.stderr,
	)
	sys.exit(1)


DEFAULT_OUTPUT = Path("weights/av_classifier.pt")


def download_from_gdrive(url_or_id: str, output: Path) -> None:
	"""Download a Google Drive file via gdown.

	Accepts:
	  - Full sharing URL (``https://drive.google.com/file/d/FILE_ID/view?...``)
	  - Open-file URL (``https://drive.google.com/open?id=FILE_ID``)
	  - Bare file ID

	gdown 4.x and 5.x have slightly different APIs:
	  - 4.x needs ``fuzzy=True`` to accept various URL formats
	  - 5.x removed ``fuzzy`` (the behavior is now default)
	We try the 5.x signature first, fall back to 4.x.
	"""
	output.parent.mkdir(parents=True, exist_ok=True)

	is_url = "drive.google.com" in url_or_id or url_or_id.startswith("http")

	def _download(**extra):
		if is_url:
			return gdown.download(
				url=url_or_id, output=str(output), quiet=False, **extra,
			)
		return gdown.download(
			id=url_or_id, output=str(output), quiet=False, **extra,
		)

	try:
		# gdown 5.x: no fuzzy kwarg
		result = _download()
	except TypeError:
		# gdown 4.x: needs fuzzy=True to accept the sharing-URL format
		result = _download(fuzzy=True)

	if result is None or not output.exists():
		print(
			"\nERROR: Download failed. Possible causes:\n"
			"  - The link is not publicly shared (set sharing to 'Anyone "
			"    with the link').\n"
			"  - The file ID or URL is incorrect.\n"
			"  - Google Drive rate limit reached — try again later, or "
			"    download the file manually from the link in a browser and "
			"    place it at the --output path.\n",
			file=sys.stderr,
		)
		sys.exit(1)

	size_mb = output.stat().st_size / 1024 / 1024
	print(f"\nDownloaded {size_mb:.1f} MB to {output}")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Download trained AVClassifier weights from Google Drive.",
	)
	parser.add_argument(
		"--url", default=os.environ.get("GDRIVE_URL"),
		help="Google Drive sharing URL or file ID. Can also be set via the "
		     "GDRIVE_URL environment variable.",
	)
	parser.add_argument(
		"--output", default=DEFAULT_OUTPUT, type=Path,
		help=f"Output path for the downloaded checkpoint "
		     f"(default: {DEFAULT_OUTPUT}).",
	)
	parser.add_argument(
		"--force", action="store_true",
		help="Overwrite an existing checkpoint file at --output.",
	)
	args = parser.parse_args()

	if not args.url:
		parser.error(
			"--url is required (or set the GDRIVE_URL environment variable). "
			"The repository owner shares the URL separately."
		)

	if args.output.exists() and not args.force:
		size_mb = args.output.stat().st_size / 1024 / 1024
		print(
			f"Weights already exist at {args.output} ({size_mb:.1f} MB).\n"
			f"Use --force to re-download, or delete the file manually."
		)
		return 0

	download_from_gdrive(args.url, args.output)
	return 0


if __name__ == "__main__":
	sys.exit(main())