from __future__ import annotations

import argparse
from pathlib import Path

from processor import Processor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face detection attendance processor")
    parser.add_argument("--root", required=True, help="Root directory containing projects")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--date", required=True, help="Photo date in YYYY-MM-DD format")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    processor = Processor()
    summary = processor.process(Path(args.root), args.project, args.date)
    print(
        f"Processed {summary.images_processed} image(s), kept {summary.detections_kept} face(s), "
        f"matched {summary.matches}, created {summary.new_people} new people."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
