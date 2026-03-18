from __future__ import annotations

import argparse
from pathlib import Path

from processor import Processor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge two detected people into one identity")
    parser.add_argument("target", help="Target person folder to keep")
    parser.add_argument("source", help="Source person folder to merge and delete")
    parser.add_argument("--root", help="Root directory containing projects")
    parser.add_argument("--project", help="Project name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.root and args.project:
        root = Path(args.root)
        project = args.project
    else:
        current_dir = Path.cwd().resolve()
        root = current_dir.parent
        project = current_dir.name
    Processor().merge_people(root, project, args.target, args.source)
    print(f"Merged {args.source} into {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
