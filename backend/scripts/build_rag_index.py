from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bootstrap_pythonpath() -> None:
    harness_root = _repo_root() / "backend" / "packages" / "harness"
    if str(harness_root) not in sys.path:
        sys.path.insert(0, str(harness_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local RAG index for Financial Agent.")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="Input file or directory to index. Can be repeated. Defaults to data/rag/raw/ if omitted.",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help="Optional source type override: auto/project_docs/finance_docs/eval_docs/user_upload.",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    _bootstrap_pythonpath()

    from deerflow.agents.rag import RagService

    inputs = args.inputs or [str(_repo_root() / "data" / "rag" / "raw")]
    service = RagService()
    metadata = service.rebuild_index(
        inputs=inputs,
        source_type=args.source_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("RAG index build complete")
    print(f"documents={metadata.get('document_count', 0)}")
    print(f"chunks={metadata.get('chunk_count', 0)}")
    print(f"dims={metadata.get('dims', 0)}")
    print(f"built_at={metadata.get('built_at', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
