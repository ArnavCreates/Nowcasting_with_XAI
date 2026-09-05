"""Build the NDMA guideline collection."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from ..config import REPO_ROOT, VectorStoreConfig, load_inference_config
from .retrieval import (
    HAZARD_FLASH_FLOOD,
    HAZARD_URBAN_WATERLOGGING,
    OFFICIAL_CORPUS,
    REGION_COASTAL,
    REGION_MOUNTAINOUS,
    REGION_PLAINS,
    NdmaRetriever,
)

logger = logging.getLogger(__name__)

#: Default source directory. Under ``data/``, which .gitignore excludes: a
#: corpus is an input to this repository, not part of it.
DEFAULT_SOURCE = REPO_ROOT / "data" / "ndma"

#: Metadata every chunk must declare. These are the fields
#: ``vector_store.metadata_filters`` queries on, and a chunk missing one can
#: never be retrieved -- it would sit in the collection contributing nothing.
REQUIRED_METADATA = ("hazard_class", "severity_tier", "region")

#: Chunk budget in characters. e5-small-v2 truncates at 512 tokens, so a
#: longer chunk loses its tail silently -- the embedding is computed from the
#: first part and the rest is never represented. Roughly four characters per
#: token leaves headroom for the "passage: " prefix.
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDocument:
    """One parsed source file: its metadata and its body."""

    path: Path
    metadata: dict[str, Any]
    body: str

    @property
    def stem(self) -> str:
        return self.path.stem


def parse_document(path: Path) -> SourceDocument | None:
    """Read one file, or return ``None`` with a reason logged."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("skipping %s: %s", path.name, exc)
        return None

    match = _FRONT_MATTER.match(text)
    if match is None:
        logger.warning(
            "skipping %s: no YAML front matter. Every chunk must declare %s, "
            "and a document without them cannot be retrieved for any "
            "situation.",
            path.name,
            ", ".join(REQUIRED_METADATA),
        )
        return None

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning(
            "skipping %s: front matter is not valid YAML: %s", path.name, exc
        )
        return None

    if not isinstance(metadata, dict):
        logger.warning("skipping %s: front matter is not a mapping", path.name)
        return None

    missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
    if missing:
        logger.warning(
            "skipping %s: front matter omits %s. Values are not inferred from "
            "the text -- a mislabelled chunk retrieves confidently for "
            "situations it does not describe.",
            path.name,
            ", ".join(missing),
        )
        return None

    body = text[match.end() :].strip()
    if not body:
        logger.warning("skipping %s: no body after the front matter", path.name)
        return None

    return SourceDocument(path=path, metadata=metadata, body=body)


def load_documents(source: Path) -> list[SourceDocument]:
    """Every readable, well-formed document under ``source``."""
    if not source.is_dir():
        raise FileNotFoundError(
            f"source directory {source} does not exist. Place NDMA documents "
            "there as Markdown or plain text with YAML front matter, or run "
            "with --create-sample to write a bootstrap corpus."
        )

    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in (".md", ".txt"))
    documents = [doc for doc in (parse_document(p) for p in paths) if doc is not None]

    logger.info("read %d of %d document(s) from %s", len(documents), len(paths), source)
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    body: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split a document on paragraph boundaries, with overlap."""
    if chunk_chars <= 0:
        raise ValueError(f"chunk_chars must be positive; got {chunk_chars}")
    if overlap_chars >= chunk_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be smaller than chunk_chars "
            f"({chunk_chars}), or every chunk would contain the previous one"
        )

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_slide(paragraph, chunk_chars, overlap_chars))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current)
    return chunks


def _slide(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-stride window over one oversized paragraph, on word boundaries."""
    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back up to a space so a window never ends mid-word, which would
            # put a fragment into the embedding.
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + step)
    return [w for w in windows if w]


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit, with the identity a citation points at."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def build_chunks(
    documents: Iterable[SourceDocument],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk every document, assigning stable ids."""
    chunks: list[Chunk] = []
    for document in documents:
        pieces = chunk_text(document.body, chunk_chars, overlap_chars)
        for index, piece in enumerate(pieces):
            metadata = {
                # Only scalars: Chroma rejects nested values, and the filters
                # compare with $eq.
                key: value
                for key, value in document.metadata.items()
                if isinstance(value, str | int | float | bool)
            }
            metadata.setdefault("source", document.stem)
            metadata["document"] = document.stem
            metadata["chunk_index"] = index
            chunks.append(
                Chunk(
                    chunk_id=f"{document.stem}_chunk_{index}",
                    text=piece,
                    metadata=metadata,
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def write_collection(
    chunks: Sequence[Chunk],
    store: VectorStoreConfig,
    corpus_label: str,
    force: bool = False,
) -> int:
    """Embed and write the collection, recording its provenance."""
    import chromadb

    if not chunks:
        raise ValueError("no chunks to index")

    retriever = NdmaRetriever(store, open_collection=False)
    logger.info("embedding %d chunk(s)", len(chunks))
    vectors = retriever.embed_documents([chunk.text for chunk in chunks])

    path = Path(store.persist_directory)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))
    existing = [c.name for c in client.list_collections()]
    if store.collection in existing:
        if not force:
            raise FileExistsError(
                f"collection {store.collection!r} already exists at {path}. "
                "Re-run with --force to replace it; a partial overwrite would "
                "leave chunks from two corpora sharing one embedding space."
            )
        logger.warning("replacing existing collection %r", store.collection)
        client.delete_collection(store.collection)

    collection = client.create_collection(
        name=store.collection,
        metadata={
            "embedding_model": store.local_embedding.model_id,
            "embedding_dimensions": store.local_embedding.dimensions,
            "document_prefix": store.local_embedding.document_prefix,
            "corpus": corpus_label,
            "chunks": len(chunks),
            "created_at": datetime.now(tz=UTC).isoformat(),
        },
    )
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=cast(Any, vectors),
    )

    logger.info(
        "wrote %d chunk(s) to %r at %s (corpus=%s)",
        len(chunks),
        store.collection,
        path,
        corpus_label,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Bootstrap corpus
# ---------------------------------------------------------------------------

_SAMPLE_BANNER = (
    "> Bootstrap document. General public-safety advice, not an NDMA\n"
    "> publication, indexed so the retrieval and advisory path can be\n"
    "> exercised before the authoritative corpus is licensed. Advisories\n"
    "> grounded in it report grounded_in_ndma: false.\n"
)

#: Generic public-safety statements of the kind many agencies publish.
#: Unattributed and deliberately non-specific: no shelter names, no evacuation
#: routes, no numeric action thresholds. Those are exactly what the
#: CONTEXT_EMPTY rule forbids a model to invent, so a bootstrap corpus must not
#: supply them either.
_SAMPLE_BODY: dict[str, list[str]] = {
    HAZARD_FLASH_FLOOD: [
        "Move away from streams, culverts and low-lying ground, and toward "
        "higher ground, as soon as heavy rain is forecast rather than once "
        "water is rising.",
        "Do not enter or drive through flowing water. Depth and current are "
        "difficult to judge from the bank, and a road surface beneath moving "
        "water may already have been scoured away.",
        "Keep drinking water, a torch, a charged phone and essential "
        "medicines together and portable, so that leaving does not require "
        "gathering them first.",
    ],
    HAZARD_URBAN_WATERLOGGING: [
        "Avoid underpasses, basements and subways during heavy rain. These "
        "fill faster than the surrounding streets and offer no exit once "
        "water reaches the entrance.",
        "Do not approach fallen cables or standing water near street "
        "furniture and electrical fittings. Water that has reached a live "
        "conductor is hazardous well beyond the point of contact.",
        "Expect stalled traffic and suspended services, and postpone "
        "non-essential travel rather than attempting an alternative route "
        "through affected areas.",
    ],
}

_SAMPLE_REGION_NOTE: dict[str, str] = {
    REGION_MOUNTAINOUS: (
        "In hilly terrain, saturated slopes may fail without warning. Stay "
        "clear of steep cut slopes, recent landslip scars and the ground "
        "directly below them."
    ),
    REGION_COASTAL: (
        "Near the coast, drainage backs up when heavy rain coincides with a "
        "high tide, and water may recede more slowly than rainfall alone "
        "suggests."
    ),
    REGION_PLAINS: (
        "On flat terrain, water spreads widely and shallowly and may cover "
        "familiar routes without appearing deep. Depth is easily "
        "underestimated."
    ),
}

_SAMPLE_SEVERITY_NOTE: dict[str, str] = {
    "low": "Monitor official updates and defer avoidable travel.",
    "moderate": ("Prepare to move at short notice and keep essential items together."),
    "high": (
        "Move to higher ground or a safer building now, before conditions "
        "make movement difficult."
    ),
    "severe": (
        "Act immediately on instructions from local authorities and do not "
        "wait for conditions to worsen further."
    ),
}


def create_sample_corpus(destination: Path) -> int:
    """Write a bootstrap corpus covering the taxonomy's label space."""
    destination.mkdir(parents=True, exist_ok=True)
    written = 0

    for hazard, statements in _SAMPLE_BODY.items():
        for region, region_note in _SAMPLE_REGION_NOTE.items():
            for severity, severity_note in _SAMPLE_SEVERITY_NOTE.items():
                name = f"SAMPLE-{hazard}-{region}-{severity}.md"
                front = {
                    "hazard_class": hazard,
                    "severity_tier": severity,
                    "region": region,
                    "source": "sample-bootstrap",
                    "authority": "general public safety; not an NDMA publication",
                    "title": (
                        f"Bootstrap guidance: {hazard.replace('_', ' ')}, "
                        f"{region}, {severity} severity"
                    ),
                }
                body = "\n\n".join(
                    [
                        _SAMPLE_BANNER,
                        f"## {front['title']}",
                        severity_note,
                        region_note,
                        *statements,
                    ]
                )
                text = (
                    "---\n"
                    + yaml.safe_dump(front, sort_keys=True)
                    + "---\n\n"
                    + body
                    + "\n"
                )
                (destination / name).write_text(text, encoding="utf-8")
                written += 1

    logger.warning(
        "wrote %d bootstrap document(s) to %s. General public-safety advice, "
        "not NDMA publications: index the authoritative corpus over it once "
        "licensed.",
        written,
        destination,
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m indra.advisory.index_corpus",
        description=(
            "Chunk, embed and index NDMA guidance into the Chroma collection "
            "the advisory layer retrieves from."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"directory of .md/.txt documents (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help=(
            "write a bootstrap corpus to --source first. General public-safety "
            "for running the pipeline end to end without sourcing documents."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing collection instead of refusing",
    )
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to nowcast.yaml (default: configs/inference/nowcast.yaml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)

    inference = load_inference_config(args.config)
    store = inference.advisory.vector_store

    # The exact string the retriever tests against. Anything else -- including
    # a typo -- makes every advisory built on this collection report
    # grounded_in_ndma=false, which is the safe direction to fail in.
    corpus_label = OFFICIAL_CORPUS
    if args.create_sample:
        create_sample_corpus(args.source)
        corpus_label = "sample-bootstrap"

    try:
        documents = load_documents(args.source)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if not documents:
        logger.error(
            "no usable documents in %s. Every document needs YAML front matter "
            "declaring %s.",
            args.source,
            ", ".join(REQUIRED_METADATA),
        )
        return 2

    # If any document is not from the authoritative set, the whole collection is
    # labelled as one: a corpus is only as authoritative as its least
    # authoritative chunk.
    if any(
        str(doc.metadata.get("source", "")).startswith("sample") for doc in documents
    ):
        corpus_label = "sample-bootstrap"

    chunks = build_chunks(documents, args.chunk_chars, args.overlap_chars)
    logger.info("%d document(s) produced %d chunk(s)", len(documents), len(chunks))

    try:
        written = write_collection(chunks, store, corpus_label, force=args.force)
    except (FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("index complete: %d chunk(s), corpus=%s", written, corpus_label)
    if corpus_label != OFFICIAL_CORPUS:
        logger.warning(
            "collection labelled %s: advisories grounded in it report "
            "grounded_in_ndma=false and cite SAMPLE- chunk ids.",
            corpus_label,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "Chunk",
    "SourceDocument",
    "build_chunks",
    "chunk_text",
    "create_sample_corpus",
    "load_documents",
    "main",
    "parse_document",
    "write_collection",
]
