"""Filtered retrieval of NDMA guidance for one district's forecast."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    HazardTaxonomyConfig,
    NormalizationMethod,
    PreprocessingConfig,
    VectorStoreConfig,
)
from ..preprocessing.normalization import (
    ChannelStats,
    denormalise_array,
    resolve_stats,
)
from ..types import AssembledWindow
from .geospatial import (
    DistrictGrid,
    DistrictImpact,
    district_class_fraction,
    district_mean,
)

logger = logging.getLogger(__name__)

#: Terrain labels. These are the vocabulary the corpus metadata must use; the
#: thresholds that select between them are configuration.
REGION_MOUNTAINOUS = "mountainous"
REGION_COASTAL = "coastal"
REGION_PLAINS = "plains"

#: Hazard labels, same contract.
HAZARD_URBAN_WATERLOGGING = "urban_waterlogging"
HAZARD_FLASH_FLOOD = "flash_flood"

_ELEVATION_CHANNEL = "dem_elevation"
_LANDUSE_CHANNEL = "lulc_mask"

#: The only ``corpus`` label that counts as authoritative NDMA guidance.
#:
#: Everything else -- a bootstrap corpus, an unlabelled collection built
#: before this field existed, a partial index -- is grounding of some kind but
#: not NDMA's, and an advisory built on it must not claim otherwise. The
#: indexer writes this value only when it indexed real documents.
OFFICIAL_CORPUS = "ndma_official"


class MissingEmbeddingModelError(FileNotFoundError):
    """Raised when the sentence-transformer weights cannot be obtained."""

    TEMPLATE = (
        "Embedding model {model!r} could not be loaded and is not in the local "
        "cache. Retrieval will not run without it. To fetch it ahead of time, "
        "run 'huggingface-cli download {model}'. Underlying error: {error}"
    )

    def __init__(self, model: str, error: Exception) -> None:
        self.model = model
        super().__init__(self.TEMPLATE.format(model=model, error=error))


class MissingCorpusError(FileNotFoundError):
    """Raised when the NDMA vector store is not present."""

    TEMPLATE = (
        "NDMA guideline corpus not found at {path}. The advisory layer will "
        "not generate mitigation guidance without it. To build the vector "
        "store from the NDMA source documents, run "
        "'python -m indra.advisory.index_corpus'."
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        super().__init__(self.TEMPLATE.format(path=self.path))


# ---------------------------------------------------------------------------
# Terrain and hazard classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistrictProfile:
    """Static character of one district, in physical units."""

    state: str
    district: str
    mean_elevation_m: float
    urban_fraction: float
    #: False when no urban land-cover classes were configured, so the fraction
    #: above is zero for want of a codebook rather than for want of cities.
    urban_fraction_available: bool


def build_profiles(
    window: AssembledWindow,
    districts: DistrictGrid,
    config: PreprocessingConfig,
    taxonomy: HazardTaxonomyConfig,
    stats: dict[str, ChannelStats],
) -> dict[tuple[str, str], DistrictProfile]:
    """District-level terrain summary from the static priors in the window."""
    elevation_index = window.channel_index(_ELEVATION_CHANNEL)
    landuse_index = window.channel_index(_LANDUSE_CHANNEL)

    elevation_norm = np.asarray(window.tensor[0, elevation_index], dtype=np.float32)
    elevation_stats = resolve_stats(_ELEVATION_CHANNEL, stats, config.normalization)
    elevation_m = denormalise_array(
        elevation_norm,
        elevation_stats,
        NormalizationMethod.MINMAX,
        config.normalization,
    )

    landuse = np.asarray(window.tensor[0, landuse_index])

    means = district_mean(elevation_m, districts)
    urban_classes = list(taxonomy.urban_lulc_classes)
    fractions = district_class_fraction(landuse, districts, urban_classes)

    if not urban_classes:
        logger.warning(
            "no urban land-cover classes configured "
            "(advisory.taxonomy.urban_lulc_classes is empty), so urban "
            "classification is unavailable and every district will take the "
            "default hazard class"
        )

    return {
        key: DistrictProfile(
            state=key[0],
            district=key[1],
            mean_elevation_m=float(means[index]),
            urban_fraction=float(fractions[index]),
            urban_fraction_available=bool(urban_classes),
        )
        for index, key in enumerate(districts.keys)
    }


def classify_region(profile: DistrictProfile, taxonomy: HazardTaxonomyConfig) -> str:
    """Terrain band from mean elevation."""
    if not np.isfinite(profile.mean_elevation_m):
        # A district with no resolvable elevation is not a plain; saying so
        # would be a determination. Plains is the default label, and the
        # context records that it was a default.
        return REGION_PLAINS
    if profile.mean_elevation_m > taxonomy.mountainous_elevation_m:
        return REGION_MOUNTAINOUS
    if profile.mean_elevation_m < taxonomy.coastal_elevation_m:
        return REGION_COASTAL
    return REGION_PLAINS


def classify_hazard(profile: DistrictProfile, taxonomy: HazardTaxonomyConfig) -> str:
    """Hazard class from urban land-cover share."""
    if not profile.urban_fraction_available:
        return HAZARD_FLASH_FLOOD
    if profile.urban_fraction > taxonomy.urban_fraction:
        return HAZARD_URBAN_WATERLOGGING
    return HAZARD_FLASH_FLOOD


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalContext:
    """What is being asked about, and how confident each label is."""

    state: str
    district: str
    hazard_class: str
    severity_tier: str
    region: str
    #: Labels that were defaulted rather than determined. Carried through to
    #: the advisory, because "classified as flash flood" and "not classifiable
    #: as urban, so flash flood" are different claims.
    defaulted: tuple[str, ...] = ()

    @classmethod
    def from_impact(
        cls,
        impact: DistrictImpact,
        profile: DistrictProfile,
        taxonomy: HazardTaxonomyConfig,
    ) -> RetrievalContext:
        defaulted: list[str] = []
        if impact.severity is None:
            raise ValueError(
                f"{impact.state}/{impact.district} has no severity tier. "
                "Retrieval is filtered on it, and an unfiltered query returns "
                "guidance for the wrong severity. Supply an intensity field to "
                "extract_impacts so the band can be resolved."
            )
        if not profile.urban_fraction_available:
            defaulted.append("hazard_class")
        if not np.isfinite(profile.mean_elevation_m):
            defaulted.append("region")

        return cls(
            state=impact.state,
            district=impact.district,
            hazard_class=classify_hazard(profile, taxonomy),
            severity_tier=impact.severity,
            region=classify_region(profile, taxonomy),
            defaulted=tuple(defaulted),
        )

    def as_filter(self, names: Sequence[str]) -> dict[str, Any]:
        """A ChromaDB ``where`` clause over the configured filter names."""
        available = {
            "hazard_class": self.hazard_class,
            "severity_tier": self.severity_tier,
            "region": self.region,
        }
        clauses: list[dict[str, Any]] = []
        for name in names:
            if name not in available:
                raise KeyError(
                    f"metadata_filters names {name!r}, which this context "
                    f"cannot supply; it provides {sorted(available)}"
                )
            clauses.append({name: {"$eq": available[name]}})

        if not clauses:
            raise ValueError(
                "no metadata filters configured; an unfiltered query returns "
                "generic guidance for every event, which is what the filters "
                "exist to prevent"
            )
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def query_text(self, impact: DistrictImpact) -> str:
        """The natural-language query embedded for retrieval."""
        return (
            f"{self.severity_tier} {self.hazard_class.replace('_', ' ')} "
            f"in a {self.region} district, "
            f"peak rainfall probability {impact.peak_probability:.0%}, "
            f"affecting {impact.affected_fraction:.0%} of the area. "
            "Preparedness and mitigation actions."
        )


@dataclass(frozen=True)
class RetrievedGuideline:
    """One NDMA chunk, with what is needed to cite it."""

    #: Stable identifier from the collection. ``require_citations: true`` means
    #: the advisory must point at specific chunks, so this has to survive
    #: retrieval intact.
    chunk_id: str
    text: str
    #: Cosine distance as Chroma reports it: smaller is closer.
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    def citation(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.metadata.get("section"),
            "distance": round(self.distance, 4),
        }


@dataclass(frozen=True)
class RetrievalResult:
    """What came back, and when nothing did, why."""

    context: RetrievalContext
    guidelines: tuple[RetrievedGuideline, ...]
    query_text: str
    #: Set when ``guidelines`` is empty. The advisory layer branches on this.
    empty_reason: str | None = None
    #: The collection's ``corpus`` label, carried through so the advisory can
    #: say what it was actually grounded in.
    corpus: str = "unlabelled"

    def __bool__(self) -> bool:
        return bool(self.guidelines)

    @property
    def is_official(self) -> bool:
        """True only for retrieved chunks from the authoritative NDMA corpus."""
        return bool(self.guidelines) and self.corpus == OFFICIAL_CORPUS

    def citations(self) -> list[dict[str, Any]]:
        return [guideline.citation() for guideline in self.guidelines]

    def summary(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "official": self.is_official,
            "district": f"{self.context.state}/{self.context.district}",
            "hazard_class": self.context.hazard_class,
            "severity_tier": self.context.severity_tier,
            "region": self.context.region,
            "defaulted_labels": list(self.context.defaulted),
            "retrieved": len(self.guidelines),
            "empty_reason": self.empty_reason,
            "citations": self.citations(),
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class NdmaRetriever:
    """Filtered semantic retrieval over the NDMA guideline collection."""

    def __init__(self, config: VectorStoreConfig, open_collection: bool = True) -> None:
        self.config = config
        self.embedding = config.local_embedding
        # Loaded at construction, not on first query. A model that cannot be
        # obtained is a startup condition the lifecycle can record and degrade
        # around, rather than a surprise the first user request discovers.
        #: Set when the collection is opened. "unlabelled" for an index built
        #: before provenance was recorded, which is not the same as official.
        self.corpus: str = "unlabelled"
        self._encoder = self._load_encoder()
        # The indexer builds the collection this would open, so it constructs
        # with open_collection=False and uses embed_documents alone. Sharing
        # this class rather than reimplementing the encoder is what guarantees
        # the index and the queries agree on model, prefix and normalisation --
        # three things that fail silently when they diverge.
        self._collection = self._open() if open_collection else None

    # ------------------------------------------------------------- lifecycle
    def _load_encoder(self) -> Any:
        """Load the sentence-transformer, on the configured device."""
        from sentence_transformers import SentenceTransformer

        try:
            encoder = SentenceTransformer(
                self.embedding.model_id, device=self.embedding.device
            )
        except Exception as exc:
            raise MissingEmbeddingModelError(self.embedding.model_id, exc) from exc

        dim = encoder.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(
                f"Model {self.embedding.model_id!r} returned no embedding dimension."
            )
        actual = int(dim)
        if actual != self.embedding.dimensions:
            raise ValueError(
                f"{self.embedding.model_id!r} produces {actual}-dimensional "
                f"vectors but {self.embedding.dimensions} is configured. The "
                "collection was built against the configured value, and "
                "querying it with a different width fails at best and returns "
                "arbitrary neighbours at worst."
            )
        logger.info(
            "embedding with %s (%d-dim) on %s",
            self.embedding.model_id,
            actual,
            self.embedding.device,
        )
        return encoder

    def _open(self) -> Any:
        import chromadb

        path = Path(self.config.persist_directory)
        if not path.is_dir():
            raise MissingCorpusError(path)

        client = chromadb.PersistentClient(path=str(path))
        try:
            collection = client.get_collection(self.config.collection)
        except Exception as exc:
            raise MissingCorpusError(path / self.config.collection) from exc

        self._verify_provenance(collection)

        metadata = dict(getattr(collection, "metadata", None) or {})
        self.corpus = str(metadata.get("corpus") or "unlabelled")
        if self.corpus != OFFICIAL_CORPUS:
            # Loud, and once per process. An advisory grounded in a bootstrap
            # is still an advisory a control room might act on, and the only
            # thing separating it from an authoritative one is this label.
            logger.warning(
                "collection %r is labelled corpus=%r, not %r. Advisories built "
                "from it will report grounded_in_ndma=false.",
                self.config.collection,
                self.corpus,
                OFFICIAL_CORPUS,
            )

        logger.info(
            "NDMA collection %r opened with %d chunks (corpus=%s)",
            self.config.collection,
            collection.count(),
            self.corpus,
        )
        return collection

    def _verify_provenance(self, collection: Any) -> None:
        """Refuse a collection built with a different embedding model."""
        metadata = dict(getattr(collection, "metadata", None) or {})
        recorded_model = metadata.get("embedding_model")
        recorded_dims = metadata.get("embedding_dimensions")

        if recorded_model is None and recorded_dims is None:
            logger.warning(
                "collection %r records no embedding provenance, so it cannot "
                "be verified against %s. An index built with a different model "
                "would return confident but unrelated guidance.",
                self.config.collection,
                self.embedding.model_id,
            )
            return

        if recorded_model is not None and recorded_model != self.embedding.model_id:
            raise ValueError(
                f"collection {self.config.collection!r} was built with "
                f"{recorded_model!r} but is configured to be queried with "
                f"{self.embedding.model_id!r}. Cross-space retrieval returns "
                "plausible nonsense; rebuild the index or correct the "
                "configuration. An index built before the move off the hosted "
                "embedding model is exactly this case."
            )
        if (
            recorded_dims is not None
            and int(recorded_dims) != self.embedding.dimensions
        ):
            raise ValueError(
                f"collection {self.config.collection!r} holds "
                f"{recorded_dims}-dimensional vectors but "
                f"{self.embedding.dimensions} is configured"
            )

    # ------------------------------------------------------------- embedding
    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        vectors = self._encoder.encode(
            [f"{prefix}{text}" for text in texts],
            batch_size=self.embedding.batch_size,
            normalize_embeddings=self.embedding.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in np.atleast_2d(vectors)]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query, with the query prefix."""
        return self._encode([text], self.embedding.query_prefix)[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks, with the document prefix."""
        return self._encode(texts, self.embedding.document_prefix)

    # ------------------------------------------------------------- retrieval
    def retrieve(
        self,
        context: RetrievalContext,
        impact: DistrictImpact,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Fetch guidance for one district's forecast situation."""
        query = context.query_text(impact)
        where = context.as_filter(self.config.metadata_filters)
        k = top_k or self.config.top_k

        try:
            embedding = self.embed_query(query)
        except Exception as exc:
            # The forecast is still valid and still served. Only the advisory
            # is unavailable, and it says so.
            logger.warning("query embedding failed: %s", exc)
            return RetrievalResult(
                context=context,
                guidelines=(),
                query_text=query,
                empty_reason=f"embedding unavailable: {type(exc).__name__}",
                corpus=self.corpus,
            )

        response = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]

        if not documents:
            reason = (
                f"no NDMA guidance indexed for hazard_class="
                f"{context.hazard_class!r}, severity_tier="
                f"{context.severity_tier!r}, region={context.region!r}"
            )
            logger.info("%s/%s: %s", context.state, context.district, reason)
            return RetrievalResult(
                context=context,
                guidelines=(),
                query_text=query,
                empty_reason=reason,
                corpus=self.corpus,
            )

        guidelines = tuple(
            RetrievedGuideline(
                chunk_id=str(chunk_id),
                text=str(document),
                distance=float(distance),
                metadata=dict(metadata or {}),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            )
        )
        logger.info(
            "%s/%s: %d chunk(s) retrieved from corpus=%s, nearest distance %.4f",
            context.state,
            context.district,
            len(guidelines),
            self.corpus,
            guidelines[0].distance,
        )
        return RetrievalResult(
            context=context,
            guidelines=guidelines,
            query_text=query,
            corpus=self.corpus,
        )


__all__ = [
    "OFFICIAL_CORPUS",
    "HAZARD_FLASH_FLOOD",
    "HAZARD_URBAN_WATERLOGGING",
    "REGION_COASTAL",
    "REGION_MOUNTAINOUS",
    "REGION_PLAINS",
    "DistrictProfile",
    "MissingCorpusError",
    "MissingEmbeddingModelError",
    "NdmaRetriever",
    "RetrievalContext",
    "RetrievalResult",
    "RetrievedGuideline",
    "build_profiles",
    "classify_hazard",
    "classify_region",
]
