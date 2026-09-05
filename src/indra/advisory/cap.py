"""Builds CAP 1.2 alerts from retrieved NDMA guidance."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import CAPConfig, LocalAdvisoryConfig
from .geospatial import DistrictImpact
from .retrieval import RetrievalResult
from .schemas import (
    CAPAlert,
    CAPAlertInfo,
    CAPArea,
    CAPCategory,
    CAPCertainty,
    CAPInfo,
    CAPMsgType,
    CAPParameter,
    CAPScope,
    CAPSeverity,
    CAPStatus,
    CAPUrgency,
)

logger = logging.getLogger(__name__)

#: Band name from ``thresholds.severity_bands`` to the CAP severity it floors.
#: The bands are this system's own vocabulary and CAP's is fixed, so the
#: mapping is stated once here rather than inferred at each use.
_SEVERITY_FLOOR: dict[str, CAPSeverity] = {
    "low": CAPSeverity.MINOR,
    "moderate": CAPSeverity.MODERATE,
    "high": CAPSeverity.SEVERE,
    "severe": CAPSeverity.EXTREME,
}

#: Ordering for the certainty ceiling. Lives here rather than on the enum
#: because CAP does not define certainty as ordered -- this is this system's
#: reading of it, used only to stop a generation claiming more confidence than
#: the ensemble supports.
_CERTAINTY_RANK: dict[CAPCertainty, int] = {
    CAPCertainty.UNKNOWN: 0,
    CAPCertainty.UNLIKELY: 1,
    CAPCertainty.POSSIBLE: 2,
    CAPCertainty.LIKELY: 3,
    CAPCertainty.OBSERVED: 4,
}

_SYSTEM_INSTRUCTION = """\
You are drafting the content of an OASIS CAP 1.2 weather advisory for an \
Indian district, on behalf of a nowcasting system. You write six fields and \
nothing else; identifiers, timestamps, areas and codes are supplied by the \
system.

RULES, in order of precedence:

1. GROUNDING. Compose `instruction` only from the NDMA guideline excerpts \
supplied below. Do not add protective measures that do not appear in them. \
Do not generalise an excerpt into a stronger or broader instruction than it \
states.

2. CONTEXT_EMPTY. If the excerpts section reads CONTEXT_EMPTY, no NDMA \
guidance was retrieved for this situation. Emit a generic heavy rainfall \
advisory of the kind routinely issued by the India Meteorological Department: \
stay indoors, avoid waterlogged underpasses, do not cross flooded roads, heed \
local authorities. You are then STRICTLY FORBIDDEN from stating specific \
mitigation protocols, named shelters, evacuation routes, road closures, or \
numeric action thresholds. Specific instructions that no authority issued are \
more dangerous than general advice everyone already knows.

3. NO FABRICATION. Do not invent rainfall figures, casualty estimates, river \
levels, or the names of agencies, officials or places. Every quantity you \
state must appear in the forecast summary below.

4. SEVERITY. `severity` must not be lower than the forecast band stated in \
the summary. You may raise it if the described impact warrants; you may not \
lower it.

5. TONE. Write for a district control room and the public: plain, calm, \
specific about time and place. No speculation about causes. No reassurance \
that the forecast does not support.
"""


class AdvisoryValidationError(ValueError):
    """A generation that violated a rule the prompt stated."""


# ---------------------------------------------------------------------------
# Deterministic fields
# ---------------------------------------------------------------------------


def urgency_for_lead(onset_hours: float, config: CAPConfig) -> CAPUrgency:
    """CAP urgency from how soon impact begins."""
    immediate = config.urgency_by_lead_hours.get("immediate")
    expected = config.urgency_by_lead_hours.get("expected")

    if immediate is not None and onset_hours <= immediate:
        return CAPUrgency.IMMEDIATE
    if expected is not None and onset_hours <= expected:
        return CAPUrgency.EXPECTED
    return CAPUrgency.FUTURE


def certainty_for_probability(probability: float) -> CAPCertainty:
    """CAP certainty from ensemble agreement."""
    if probability >= 0.7:
        return CAPCertainty.LIKELY
    if probability >= 0.3:
        return CAPCertainty.POSSIBLE
    return CAPCertainty.UNLIKELY


def severity_floor(band: str | None) -> CAPSeverity:
    """The lowest CAP severity this forecast may be described with."""
    if band is None:
        return CAPSeverity.UNKNOWN
    floor = _SEVERITY_FLOOR.get(band)
    if floor is None:
        raise KeyError(
            f"severity band {band!r} has no CAP mapping; known bands are "
            f"{sorted(_SEVERITY_FLOOR)}"
        )
    return floor


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvisoryRequest:
    """Everything the prompt is built from, in one auditable object."""

    impact: DistrictImpact
    retrieval: RetrievalResult
    valid_time: datetime
    lead_interval_minutes: int
    #: Leading attribution drivers, as ``advisory.xai`` reports them. Optional
    #: because an advisory must still be issuable when explanation failed.
    drivers: Sequence[dict[str, Any]] = ()

    @property
    def onset_hours(self) -> float:
        return (self.impact.onset_lead_index + 1) * self.lead_interval_minutes / 60.0

    @property
    def expiry_hours(self) -> float:
        return (self.impact.expiry_lead_index + 1) * self.lead_interval_minutes / 60.0

    def onset_time(self) -> datetime:
        return self.valid_time + timedelta(hours=self.onset_hours)

    def expiry_time(self) -> datetime:
        return self.valid_time + timedelta(hours=self.expiry_hours)


def build_prompt(request: AdvisoryRequest) -> str:
    """The user-turn prompt: the situation, then the excerpts."""
    impact = request.impact
    lines = [
        "FORECAST SUMMARY",
        f"District: {impact.district}, {impact.state}",
        f"Issued (UTC): {request.valid_time.isoformat()}",
        f"Impact begins: +{request.onset_hours:.1f} h "
        f"({request.onset_time().isoformat()})",
        f"Impact ends: +{request.expiry_hours:.1f} h "
        f"({request.expiry_time().isoformat()})",
        f"Peak probability of heavy rain: {impact.peak_probability:.0%}",
        f"Affected area: {impact.affected_area_km2:.0f} km2 "
        f"({impact.affected_fraction:.0%} of the district)",
        f"Forecast severity band: {impact.severity or 'unclassified'}",
    ]
    if impact.peak_intensity_mm_h is not None:
        lines.append(f"Peak rain rate: {impact.peak_intensity_mm_h:.0f} mm/h")

    context = request.retrieval.context
    lines += [
        f"Hazard class: {context.hazard_class}",
        f"Terrain: {context.region}",
    ]
    if context.defaulted:
        # The model should not describe a defaulted label as a determination.
        lines.append(
            f"NOTE: {', '.join(context.defaulted)} could not be determined and "
            "fell back to a default. Do not describe the district's terrain or "
            "hazard type as established."
        )

    if request.drivers:
        lines += ["", "LEADING FORECAST DRIVERS (for context, not for quoting)"]
        lines += [
            f"- {driver['channel']}: {driver['direction']} the probability"
            for driver in request.drivers
        ]

    lines += ["", "NDMA GUIDELINE EXCERPTS"]
    if not request.retrieval:
        lines.append("CONTEXT_EMPTY")
        lines.append(f"Reason: {request.retrieval.empty_reason}")
    else:
        for guideline in request.retrieval.guidelines:
            lines.append(f"[{guideline.chunk_id}] ({guideline.source})")
            lines.append(guideline.text.strip())
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class MissingAdvisoryModelError(FileNotFoundError):
    """Raised when the local advisory model is not on disk."""

    TEMPLATE = (
        "Advisory model not found at {path}. District advisories will not be "
        "generated without it. To fetch it, run "
        "'huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir "
        "{path}'."
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        super().__init__(self.TEMPLATE.format(path=self.path))


class AdvisoryGenerator:
    """The local model, loaded once, decoding under a schema constraint."""

    def __init__(self, config: LocalAdvisoryConfig) -> None:
        import outlines
        import torch

        self.config = config
        path = config.resolved_path()
        if not path.is_dir():
            # Local weights only. A bare Hugging Face id would otherwise
            # trigger a download on first use, which is not a thing a
            # forecasting service should do while answering a request.
            raise MissingAdvisoryModelError(path)

        device = config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "advisory model requested on %s; using %s",
                config.device,
                config.fallback_device,
            )
            device = config.fallback_device
        self.device = device

        self._model = outlines.models.transformers(str(path), device=device)
        self._generator = outlines.generate.json(
            self._model,
            CAPAlertInfo,
            sampler=outlines.samplers.multinomial(
                temperature=config.temperature or None
            ),
        )
        logger.info("advisory model loaded from %s on %s", path, device)

    def _render(self, request: AdvisoryRequest) -> str:
        """System instruction and prompt, through the model's chat template."""
        user = build_prompt(request)
        tokenizer = getattr(self._model, "tokenizer", None)
        apply = getattr(tokenizer, "apply_chat_template", None)
        if apply is None:
            return f"{_SYSTEM_INSTRUCTION}\n\n{user}"
        return apply(
            [
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, request: AdvisoryRequest) -> CAPAlertInfo | None:
        """One guided generation. ``None`` if the model itself failed."""
        try:
            return self._generator(
                self._render(request), max_tokens=self.config.max_tokens
            )
        except Exception as exc:
            logger.warning(
                "advisory generation failed: %s: %s", type(exc).__name__, exc
            )
            return None


def validate_generation(
    info: CAPAlertInfo, request: AdvisoryRequest, config: CAPConfig
) -> None:
    """Enforce the rules the prompt stated. Raises rather than corrects."""
    floor = severity_floor(request.impact.severity)
    if info.severity.rank < floor.rank:
        raise AdvisoryValidationError(
            f"generated severity {info.severity.value!r} is below the forecast "
            f"band's floor {floor.value!r} for {request.impact.district}. An "
            "advisory may be raised above the forecast but never lowered "
            "below it."
        )

    expected = urgency_for_lead(request.onset_hours, config)
    if info.urgency != expected:
        raise AdvisoryValidationError(
            f"generated urgency {info.urgency.value!r} contradicts an onset of "
            f"+{request.onset_hours:.1f} h, which is {expected.value!r}. "
            "Urgency follows the clock, not the prose."
        )

    if info.certainty is CAPCertainty.OBSERVED:
        raise AdvisoryValidationError(
            "certainty 'Observed' asserts that the event has been seen. This "
            "is a forecast, and the claim would be false."
        )

    # Severity has a floor; certainty has a ceiling. The asymmetry mirrors the
    # asymmetry of the harm: understating severity leaves a district
    # unprepared, while overstating certainty spends the credibility that
    # makes the next warning worth acting on.
    ceiling = certainty_for_probability(request.impact.peak_probability)
    if _CERTAINTY_RANK[info.certainty] > _CERTAINTY_RANK[ceiling]:
        raise AdvisoryValidationError(
            f"generated certainty {info.certainty.value!r} exceeds "
            f"{ceiling.value!r}, which is what a peak ensemble probability of "
            f"{request.impact.peak_probability:.0%} supports. Certainty may be "
            "lowered but not raised."
        )

    if config.require_citations and request.retrieval:
        cited = [
            guideline.chunk_id
            for guideline in request.retrieval.guidelines
            if guideline.chunk_id in info.instruction
        ]
        if not cited:
            logger.info(
                "advisory instruction cites no chunk id inline; citations are "
                "carried as CAP parameters instead"
            )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _parameters(request: AdvisoryRequest, grounded: bool) -> tuple[CAPParameter, ...]:
    """Machine-readable provenance, in CAP's own extension slot."""
    impact = request.impact
    parameters = [
        CAPParameter(
            value_name="peak_probability", value=f"{impact.peak_probability:.4f}"
        ),
        # Which collection the guidance came from, beside the boolean it
        # determines. "grounded_in_ndma: false" invites the question this
        # answers, and answering it here saves a reader inferring the corpus
        # from the shape of the chunk ids.
        CAPParameter(value_name="corpus", value=request.retrieval.corpus),
        CAPParameter(
            value_name="affected_area_km2", value=f"{impact.affected_area_km2:.1f}"
        ),
        CAPParameter(
            value_name="affected_fraction", value=f"{impact.affected_fraction:.4f}"
        ),
        CAPParameter(
            value_name="hazard_class", value=request.retrieval.context.hazard_class
        ),
        CAPParameter(value_name="region_class", value=request.retrieval.context.region),
        # The single most important flag on the whole alert: whether the
        # instruction rests on retrieved NDMA text or on generic advice.
        CAPParameter(value_name="grounded_in_ndma", value=str(grounded).lower()),
    ]
    if impact.peak_intensity_mm_h is not None:
        parameters.append(
            CAPParameter(
                value_name="peak_intensity_mm_h",
                value=f"{impact.peak_intensity_mm_h:.2f}",
            )
        )
    if request.retrieval.context.defaulted:
        parameters.append(
            CAPParameter(
                value_name="defaulted_labels",
                value=",".join(request.retrieval.context.defaulted),
            )
        )
    for guideline in request.retrieval.guidelines:
        parameters.append(
            CAPParameter(value_name="ndma_citation", value=guideline.chunk_id)
        )
    for driver in request.drivers:
        parameters.append(
            CAPParameter(
                value_name="forecast_driver",
                value=f"{driver['channel']}:{driver['direction']}",
            )
        )
    return tuple(parameters)


def build_alert(
    info: CAPAlertInfo,
    request: AdvisoryRequest,
    config: CAPConfig,
    event: str | None = None,
) -> CAPAlert:
    """Wrap validated content in the administrative CAP envelope."""
    # Not merely "retrieval returned something". Chunks from a bootstrap
    # corpus are grounding of a sort, and they are not NDMA's -- an advisory
    # claiming otherwise would attribute generic safety advice to a national
    # authority, which is the one provenance claim this system must never make.
    grounded = request.retrieval.is_official
    hazard = request.retrieval.context.hazard_class.replace("_", " ").title()

    block = CAPInfo(
        language=config.language,
        category=tuple(CAPCategory(name) for name in config.categories),
        event=event or f"Heavy Rainfall - {hazard}",
        urgency=info.urgency,
        severity=info.severity,
        certainty=info.certainty,
        headline=info.headline,
        description=info.description,
        instruction=info.instruction,
        sender_name=config.sender,
        effective=request.valid_time,
        onset=request.onset_time(),
        expires=request.expiry_time(),
        area=(CAPArea(area_desc=f"{request.impact.district}, {request.impact.state}"),),
        parameter=_parameters(request, grounded),
    )

    return CAPAlert(
        # Unique per sender, generated here. A model-supplied identifier that
        # collided would make an update indistinguishable from a new alert.
        identifier=f"{config.sender}-{uuid.uuid4()}",
        sender=config.sender,
        sent=datetime.now(tz=UTC),
        status=CAPStatus(config.status),
        msg_type=CAPMsgType(config.msg_type),
        scope=CAPScope(config.scope),
        info=(block,),
    )


def compose_advisory(
    request: AdvisoryRequest,
    generator: AdvisoryGenerator,
    config: CAPConfig,
) -> CAPAlert | None:
    """Generate, validate and assemble one district's advisory."""
    if not request.retrieval:
        logger.warning(
            "%s/%s: composing an ungrounded advisory (%s)",
            request.impact.state,
            request.impact.district,
            request.retrieval.empty_reason,
        )

    info = generator.generate(request)
    if info is None:
        return None

    try:
        validate_generation(info, request, config)
    except AdvisoryValidationError as exc:
        logger.error("advisory rejected for %s: %s", request.impact.district, exc)
        return None

    alert = build_alert(info, request, config)
    logger.info(
        "advisory issued for %s/%s: %s (%s, %s, corpus=%s, grounded=%s)",
        request.impact.state,
        request.impact.district,
        info.severity.value,
        info.urgency.value,
        info.certainty.value,
        request.retrieval.corpus,
        request.retrieval.is_official,
    )
    return alert


__all__ = [
    "AdvisoryGenerator",
    "AdvisoryRequest",
    "AdvisoryValidationError",
    "MissingAdvisoryModelError",
    "build_alert",
    "build_prompt",
    "certainty_for_probability",
    "compose_advisory",
    "severity_floor",
    "urgency_for_lead",
    "validate_generation",
]
