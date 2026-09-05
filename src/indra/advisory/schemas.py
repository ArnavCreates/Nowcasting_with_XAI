"""OASIS Common Alerting Protocol 1.2 models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: CAP recommends a headline of 160 characters or fewer. Longer text belongs
#: in ``description``; a headline is what appears in an SMS or a ticker.
_HEADLINE_MAX = 160

#: Below this a text field cannot carry an actionable instruction or a
#: meteorological explanation. It exists to catch a generation that returned
#: a placeholder rather than to police prose.
_PROSE_MIN = 20


class CAPModel(BaseModel):
    """Base for every CAP model: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# CAP 1.2 enumerations
# ---------------------------------------------------------------------------


class CAPSeverity(str, Enum):
    EXTREME = "Extreme"
    SEVERE = "Severe"
    MODERATE = "Moderate"
    MINOR = "Minor"
    UNKNOWN = "Unknown"

    @property
    def rank(self) -> int:
        """Ordering for comparison. ``Unknown`` sits below every real level."""
        return {
            CAPSeverity.UNKNOWN: 0,
            CAPSeverity.MINOR: 1,
            CAPSeverity.MODERATE: 2,
            CAPSeverity.SEVERE: 3,
            CAPSeverity.EXTREME: 4,
        }[self]


class CAPUrgency(str, Enum):
    IMMEDIATE = "Immediate"
    EXPECTED = "Expected"
    FUTURE = "Future"
    PAST = "Past"
    UNKNOWN = "Unknown"


class CAPCertainty(str, Enum):
    OBSERVED = "Observed"
    LIKELY = "Likely"
    POSSIBLE = "Possible"
    UNLIKELY = "Unlikely"
    UNKNOWN = "Unknown"


class CAPStatus(str, Enum):
    ACTUAL = "Actual"
    EXERCISE = "Exercise"
    SYSTEM = "System"
    TEST = "Test"
    DRAFT = "Draft"


class CAPMsgType(str, Enum):
    ALERT = "Alert"
    UPDATE = "Update"
    CANCEL = "Cancel"
    ACK = "Ack"
    ERROR = "Error"


class CAPScope(str, Enum):
    PUBLIC = "Public"
    RESTRICTED = "Restricted"
    PRIVATE = "Private"


class CAPCategory(str, Enum):
    MET = "Met"
    GEO = "Geo"
    SAFETY = "Safety"
    SECURITY = "Security"
    RESCUE = "Rescue"
    FIRE = "Fire"
    HEALTH = "Health"
    ENV = "Env"
    TRANSPORT = "Transport"
    INFRA = "Infra"
    CBRNE = "CBRNE"
    OTHER = "Other"


# ---------------------------------------------------------------------------
# The generated block
# ---------------------------------------------------------------------------


class CAPAlertInfo(CAPModel):
    """The generated alert content."""

    headline: str = Field(
        min_length=1,
        max_length=_HEADLINE_MAX,
        description=(
            "One-line alert headline, at most 160 characters, naming the "
            "hazard and the affected district."
        ),
    )
    description: str = Field(
        min_length=_PROSE_MIN,
        description=(
            "The forecast situation: expected rainfall intensity, the "
            "affected area, and the time window."
        ),
    )
    instruction: str = Field(
        min_length=_PROSE_MIN,
        description=(
            "Actionable protective measures, composed only from the supplied "
            "NDMA guideline excerpts."
        ),
    )
    severity: CAPSeverity = Field(description="CAP severity of the forecast impact.")
    urgency: CAPUrgency = Field(
        description="CAP urgency, reflecting how soon action is required."
    )
    certainty: CAPCertainty = Field(
        description="CAP certainty, reflecting ensemble confidence."
    )

    @field_validator("headline", "description", "instruction")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only prose."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("field is blank; an advisory cannot omit it")
        return stripped


# ---------------------------------------------------------------------------
# The assembled alert
# ---------------------------------------------------------------------------


class CAPParameter(CAPModel):
    """A ``<parameter>`` pair."""

    value_name: str = Field(serialization_alias="valueName")
    value: str


class CAPArea(CAPModel):
    """The affected area. ``area_desc`` is required by CAP 1.2."""

    area_desc: str = Field(serialization_alias="areaDesc", min_length=1)
    #: ``(valueName, value)`` administrative codes, e.g. a census district code.
    geocode: tuple[CAPParameter, ...] = ()


class CAPInfo(CAPModel):
    """A complete ``<info>`` block: the generated content plus system fields."""

    language: str
    category: tuple[CAPCategory, ...]
    event: str = Field(min_length=1)
    urgency: CAPUrgency
    severity: CAPSeverity
    certainty: CAPCertainty
    headline: str = Field(max_length=_HEADLINE_MAX)
    description: str
    instruction: str
    sender_name: str = Field(serialization_alias="senderName")
    #: When the forecast was issued, when impact begins, and when the advisory
    #: lapses. ``expires`` is not decoration: a warning with no expiry stays on
    #: a dashboard after the storm has passed.
    effective: datetime
    onset: datetime
    expires: datetime
    area: tuple[CAPArea, ...]
    parameter: tuple[CAPParameter, ...] = ()

    @field_validator("expires")
    @classmethod
    def _expiry_is_future(cls, value: datetime, info: Any) -> datetime:
        onset = info.data.get("onset")
        if onset is not None and value <= onset:
            raise ValueError(
                f"expires {value.isoformat()} does not follow onset "
                f"{onset.isoformat()}; the advisory would lapse before the "
                "impact it warns about"
            )
        return value


class CAPAlert(CAPModel):
    """A complete CAP 1.2 alert."""

    identifier: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    sent: datetime
    status: CAPStatus
    msg_type: CAPMsgType = Field(serialization_alias="msgType")
    scope: CAPScope
    info: tuple[CAPInfo, ...] = Field(min_length=1)

    def to_cap_dict(self) -> dict[str, Any]:
        """CAP-named mapping, ready for XML or JSON serialisation."""
        return self.model_dump(by_alias=True, mode="json")


__all__ = [
    "CAPAlert",
    "CAPAlertInfo",
    "CAPArea",
    "CAPCategory",
    "CAPCertainty",
    "CAPInfo",
    "CAPMsgType",
    "CAPModel",
    "CAPParameter",
    "CAPScope",
    "CAPSeverity",
    "CAPStatus",
    "CAPUrgency",
]
