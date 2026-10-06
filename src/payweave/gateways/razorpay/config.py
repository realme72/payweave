"""Razorpay configuration.

Docs: docs/sources/razorpay/ (see SNAPSHOT.tsv). Secrets are ``SecretStr`` so
they never show up in reprs, logs or validation errors.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

API_BASE_URL = "https://api.razorpay.com/v1/"


class RazorpayConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key_id: str = Field(pattern=r"^rzp_(test|live)_[A-Za-z0-9]+$")
    key_secret: SecretStr = Field(min_length=1)
    # Current secret first. Keep old ones while rotating: Razorpay signs retries
    # of older events with the old secret (webhooks__validate-test.md).
    webhook_secrets: tuple[SecretStr, ...] = ()
    allow_live: bool = False  # live keys are refused unless explicitly enabled
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    # Order capture settings (payments__payments__capture-settings__api.md)
    automatic_capture_expiry_minutes: int = Field(default=12, ge=12)
    manual_capture_expiry_minutes: int = Field(default=7200, ge=12, le=7200)
    uncaptured_refund_speed: Literal["normal", "optimum"] = "normal"

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.is_live and not self.allow_live:
            raise ValueError("live key given but allow_live is false")
        if self.manual_capture_expiry_minutes < self.automatic_capture_expiry_minutes:
            raise ValueError("manual capture expiry must be >= automatic capture expiry")
        if any(not s.get_secret_value() for s in self.webhook_secrets):
            raise ValueError("webhook secrets must not be empty")
        return self

    @property
    def is_live(self) -> bool:
        return self.key_id.startswith("rzp_live_")
