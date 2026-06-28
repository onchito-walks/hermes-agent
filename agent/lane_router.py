"""Deterministic lane router for model/tool/reasoning selection.

This module is intentionally side-effect free. It does not mutate config, create
clients, or rewrite tool/skill settings. Callers pass the loaded config and get a
small ``LaneRoute`` description that can be applied by CLI/Gateway orchestration.

Phase 1 scope:
- parse ``model_lane_router`` config;
- classify a user turn into a lane using cheap deterministic rules;
- select the first healthy model in that lane;
- return route-scoped reasoning/toolset/skill metadata;
- keep GPT-5.5/senior lanes explicit-only by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

from hermes_constants import get_hermes_home, parse_reasoning_effort


_LIMIT_LIKE_REASONS = {
    "rate_limit",
    "quota",
    "quota_exhausted",
    "overload",
    "timeout",
    "connection_error",
    "provider_error",
    "server_error",
}

_SENIOR_KEYWORDS = re.compile(
    r"\b("
    r"senior review|boss pass|boss-pass|final gate|doctrine|data loss|"
    r"source of truth|resync|re-sync|import/export|bulk movement|graph cleanup|"
    r"delete all|drop database|migration|production|professional liability"
    r")\b",
    re.IGNORECASE,
)

_INSTANT_EXEC_KEYWORDS = re.compile(
    r"\b("
    r"run tests?|pytest|syntax check|py_compile|typecheck|lint|build|compile|"
    r"execute|run command|terminal|checksum|sha256|hash|base64|calculate|"
    r"what time|current date|disk usage|ports?|processes?|git status"
    r")\b",
    re.IGNORECASE,
)

_CODER_KEYWORDS = re.compile(
    r"\b("
    r"implement|fix bug|debug|patch|refactor|code review|unit tests?|"
    r"integration tests?|stack trace|traceback|exception|failing test|PR|diff"
    r")\b",
    re.IGNORECASE,
)

_RESEARCH_KEYWORDS = re.compile(
    r"\b("
    r"research|latest docs?|online|web|browser|source-backed|current docs?|"
    r"documentation|release notes?|changelog|compare|benchmark|find sources?"
    r")\b",
    re.IGNORECASE,
)

_PLANNER_KEYWORDS = re.compile(
    r"\b("
    r"plan|design|architecture|review the plan|critique|decompose|strategy|"
    r"roadmap|proposal|spec|evaluate approach"
    r")\b",
    re.IGNORECASE,
)

_LARGE_CONTEXT_KEYWORDS = re.compile(
    r"\b("
    r"huge log|large context|long transcript|whole repo|entire codebase|"
    r"massive file|all files|full repository"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LaneRoute:
    """Resolved lane-routing decision for one user turn."""

    enabled: bool
    lane: str | None = None
    model_ref: str | None = None
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_mode: str | None = None
    reasoning_effort: str | None = None
    reasoning_config: dict[str, Any] | None = None
    enabled_toolsets: tuple[str, ...] = ()
    disabled_toolsets: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    advisory_only: bool = False
    approval_required: bool = False
    explanation: str = ""
    skipped_models: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signature_fragment(self) -> tuple[Any, ...]:
        """Stable fragment callers can include in their agent cache key."""
        return (
            self.lane,
            self.model_ref,
            self.model,
            self.provider,
            self.base_url,
            self.api_mode,
            self.reasoning_effort,
            self.enabled_toolsets,
            self.disabled_toolsets,
            self.advisory_only,
            self.approval_required,
        )


def router_enabled(config: Mapping[str, Any] | None) -> bool:
    section = _router_section(config)
    return bool(section.get("enabled"))


def classify_lane(prompt: str, config: Mapping[str, Any] | None = None) -> str:
    """Classify a prompt into a configured lane using deterministic rules."""
    section = _router_section(config)
    default_lane = str(section.get("default_lane") or "executor-fast")
    text = prompt or ""

    if _SENIOR_KEYWORDS.search(text):
        return _first_existing_lane(section, ("senior-reviewer", default_lane))
    if _LARGE_CONTEXT_KEYWORDS.search(text) or len(text) > 12000:
        return _first_existing_lane(section, ("large-context", default_lane))
    if _RESEARCH_KEYWORDS.search(text):
        return _first_existing_lane(section, ("researcher", default_lane))
    if _INSTANT_EXEC_KEYWORDS.search(text):
        return _first_existing_lane(section, ("instant-code-executor", default_lane))
    if _CODER_KEYWORDS.search(text):
        return _first_existing_lane(section, ("coder-build", default_lane))
    if _PLANNER_KEYWORDS.search(text):
        return _first_existing_lane(section, ("planner-reviewer", default_lane))
    return default_lane


def resolve_lane_route(
    prompt: str,
    config: Mapping[str, Any] | None,
    *,
    health_state: Mapping[str, Any] | None = None,
    explicit_senior_approval: bool = False,
    now: datetime | None = None,
) -> LaneRoute:
    """Resolve a prompt to a lane route.

    Returns ``LaneRoute(enabled=False)`` when the config section is missing or
    disabled so callers can preserve current Hermes behavior exactly.
    """
    section = _router_section(config)
    if not section.get("enabled"):
        return LaneRoute(enabled=False, explanation="model_lane_router disabled")

    lane_name = classify_lane(prompt, config)
    lanes = section.get("lanes") if isinstance(section.get("lanes"), Mapping) else {}
    lane_cfg = lanes.get(lane_name) if isinstance(lanes.get(lane_name), Mapping) else {}

    policy = section.get("policy") if isinstance(section.get("policy"), Mapping) else {}
    approval_required = bool(lane_cfg.get("approval_required"))
    advisory_only = bool(lane_cfg.get("advisory_only"))

    if approval_required and not explicit_senior_approval:
        selected_ref = _first_model_ref(lane_cfg)
        model_info = _model_info(section, selected_ref)
        return LaneRoute(
            enabled=True,
            lane=lane_name,
            model_ref=selected_ref,
            model=model_info.get("model"),
            provider=model_info.get("provider"),
            base_url=model_info.get("base_url") or model_info.get("endpoint"),
            api_mode=model_info.get("api_mode"),
            reasoning_effort=_reasoning_effort(lane_cfg),
            reasoning_config=parse_reasoning_effort(_reasoning_effort(lane_cfg) or ""),
            enabled_toolsets=tuple(_str_list(lane_cfg.get("toolsets"))),
            disabled_toolsets=tuple(_str_list(lane_cfg.get("disabled_toolsets"))),
            skills=tuple(_str_list(lane_cfg.get("skills"))),
            advisory_only=advisory_only,
            approval_required=True,
            explanation=(
                f"lane '{lane_name}' requires explicit approval; "
                "not auto-routing into senior/reviewer model"
            ),
            metadata={"policy": dict(policy)},
        )

    selected_ref, skipped = _select_healthy_model(lane_cfg, health_state or {}, now=now)
    model_info = _model_info(section, selected_ref)
    effort = _reasoning_effort(lane_cfg)

    return LaneRoute(
        enabled=True,
        lane=lane_name,
        model_ref=selected_ref,
        model=model_info.get("model") or selected_ref,
        provider=model_info.get("provider"),
        base_url=model_info.get("base_url") or model_info.get("endpoint"),
        api_mode=model_info.get("api_mode"),
        reasoning_effort=effort,
        reasoning_config=parse_reasoning_effort(effort or ""),
        enabled_toolsets=tuple(_str_list(lane_cfg.get("toolsets"))),
        disabled_toolsets=tuple(_str_list(lane_cfg.get("disabled_toolsets"))),
        skills=tuple(_str_list(lane_cfg.get("skills"))),
        advisory_only=advisory_only,
        approval_required=False,
        explanation=f"lane '{lane_name}' selected model '{selected_ref}'",
        skipped_models=tuple(skipped),
        metadata={"policy": dict(policy)},
    )


def mark_model_unhealthy(
    health_state: dict[str, Any],
    model_ref: str,
    *,
    reason: str,
    until_iso: str | None = None,
) -> dict[str, Any]:
    """Return an updated health-state dict with a model marked unhealthy.

    This helper is pure-ish: it mutates and returns the passed mapping so callers
    can decide when/how to persist it.
    """
    if not model_ref:
        return health_state
    entry = dict(health_state.get(model_ref) or {})
    entry["last_error"] = reason
    entry["last_error_at"] = datetime.now(timezone.utc).isoformat()
    entry["failures"] = int(entry.get("failures") or 0) + 1
    if until_iso:
        entry["unhealthy_until"] = until_iso
    health_state[model_ref] = entry
    return health_state


def should_failover(reason: str) -> bool:
    """Whether a classified failure reason should try same-lane fallback."""
    return (reason or "").strip().lower() in _LIMIT_LIKE_REASONS


def load_health_state(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Load the persisted lane-router health ledger.

    Missing, disabled, or malformed ledgers are treated as empty. The path is
    profile-aware and relative paths resolve under ``HERMES_HOME``.
    """
    section = _router_section(config)
    health_cfg = section.get("health") if isinstance(section.get("health"), Mapping) else {}
    if health_cfg.get("enabled") is False:
        return {}
    path = _health_state_path(section)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_health_state(config: Mapping[str, Any] | None, health_state: Mapping[str, Any]) -> None:
    """Persist the lane-router health ledger atomically enough for one process."""
    section = _router_section(config)
    health_cfg = section.get("health") if isinstance(section.get("health"), Mapping) else {}
    if health_cfg.get("enabled") is False:
        return
    path = _health_state_path(section)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(health_state), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def note_route_error(
    config: Mapping[str, Any] | None,
    route: LaneRoute | None,
    error: Any,
    *,
    ttl_seconds: int = 900,
) -> bool:
    """Record a limit-like route error in the persisted health ledger.

    Returns True when a model was marked unhealthy. This intentionally only
    handles limit/transient classes; auth/permanent/bad-request failures should
    not silently rotate models.
    """
    if not route or not route.enabled or not route.model_ref:
        return False
    reason = _classify_error_reason(error)
    if not should_failover(reason):
        return False
    state = load_health_state(config)
    until = datetime.now(timezone.utc) + timedelta(seconds=max(60, int(ttl_seconds or 900)))
    mark_model_unhealthy(
        state,
        route.model_ref,
        reason=reason,
        until_iso=until.isoformat(),
    )
    save_health_state(config, state)
    return True


def _health_state_path(section: Mapping[str, Any]) -> Path:
    health_cfg = section.get("health") if isinstance(section.get("health"), Mapping) else {}
    raw = str(health_cfg.get("state_file") or "state/lane-router-health.json").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = get_hermes_home() / path
    return path


def _classify_error_reason(error: Any) -> str:
    text = str(error or "").lower()
    if any(term in text for term in ("rate limit", "429", "too many requests")):
        return "rate_limit"
    if any(term in text for term in ("quota", "insufficient credits", "billing")):
        return "quota"
    if any(term in text for term in ("overloaded", "overload", "529")):
        return "overload"
    if any(term in text for term in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(term in text for term in ("connection", "temporarily unavailable", "server error", "502", "503", "504")):
        return "provider_error"
    return "unknown"


def _router_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    section = config.get("model_lane_router")
    return dict(section) if isinstance(section, Mapping) else {}


def _first_existing_lane(section: Mapping[str, Any], candidates: tuple[str, ...]) -> str:
    lanes = section.get("lanes") if isinstance(section.get("lanes"), Mapping) else {}
    for name in candidates:
        if name in lanes:
            return name
    return candidates[-1]


def _first_model_ref(lane_cfg: Mapping[str, Any]) -> str | None:
    refs = _str_list(lane_cfg.get("models") or lane_cfg.get("model_order"))
    return refs[0] if refs else None


def _select_healthy_model(
    lane_cfg: Mapping[str, Any],
    health_state: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[str | None, list[str]]:
    refs = _str_list(lane_cfg.get("models") or lane_cfg.get("model_order"))
    if not refs:
        return None, []
    now = now or datetime.now(timezone.utc)
    skipped: list[str] = []
    for ref in refs:
        if _is_model_healthy(ref, health_state, now=now):
            return ref, skipped
        skipped.append(ref)
    return refs[0], skipped


def _is_model_healthy(model_ref: str, health_state: Mapping[str, Any], *, now: datetime) -> bool:
    entry = health_state.get(model_ref)
    if not isinstance(entry, Mapping):
        return True
    until = entry.get("unhealthy_until") or entry.get("rate_limited_until") or entry.get("quota_exhausted_until")
    if not until:
        return True
    try:
        until_dt = datetime.fromisoformat(str(until).replace("Z", "+00:00"))
    except ValueError:
        return True
    if until_dt.tzinfo is None:
        until_dt = until_dt.replace(tzinfo=timezone.utc)
    return now >= until_dt


def _model_info(section: Mapping[str, Any], model_ref: str | None) -> dict[str, Any]:
    models = section.get("models") if isinstance(section.get("models"), Mapping) else {}
    info = models.get(model_ref or "")
    if isinstance(info, Mapping):
        return dict(info)
    return {"model": model_ref} if model_ref else {}


def _reasoning_effort(lane_cfg: Mapping[str, Any]) -> str | None:
    raw = lane_cfg.get("reasoning_effort", lane_cfg.get("reasoning"))
    if raw is None:
        return None
    return str(raw).strip().lower() or None


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []
