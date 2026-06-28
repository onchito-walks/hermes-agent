from __future__ import annotations

from datetime import datetime, timezone

from agent.lane_router import (
    classify_lane,
    mark_model_unhealthy,
    resolve_lane_route,
    router_enabled,
    should_failover,
)


def _config():
    return {
        "model_lane_router": {
            "enabled": True,
            "default_lane": "executor-fast",
            "policy": {
                "senior_reviewer_requires_explicit_approval": True,
                "never_auto_escalate_to_senior_on_rate_limit": True,
            },
            "models": {
                "glm4_7": {"provider": "custom", "model": "z-ai/glm4.7"},
                "deepseek_v4_flash": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
                "qwen_397b": {"provider": "openrouter", "model": "qwen/qwen3.5-397b-a17b"},
                "gpt_5_4_mini": {"provider": "openai-codex", "model": "gpt-5.4-mini"},
                "gpt_5_5": {"provider": "openai-codex", "model": "gpt-5.5"},
            },
            "lanes": {
                "executor-fast": {
                    "models": ["glm4_7", "deepseek_v4_flash"],
                    "reasoning_effort": "low",
                    "toolsets": ["terminal", "file", "todo"],
                },
                "instant-code-executor": {
                    "models": ["glm4_7", "deepseek_v4_flash"],
                    "reasoning_effort": "none",
                    "toolsets": ["terminal", "file"],
                },
                "coder-build": {
                    "models": ["deepseek_v4_flash", "glm4_7"],
                    "reasoning_effort": "low",
                    "toolsets": ["terminal", "file", "todo", "skills"],
                    "skills": ["systematic-debugging", "test-driven-development"],
                },
                "researcher": {
                    "models": ["qwen_397b", "deepseek_v4_flash"],
                    "reasoning_effort": "medium",
                    "toolsets": ["web", "browser", "file"],
                },
                "planner-reviewer": {
                    "models": ["gpt_5_4_mini", "qwen_397b"],
                    "reasoning_effort": "medium",
                    "toolsets": ["clarify", "session_search", "skills"],
                    "advisory_only": True,
                },
                "senior-reviewer": {
                    "models": ["gpt_5_5"],
                    "reasoning_effort": "high",
                    "toolsets": [],
                    "advisory_only": True,
                    "approval_required": True,
                },
                "large-context": {
                    "models": ["deepseek_v4_flash", "qwen_397b"],
                    "reasoning_effort": "low",
                    "toolsets": ["file", "terminal"],
                },
            },
        }
    }


def test_router_disabled_preserves_existing_behavior():
    assert not router_enabled({})
    route = resolve_lane_route("run tests", {})
    assert route.enabled is False
    assert route.model is None


def test_classifies_instant_execution_before_coding():
    cfg = _config()
    assert classify_lane("run pytest for the failing tests", cfg) == "instant-code-executor"
    route = resolve_lane_route("run pytest for the failing tests", cfg)
    assert route.lane == "instant-code-executor"
    assert route.model_ref == "glm4_7"
    assert route.reasoning_config == {"enabled": False}
    assert route.enabled_toolsets == ("terminal", "file")


def test_classifies_research_with_source_tools():
    route = resolve_lane_route("research the latest Hermes Agent docs online", _config())
    assert route.lane == "researcher"
    assert route.model_ref == "qwen_397b"
    assert route.reasoning_config == {"enabled": True, "effort": "medium"}
    assert "web" in route.enabled_toolsets
    assert "browser" in route.enabled_toolsets


def test_classifies_coder_and_returns_skill_suggestions():
    route = resolve_lane_route("debug this traceback and patch the code", _config())
    assert route.lane == "coder-build"
    assert route.model_ref == "deepseek_v4_flash"
    assert "systematic-debugging" in route.skills
    assert "test-driven-development" in route.skills


def test_senior_lane_requires_explicit_approval_and_does_not_execute_tools():
    route = resolve_lane_route("final gate review before source of truth resync", _config())
    assert route.lane == "senior-reviewer"
    assert route.model_ref == "gpt_5_5"
    assert route.approval_required is True
    assert route.advisory_only is True
    assert route.enabled_toolsets == ()
    assert "requires explicit approval" in route.explanation


def test_senior_lane_can_be_resolved_when_explicitly_approved():
    route = resolve_lane_route(
        "final gate review before source of truth resync",
        _config(),
        explicit_senior_approval=True,
    )
    assert route.lane == "senior-reviewer"
    assert route.model_ref == "gpt_5_5"
    assert route.approval_required is False
    assert route.reasoning_config == {"enabled": True, "effort": "high"}


def test_unhealthy_primary_skips_to_next_same_lane_model():
    health = {
        "glm4_7": {
            "unhealthy_until": "2099-01-01T00:00:00+00:00",
            "last_error": "rate_limit",
        }
    }
    route = resolve_lane_route(
        "run pytest",
        _config(),
        health_state=health,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    assert route.lane == "instant-code-executor"
    assert route.model_ref == "deepseek_v4_flash"
    assert route.skipped_models == ("glm4_7",)


def test_all_unhealthy_falls_back_to_first_model_for_caller_error_handling():
    health = {
        "glm4_7": {"unhealthy_until": "2099-01-01T00:00:00+00:00"},
        "deepseek_v4_flash": {"unhealthy_until": "2099-01-01T00:00:00+00:00"},
    }
    route = resolve_lane_route("run tests", _config(), health_state=health)
    assert route.model_ref == "glm4_7"
    assert route.skipped_models == ("glm4_7", "deepseek_v4_flash")


def test_failover_reason_policy_is_limit_like_only():
    assert should_failover("rate_limit") is True
    assert should_failover("timeout") is True
    assert should_failover("auth_permanent") is False
    assert should_failover("bad_request") is False


def test_mark_model_unhealthy_records_failure_metadata():
    state = {}
    updated = mark_model_unhealthy(
        state,
        "glm4_7",
        reason="rate_limit",
        until_iso="2026-05-01T21:00:00+00:00",
    )
    assert updated is state
    assert state["glm4_7"]["last_error"] == "rate_limit"
    assert state["glm4_7"]["failures"] == 1
    assert state["glm4_7"]["unhealthy_until"] == "2026-05-01T21:00:00+00:00"


def test_health_ledger_persists_and_drives_selection(tmp_path, monkeypatch):
    from agent import lane_router

    cfg = _config()
    cfg["model_lane_router"]["health"] = {
        "enabled": True,
        "state_file": str(tmp_path / "lane-health.json"),
    }
    route = lane_router.resolve_lane_route("run pytest", cfg)
    assert route.model_ref == "glm4_7"

    assert lane_router.note_route_error(cfg, route, "429 rate limit") is True
    persisted = lane_router.load_health_state(cfg)
    assert persisted["glm4_7"]["last_error"] == "rate_limit"

    rerouted = lane_router.resolve_lane_route("run pytest", cfg, health_state=persisted)
    assert rerouted.model_ref == "deepseek_v4_flash"
    assert rerouted.skipped_models == ("glm4_7",)


def test_health_ledger_ignores_non_limit_errors(tmp_path):
    from agent import lane_router

    cfg = _config()
    cfg["model_lane_router"]["health"] = {
        "enabled": True,
        "state_file": str(tmp_path / "lane-health.json"),
    }
    route = lane_router.resolve_lane_route("run pytest", cfg)
    assert lane_router.note_route_error(cfg, route, "400 bad request") is False
    assert lane_router.load_health_state(cfg) == {}
