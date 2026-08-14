"""Tests for cold prompt-cache policy and pricing helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from deepagents_code.cold_cache import (
    CacheWriteBucket,
    PromptCachePolicy,
    endpoint_cache_identity,
    estimate_rewarm_cost,
    format_cache_age,
    format_cache_window,
    load_trusted_cache_endpoints,
    parse_cache_timestamp,
    resolve_prompt_cache_policy,
)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "default"),
        (
            "https://Proxy.EXAMPLE.com:443/v1/?ignored=yes#fragment",
            "https://proxy.example.com/v1",
        ),
        ("http://proxy.example.com:8080/v1/", "http://proxy.example.com:8080/v1"),
        ("not a URL", "invalid:not a URL"),
    ],
)
def test_endpoint_cache_identity_normalizes_endpoint_spelling(
    base_url: str | None, expected: str
) -> None:
    assert endpoint_cache_identity(base_url) == expected


def test_resolves_anthropic_default_and_one_hour_policies() -> None:
    default = resolve_prompt_cache_policy("anthropic:claude-sonnet-4-6")
    extended = resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6",
        {"cache_control": {"type": "ephemeral", "ttl": "1h"}},
    )

    assert default == PromptCachePolicy("Anthropic", 300, "expired", 1024, "5m")
    assert extended == PromptCachePolicy("Anthropic", 3600, "expired", 1024, "1h")


@pytest.mark.parametrize(
    ("model", "minimum"),
    [
        ("claude-opus-5", 512),
        ("claude-fable-5", 512),
        ("claude-mythos-5", 512),
        ("claude-opus-4-8", 1024),
        ("claude-sonnet-5", 1024),
        ("claude-sonnet-4-6", 1024),
        ("claude-opus-4-7", 2048),
        ("claude-mythos-preview", 2048),
        ("claude-opus-4-6", 4096),
        ("claude-opus-4-5", 4096),
        ("claude-haiku-4-5", 4096),
    ],
)
def test_resolves_anthropic_per_model_minimums(model: str, minimum: int) -> None:
    policy = resolve_prompt_cache_policy(f"anthropic:{model}")

    assert policy is not None
    assert policy.minimum_tokens == minimum


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-pro", "gpt-6"])
def test_resolves_current_openai_minimum_retention(model: str) -> None:
    policy = resolve_prompt_cache_policy(f"openai:{model}")

    # 30 minutes is the documented guaranteed minimum, but OpenAI may retain
    # the prefix longer, so past the window it may still be warm.
    assert policy == PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")


def test_resolves_explicit_older_openai_retention() -> None:
    in_memory = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "in_memory"},
    )
    extended = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "24h"},
    )

    # Both windows are documented maximums ("up to"), so past the window the
    # cache may still be warm.
    assert in_memory == PromptCachePolicy(
        "OpenAI", 3600, "may_be_cold", 1024, "generic"
    )
    assert extended == PromptCachePolicy(
        "OpenAI", 86400, "may_be_cold", 1024, "generic"
    )


def test_estimate_rewarm_cost_respects_per_model_anthropic_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4,096-token floor rejects 3,000 tokens on Haiku but not on Opus 5."""
    monkeypatch.setattr(
        "deepagents_code.cost_tracking.estimate_cost", lambda *_args: 1.0
    )
    haiku = resolve_prompt_cache_policy("anthropic:claude-haiku-4-5")
    opus = resolve_prompt_cache_policy("anthropic:claude-opus-5")

    assert haiku is not None
    assert opus is not None
    assert estimate_rewarm_cost(3000, "anthropic:claude-haiku-4-5", haiku) is None
    assert estimate_rewarm_cost(3000, "anthropic:claude-opus-5", opus) is not None


def test_skips_unresolved_or_custom_provider_policies() -> None:
    assert resolve_prompt_cache_policy("openai:gpt-5.5") is None
    assert resolve_prompt_cache_policy("google_genai:gemini-3.6-flash") is None
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6",
            base_url="https://gateway.example.com/v1",
        )
        is None
    )
    assert (
        resolve_prompt_cache_policy(
            "anthropic:claude-sonnet-4-6",
            base_url="https://gateway.example.com",
        )
        is None
    )


def test_trusted_endpoints_enable_policies_on_alternate_hosts() -> None:
    gateway = "https://gateway.example.com/v1"
    trusted = {"gateway.example.com"}

    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6", base_url=gateway, trusted_endpoints=trusted
    ) == PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    assert resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6", base_url=gateway, trusted_endpoints=trusted
    ) == PromptCachePolicy("Anthropic", 300, "expired", 1024, "5m")
    # A different, untrusted host on the same spec still resolves nothing.
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6",
            base_url="https://other.example.com",
            trusted_endpoints=trusted,
        )
        is None
    )


_ANTHROPIC_5M = PromptCachePolicy("Anthropic", 300, "expired", 1024, "5m")
"""Policy a same-format Anthropic route resolves, used as the positive control.

The cross-format tests below pair each suppressed spec with a spec that *does*
resolve on the same host, so a guard that stopped firing would flip a visible
assertion rather than leave every case at `None`.
"""


@pytest.mark.parametrize(
    "base_url",
    ["https://smith.langchain.com", "https://acme.smith.langchain.com"],
)
def test_langsmith_gateway_same_format_routes_keep_policies(base_url: str) -> None:
    trusted = {"smith.langchain.com", "acme.smith.langchain.com"}

    # Bare model names are served by the wire format's own provider.
    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6", base_url=base_url, trusted_endpoints=trusted
    ) == PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    # An explicit matching prefix is the same route; the prefix must be
    # stripped before model-family detection rather than defeating it.
    assert resolve_prompt_cache_policy(
        "openai:openai/gpt-5.6", base_url=base_url, trusted_endpoints=trusted
    ) == PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    assert resolve_prompt_cache_policy(
        "anthropic:anthropic/claude-opus-4-5",
        base_url=base_url,
        trusted_endpoints=trusted,
    ) == PromptCachePolicy("Anthropic", 300, "expired", 4096, "5m")


@pytest.mark.parametrize(
    "model_spec",
    [
        # Anthropic wire format routed to an OpenAI model.
        "anthropic:openai/gpt-5.6",
        # Prefixes for providers this module cannot price are crossings too:
        # the gateway still translates, so no policy may be assumed.
        "anthropic:google_genai/gemini-3",
        "anthropic:baseten/some-model",
        "anthropic:myorg/claude-opus-4-5",
    ],
)
def test_langsmith_gateway_cross_format_routes_resolve_no_policy(
    model_spec: str,
) -> None:
    gateway = "https://smith.langchain.com"
    trusted = {"smith.langchain.com"}

    # The same host and trust set resolve a policy for a same-format route,
    # so `None` here can only come from the cross-format guard.
    assert (
        resolve_prompt_cache_policy(
            "anthropic:claude-sonnet-4-6", base_url=gateway, trusted_endpoints=trusted
        )
        == _ANTHROPIC_5M
    )
    assert (
        resolve_prompt_cache_policy(
            model_spec, base_url=gateway, trusted_endpoints=trusted
        )
        is None
    )
    # An untrusted gateway resolves nothing regardless.
    assert resolve_prompt_cache_policy(model_spec, base_url=gateway) is None


def test_gateway_detection_rejects_lookalike_hosts() -> None:
    """A lookalike host is not the gateway, so no translation is assumed."""
    for host in ("notsmith.langchain.com", "smith.langchain.com.evil.example"):
        assert (
            resolve_prompt_cache_policy(
                "anthropic:openai/gpt-5.6",
                base_url=f"https://{host}",
                trusted_endpoints={host},
            )
            == _ANTHROPIC_5M
        )


def test_gateway_detection_normalizes_trailing_root_dot() -> None:
    """A fully-qualified host must not slip past the cross-format guard."""
    assert (
        resolve_prompt_cache_policy(
            "anthropic:openai/gpt-5.6",
            base_url="https://smith.langchain.com./gw",
            trusted_endpoints={"smith.langchain.com."},
        )
        is None
    )


def test_cross_format_specs_resolve_normally_off_gateway() -> None:
    """Only the gateway reads a `provider/` prefix as a routing instruction."""
    assert (
        resolve_prompt_cache_policy(
            "anthropic:openai/gpt-5.6",
            base_url="https://gateway.example.com",
            trusted_endpoints={"gateway.example.com"},
        )
        == _ANTHROPIC_5M
    )


def test_trusted_endpoints_accepts_urls_as_well_as_hostnames() -> None:
    """A URL in the trust set must not silently no-op."""
    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6",
        base_url="https://gateway.example.com/v1",
        trusted_endpoints={"https://gateway.example.com/v1"},
    ) == PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")


def test_load_trusted_cache_endpoints_parses_hosts_and_urls() -> None:
    config = {
        "warnings": {
            "trusted_cache_endpoints": [
                "https://smith.langchain.com/gw",
                "gateway.example.com",
                "  spaced.example.com  ",
                "PORTED.example.com:8443",
                "UPPER.example.com",
            ]
        }
    }

    assert load_trusted_cache_endpoints(config) == frozenset(
        {
            "smith.langchain.com",
            "gateway.example.com",
            "spaced.example.com",
            # A port is not part of the trust grain: the host is trusted on
            # every port and both schemes.
            "ported.example.com",
            "upper.example.com",
        }
    )


@pytest.mark.parametrize(
    "entry",
    [
        "",
        42,
        None,
        True,
        ["nested"],
        "not a url at all ::",
        "not a url at all",
        # A comma for a dot used to be accepted as a hostname outright.
        "smith.langchain,com",
        # Wildcards are not supported; storing one would never match.
        "*.example.com",
        "ftp://gateway.example.com",
        "https://",
    ],
)
def test_load_trusted_cache_endpoints_rejects_and_logs_bad_entries(
    entry: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = {"warnings": {"trusted_cache_endpoints": [entry]}}

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset()

    assert "trusted_cache_endpoints" in caplog.text


def test_load_trusted_cache_endpoints_tolerates_missing_or_malformed() -> None:
    assert load_trusted_cache_endpoints({}) == frozenset()
    assert load_trusted_cache_endpoints({"warnings": []}) == frozenset()
    assert load_trusted_cache_endpoints({"warnings": {}}) == frozenset()


def test_load_trusted_cache_endpoints_logs_a_non_list_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare string is the likeliest hand-edit and must not fail silently."""
    config = {"warnings": {"trusted_cache_endpoints": "smith.langchain.com"}}

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset()

    assert "expected a list" in caplog.text


@pytest.mark.parametrize(
    ("bucket", "detail_key"),
    [
        ("generic", "cache_write"),
        ("5m", "ephemeral_5m_input_tokens"),
        ("1h", "ephemeral_1h_input_tokens"),
    ],
)
def test_estimate_rewarm_cost_uses_policy_bucket(
    monkeypatch: pytest.MonkeyPatch,
    bucket: CacheWriteBucket,
    detail_key: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_estimate(
        usage: dict[str, Any],
        model_name: str,
        provider: str,
    ) -> float:
        calls.append(usage)
        assert model_name == "model"
        assert provider == "anthropic"
        details = usage["input_token_details"]
        return 0.1 if "cache_read" in details else 1.25

    monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_estimate)
    policy = PromptCachePolicy(
        "Anthropic",
        300,
        "expired",
        1024,
        bucket,
    )

    estimate = estimate_rewarm_cost(50_000, "anthropic:model", policy)

    assert estimate is not None
    assert estimate.cold_cost_usd == pytest.approx(1.25)
    assert estimate.incremental_cost_usd == pytest.approx(1.15)
    assert calls[0]["input_tokens"] == 50_000
    assert calls[0]["input_token_details"] == {"cache_read": 50_000}
    assert calls[1]["input_token_details"] == {detail_key: 50_000}


def test_estimate_rewarm_cost_requires_cacheable_and_priceable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    assert estimate_rewarm_cost(100, "openai:gpt-5.6", policy) is None

    monkeypatch.setattr(
        "deepagents_code.cost_tracking.estimate_cost",
        lambda *_args: None,
    )
    assert estimate_rewarm_cost(5000, "openai:gpt-5.6", policy) is None


def test_parse_cache_timestamp_requires_timezone() -> None:
    timestamp = datetime(2026, 8, 11, 12, 30, tzinfo=UTC)

    assert parse_cache_timestamp(timestamp.isoformat()) == timestamp
    assert parse_cache_timestamp("2026-08-11T12:30:00") is None
    assert parse_cache_timestamp("not-a-time") is None
    assert parse_cache_timestamp(None) is None


def test_cache_time_formatting() -> None:
    assert format_cache_age(11_520) == "3h 12m"
    assert format_cache_age(300) == "5m"
    assert format_cache_window(1800) == "30m"
    assert format_cache_window(3600) == "1h"
