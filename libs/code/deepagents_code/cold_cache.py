"""Provider policy and pricing helpers for cold prompt-cache warnings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Collection

CacheConfidence = Literal["expired", "may_be_cold"]
CacheWriteBucket = Literal["generic", "5m", "1h"]

_OPENAI_MODEL_VERSION = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?")

_ANTHROPIC_MINIMUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("claude-opus-5", 512),
    ("claude-fable-5", 512),
    ("claude-mythos-5", 512),
    ("claude-opus-4-7", 2048),
    ("claude-mythos-preview", 2048),
    ("claude-haiku-3-5", 2048),
    ("claude-opus-4-6", 4096),
    ("claude-opus-4-5", 4096),
    ("claude-haiku-4-5", 4096),
)
"""Prefixes for Claude models whose cache minimum differs from 1,024 tokens.

From the per-model minimums in Anthropic's prompt-caching docs. Order matters:
the first matching prefix wins, so `claude-mythos-preview` must precede a bare
`claude-mythos` entry if one is ever added.
"""

_ANTHROPIC_DEFAULT_MINIMUM_TOKENS = 1024
"""Cache minimum for the Claude models the table above does not name."""


@dataclass(frozen=True, slots=True)
class PromptCachePolicy:
    """Prompt-cache behavior needed to decide and price a warning."""

    provider_name: str
    window_seconds: int
    confidence: CacheConfidence
    minimum_tokens: int
    write_bucket: CacheWriteBucket


@dataclass(frozen=True, slots=True)
class RewarmEstimate:
    """Estimated input cost for a cold prefix and its warm-cache delta."""

    cold_cost_usd: float
    incremental_cost_usd: float


def _official_endpoint(base_url: str | None, hostname: str) -> bool:
    """Return whether an optional endpoint targets the provider's official API."""
    if not base_url:
        return True
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname == hostname


def _endpoint_hostname(base_url: str) -> str | None:
    """Extract a lowercase hostname from an endpoint URL.

    Returns:
        The lowercase hostname, or `None` when the URL is unusable.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.hostname
    if not host or host != host.strip():
        return None
    return host.lower()


def load_trusted_cache_endpoints(
    config: dict[str, Any] | None = None,
) -> frozenset[str]:
    """Read `[warnings].trusted_cache_endpoints` as a set of hostnames.

    Entries declare that an alternate endpoint forwards cache-affecting request
    fields (`cache_control`, `prompt_cache_key`, `prompt_cache_retention`) and
    honors the upstream provider's documented retention. Non-string entries are
    ignored so a hand-edited typo narrows the set instead of failing the read.

    Args:
        config: Parsed `config.toml` mapping; loaded from disk when omitted.

    Returns:
        Lowercase hostnames of user-trusted endpoints (possibly empty).
    """
    if config is None:
        from deepagents_code.config_manifest import load_config_toml

        config = load_config_toml()
    warnings_section = config.get("warnings", {})
    if not isinstance(warnings_section, dict):
        return frozenset()
    entries = warnings_section.get("trusted_cache_endpoints", [])
    if not isinstance(entries, list):
        return frozenset()
    hosts: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            continue
        # Bare hosts are accepted alongside URLs for convenience.
        candidate = entry.strip()
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        host = _endpoint_hostname(candidate)
        if host:
            hosts.add(host)
    return frozenset(hosts)


_LANGSMITH_GATEWAY_HOST_SUFFIX = "smith.langchain.com"
"""Host suffix identifying LangSmith's managed model gateway.

Subdomains (org-scoped gateway URLs) are included; the suffix match still
requires a dot boundary so a lookalike such as `notsmith.langchain.com` or
`smith.langchain.com.evil.example` does not qualify.
"""

_KNOWN_UPSTREAM_PROVIDERS = frozenset({"anthropic", "openai"})
"""Providers whose models the LangSmith gateway can route cross-format."""


def _is_langsmith_gateway_host(host: str | None) -> bool:
    """Return whether a hostname is the LangSmith managed gateway."""
    return host is not None and (
        host == _LANGSMITH_GATEWAY_HOST_SUFFIX
        or host.endswith(f".{_LANGSMITH_GATEWAY_HOST_SUFFIX}")
    )


def _gateway_cross_format_route(
    provider: str,
    model_name: str,
    base_url: str | None,
) -> bool:
    """Return whether a request will be translated to another API format.

    The LangSmith gateway accepts `provider/model` prefixes in the model field
    to route a request to a different provider than the wire format implies
    (an OpenAI-format request carrying `anthropic/claude-...`). That hop runs
    through the gateway's message translators, which normalize every caching
    signal to Anthropic's plain 5-minute breakpoint (OpenAI → Anthropic) or
    drop `cache_control` outright (Anthropic → OpenAI) — the provider's
    documented retention no longer applies, so no policy can be resolved.

    Same-provider routes (including an explicit matching prefix such as
    `openai/gpt-5.6`) are forwarded untranslated, and a model string without
    a known-provider prefix is served by the wire format's own provider.

    Args:
        provider: Wire-format provider from the `provider:model` spec.
        model_name: Model portion of the spec, as sent to the gateway.
        base_url: Resolved endpoint, or `None` for the provider default.

    Returns:
        `True` when the route is known to cross formats, `False` otherwise
        (including non-gateway endpoints, which apply no translation at all).
    """
    if provider not in _KNOWN_UPSTREAM_PROVIDERS or not base_url:
        return False
    if not _is_langsmith_gateway_host(_endpoint_hostname(base_url)):
        return False
    if "/" not in model_name:
        return False
    prefix = model_name.split("/", 1)[0].strip().lower()
    return prefix in _KNOWN_UPSTREAM_PROVIDERS and prefix != provider


def _openai_uses_thirty_minute_cache(model_name: str) -> bool:
    """Return whether an OpenAI model belongs to the GPT-5.6-or-newer family."""
    match = _OPENAI_MODEL_VERSION.match(model_name.lower())
    if match is None:
        return False
    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    return (major, minor) >= (5, 6)


def _anthropic_minimum_tokens(model_name: str) -> int:
    """Return the documented minimum cacheable prefix for a Claude model."""
    normalized = model_name.lower()
    return next(
        (
            minimum
            for prefix, minimum in _ANTHROPIC_MINIMUM_TOKENS
            if normalized.startswith(prefix)
        ),
        _ANTHROPIC_DEFAULT_MINIMUM_TOKENS,
    )


def resolve_prompt_cache_policy(
    model_spec: str,
    model_params: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
    trusted_endpoints: Collection[str] | None = None,
) -> PromptCachePolicy | None:
    """Resolve a documented cache policy for one effective model invocation.

    Policies apply only when the endpoint is the provider's official API or a
    user-declared trusted endpoint (see `load_trusted_cache_endpoints`), and
    the route stays in the provider's own wire format — a LangSmith gateway
    route that crosses formats (e.g. an OpenAI-format request routed to an
    Anthropic model) is never trusted, because translation rewrites or drops
    the caching fields the policy assumes.

    Returns:
        Matching policy, or `None` when retention cannot be resolved safely.
    """
    if ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not model_name:
        return None
    params = model_params or {}

    trusted = {host.lower() for host in trusted_endpoints or ()}
    if base_url and _gateway_cross_format_route(provider, model_name, base_url):
        return None

    def endpoint_ok(hostname: str) -> bool:
        if _official_endpoint(base_url, hostname):
            return True
        if not base_url or not trusted:
            return False
        return _endpoint_hostname(base_url) in trusted

    if provider == "anthropic":
        if not endpoint_ok("api.anthropic.com"):
            return None
        minimum = _anthropic_minimum_tokens(model_name)
        cache_control = params.get("cache_control")
        ttl = cache_control.get("ttl") if isinstance(cache_control, dict) else None
        if ttl == "1h":
            return PromptCachePolicy("Anthropic", 3600, "expired", minimum, "1h")
        return PromptCachePolicy("Anthropic", 300, "expired", minimum, "5m")

    if provider != "openai" or not endpoint_ok("api.openai.com"):
        return None
    if _openai_uses_thirty_minute_cache(model_name):
        # 30 minutes is the documented guaranteed minimum for GPT-5.6+, but
        # OpenAI may retain the prefix longer, so past the window it can only
        # be treated as possibly cold.
        return PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")

    retention = params.get("prompt_cache_retention")
    # `in_memory` and `24h` are documented maximums ("up to one hour", "a
    # maximum, not a guarantee"): entries may be evicted earlier, so a warning
    # is only defensible once the maximum has passed -- and even then the entry
    # may linger, so the confidence stays "may_be_cold".
    if retention == "in_memory":
        return PromptCachePolicy("OpenAI", 3600, "may_be_cold", 1024, "generic")
    if retention == "24h":
        return PromptCachePolicy("OpenAI", 86400, "may_be_cold", 1024, "generic")
    return None


def estimate_rewarm_cost(
    context_tokens: int,
    model_spec: str,
    policy: PromptCachePolicy,
) -> RewarmEstimate | None:
    """Estimate cold input spend and the incremental cost over a cache hit.

    Returns:
        Price estimate, or `None` when usage cannot be priced defensibly.
    """
    if context_tokens < policy.minimum_tokens or ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    if not provider or not model_name:
        return None

    warm_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
        "input_token_details": {"cache_read": context_tokens},
    }
    detail_key = {
        "generic": "cache_write",
        "5m": "ephemeral_5m_input_tokens",
        "1h": "ephemeral_1h_input_tokens",
    }[policy.write_bucket]
    cold_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
        "input_token_details": {detail_key: context_tokens},
    }

    from deepagents_code.cost_tracking import estimate_cost

    warm_cost = estimate_cost(warm_usage, model_name, provider)
    cold_cost = estimate_cost(cold_usage, model_name, provider)
    if warm_cost is None or cold_cost is None:
        return None
    if not math.isfinite(warm_cost) or not math.isfinite(cold_cost):
        return None
    return RewarmEstimate(
        cold_cost_usd=max(cold_cost, 0.0),
        incremental_cost_usd=max(cold_cost - warm_cost, 0.0),
    )


def parse_cache_timestamp(value: object) -> datetime | None:
    """Parse a persisted UTC timestamp, rejecting malformed or naive values.

    Returns:
        UTC datetime, or `None` when the value is unusable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def format_cache_age(seconds: float) -> str:
    """Format elapsed cache age for compact modal copy.

    Returns:
        Compact hours/minutes label.
    """
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"


def format_cache_window(seconds: int) -> str:
    """Format a provider cache window for compact modal copy.

    Returns:
        Compact hours or minutes label.
    """
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"
