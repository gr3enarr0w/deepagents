"""Provider policy and pricing helpers for cold prompt-cache warnings."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from deepagents_code.model_config import is_langsmith_gateway_host

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)

CacheConfidence = Literal["expired", "may_be_cold"]
CacheWriteBucket = Literal["generic", "5m", "1h"]

_OPENAI_MODEL_VERSION = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?")
_DEFAULT_ENDPOINT_PORTS = {"http": 80, "https": 443}

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


def _normalized_endpoint_cache_identity(base_url: str) -> str | None:
    """Normalize a valid HTTP endpoint, if possible.

    Returns:
        The normalized endpoint identity, or `None` when `base_url` is not an
        HTTP URL with a hostname.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    scheme = parsed.scheme.lower()
    hostname = host.lower().removesuffix(".")
    port = parsed.port
    authority = hostname
    if port is not None and port != _DEFAULT_ENDPOINT_PORTS[scheme]:
        authority = f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{authority}{path}{query}"


def endpoint_cache_identity(base_url: str | None) -> str:
    """Return a stable identity for the endpoint that owns a prompt cache.

    A missing endpoint means the provider's default API. URL spelling details
    that do not select a different server (case, a trailing slash, a default
    port, and fragments) are normalized away. The path and query are retained
    because proxies can route them to separate backends.

    Args:
        base_url: Resolved provider endpoint, or `None` for its default API.

    Returns:
        A checkpoint-safe endpoint identity.
    """
    if base_url is None or not base_url.strip():
        return "default"
    try:
        normalized = _normalized_endpoint_cache_identity(base_url.strip())
    except ValueError:
        # Preserve malformed values as distinct identities. This is deliberately
        # conservative: a bad endpoint must never be considered cache-equivalent
        # to the provider default or to a valid endpoint.
        return f"invalid:{base_url.strip()}"
    if normalized is not None:
        return normalized
    return f"invalid:{base_url.strip()}"


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

    A trailing root dot is removed so the fully-qualified spelling of a host
    (`smith.langchain.com.`) compares equal to the bare one on both sides of
    the trust check -- otherwise it would slip past gateway detection while
    still matching an identically-spelled trust entry.

    Returns:
        The lowercase hostname, or `None` when the scheme is not `http`/`https`
        or the host is empty, whitespace-padded, or only a root dot.
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
    normalized = host.lower().removesuffix(".")
    return normalized or None


_TRUSTED_HOST = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
"""Shape a trusted-endpoint entry must have once reduced to a hostname.

`urlparse` accepts almost any junk as a host -- `smith.langchain,com` and
`not a url` both survive it -- which would silently populate the trust set
with an entry that can never match a real endpoint. Matching this pattern
instead means a typo is reported rather than stored. IPv6 literals are not
accepted; a proxy addressed by raw IPv6 must be trusted by DNS name.
"""


def _trusted_entry_hostname(entry: object) -> str | None:
    """Reduce one configured trust entry to a hostname.

    Args:
        entry: Raw value from the TOML list; any type, validated here.

    Returns:
        The lowercase hostname, or `None` when the entry is unusable.
    """
    if not isinstance(entry, str) or not entry.strip():
        return None
    # Bare hosts are accepted alongside URLs for convenience.
    candidate = entry.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    host = _endpoint_hostname(candidate)
    if host is None or not _TRUSTED_HOST.match(host):
        return None
    return host


def load_trusted_cache_endpoints(
    config: dict[str, Any] | None = None,
) -> frozenset[str]:
    """Read `[warnings].trusted_cache_endpoints` as a set of hostnames.

    Entries declare that an alternate endpoint forwards cache-affecting request
    fields (`cache_control`, `prompt_cache_key`, `prompt_cache_retention`) and
    honors the upstream provider's documented retention.

    Malformed input never raises: an unusable entry is dropped and a value that
    is not a list is ignored wholesale. Because a dropped entry silently leaves
    the warning disabled -- the opposite of what the user edited the file to
    achieve -- every rejection is logged with the offending value.

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
    if entries == []:
        return frozenset()
    if not isinstance(entries, list):
        logger.warning(
            "Ignoring [warnings].trusted_cache_endpoints in config.toml "
            "(expected a list of hostnames, got %s)",
            type(entries).__name__,
        )
        return frozenset()
    hosts: set[str] = set()
    for entry in entries:
        host = _trusted_entry_hostname(entry)
        if host is None:
            logger.warning(
                "Ignoring [warnings].trusted_cache_endpoints entry %r in "
                "config.toml (expected a hostname or http(s) URL)",
                entry,
            )
            continue
        hosts.add(host)
    if hosts:
        logger.debug("Trusting cache endpoints: %s", ", ".join(sorted(hosts)))
    return frozenset(hosts)


def _gateway_effective_model(
    provider: str,
    model_name: str,
    base_url: str | None,
) -> str | None:
    """Resolve the model name a policy lookup should use for a gateway route.

    The LangSmith gateway reads a `provider/model` prefix in the model field to
    route a request to a provider other than the one the wire format implies
    (an OpenAI-format request carrying `anthropic/claude-...`). Such a hop is
    translated between API formats, and translation rewrites or drops the very
    fields a policy assumes (`cache_control`, `prompt_cache_*`), so the
    upstream provider's documented retention no longer describes what happens.

    Any prefix that does not match the wire-format provider is therefore
    treated as a crossing, including prefixes for providers this module cannot
    price -- suppressing a warning is safe, whereas pricing a route whose cache
    semantics were rewritten is not. A matching prefix (`openai/gpt-5.6` in
    OpenAI format) is a same-provider route, forwarded untranslated; the prefix
    is stripped so model-family detection sees the bare name it expects.

    Args:
        provider: Wire-format provider from the `provider:model` spec.
        model_name: Model portion of the spec, as sent to the gateway.
        base_url: Resolved endpoint, or `None` for the provider default.

    Returns:
        The model name to price with, or `None` when the route crosses
        formats. Non-gateway endpoints return the name unchanged -- no
        translation is detectable there.
    """
    if not base_url or not is_langsmith_gateway_host(_endpoint_hostname(base_url)):
        return model_name
    if "/" not in model_name:
        return model_name
    prefix, remainder = model_name.split("/", 1)
    if prefix.strip().lower() != provider or not remainder.strip():
        return None
    return remainder.strip()


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
    user-declared trusted endpoint (see `load_trusted_cache_endpoints`).

    On a LangSmith gateway endpoint, a route that crosses wire formats (e.g. an
    OpenAI-format request routed to an Anthropic model) resolves nothing,
    because translation rewrites or drops the caching fields the policy
    assumes. Cross-format routing through *other* trusted endpoints cannot be
    detected — declaring an endpoint trusted asserts that it does not do this.

    Args:
        model_spec: `provider:model` identifier for the invocation.
        model_params: Request params that affect retention, if any.
        base_url: Resolved endpoint, or `None` for the provider default.
        trusted_endpoints: Bare hostnames (not URLs) the user has declared
            trusted, as returned by `load_trusted_cache_endpoints`.

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

    # Accept URLs as well as hostnames: the parameter's contract is hostnames,
    # but a URL here would otherwise be a silent, total no-op.
    trusted = {
        _trusted_entry_hostname(host) or host.lower()
        for host in trusted_endpoints or ()
    }
    effective_model = _gateway_effective_model(provider, model_name, base_url)
    if effective_model is None:
        return None
    model_name = effective_model

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
