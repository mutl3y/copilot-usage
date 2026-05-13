"""Cost estimation logic for Copilot usage.

Supports both usage-based billing (June 1, 2026+) and request-based billing with
model multipliers for annual plan subscribers staying on legacy billing.

All USD values per 1,000,000 tokens unless noted otherwise.
Source: https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PER_TOKEN_UNIT = 1_000_000          # pricing denominator
AI_CREDIT_USD_VALUE = 0.01          # $0.01 per AI Credit
ESTIMATION_MONTH_DAYS = 30          # standard normalisation window

# Billing model transition date
USAGE_BASED_BILLING_DATE = datetime(2026, 6, 1)  # June 1, 2026

BillingModel = Literal["usage_based", "request_based_legacy"]

# ---------------------------------------------------------------------------
# Model pricing table
# Source: https://docs.github.com/en/copilot/concepts/billing/copilot-requests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    id: str
    display_name: str
    provider: str
    category: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0
    cache_write_per_million: float = 0.0


MODEL_PRICING: dict[str, ModelPricing] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4.1": ModelPricing(
        id="gpt-4.1", display_name="GPT-4.1", provider="OpenAI", category="Versatile",
        input_per_million=2.0, cached_input_per_million=0.5, output_per_million=8.0,
    ),
    "gpt-5-mini": ModelPricing(
        id="gpt-5-mini", display_name="GPT-5 mini", provider="OpenAI", category="Lightweight",
        input_per_million=0.25, cached_input_per_million=0.025, output_per_million=2.0,
    ),
    "gpt-5.2": ModelPricing(
        id="gpt-5.2", display_name="GPT-5.2", provider="OpenAI", category="Versatile",
        input_per_million=1.75, cached_input_per_million=0.175, output_per_million=14.0,
    ),
    "gpt-5.2-codex": ModelPricing(
        id="gpt-5.2-codex", display_name="GPT-5.2-Codex", provider="OpenAI", category="Powerful",
        input_per_million=1.75, cached_input_per_million=0.175, output_per_million=14.0,
    ),
    "gpt-5.3-codex": ModelPricing(
        id="gpt-5.3-codex", display_name="GPT-5.3-Codex", provider="OpenAI", category="Powerful",
        input_per_million=1.75, cached_input_per_million=0.175, output_per_million=14.0,
    ),
    "gpt-5.4": ModelPricing(
        id="gpt-5.4", display_name="GPT-5.4", provider="OpenAI", category="Versatile",
        input_per_million=2.5, cached_input_per_million=0.25, output_per_million=15.0,
    ),
    "gpt-5.4-mini": ModelPricing(
        id="gpt-5.4-mini", display_name="GPT-5.4 mini", provider="OpenAI", category="Lightweight",
        input_per_million=0.75, cached_input_per_million=0.075, output_per_million=4.5,
    ),
    "gpt-5.4-nano": ModelPricing(
        id="gpt-5.4-nano", display_name="GPT-5.4 nano", provider="OpenAI", category="Lightweight",
        input_per_million=0.2, cached_input_per_million=0.02, output_per_million=1.25,
    ),
    "gpt-5.5": ModelPricing(
        id="gpt-5.5", display_name="GPT-5.5", provider="OpenAI", category="Powerful",
        input_per_million=5.0, cached_input_per_million=0.5, output_per_million=30.0,
    ),
    # ── Anthropic ───────────────────────────────────────────────────────────
    "claude-haiku-4.5": ModelPricing(
        id="claude-haiku-4.5", display_name="Claude Haiku 4.5", provider="Anthropic", category="Versatile",
        input_per_million=1.0, cached_input_per_million=0.1, cache_write_per_million=1.25, output_per_million=5.0,
    ),
    "claude-sonnet-4": ModelPricing(
        id="claude-sonnet-4", display_name="Claude Sonnet 4", provider="Anthropic", category="Versatile",
        input_per_million=3.0, cached_input_per_million=0.3, cache_write_per_million=3.75, output_per_million=15.0,
    ),
    "claude-sonnet-4.5": ModelPricing(
        id="claude-sonnet-4.5", display_name="Claude Sonnet 4.5", provider="Anthropic", category="Versatile",
        input_per_million=3.0, cached_input_per_million=0.3, cache_write_per_million=3.75, output_per_million=15.0,
    ),
    "claude-sonnet-4.6": ModelPricing(
        id="claude-sonnet-4.6", display_name="Claude Sonnet 4.6", provider="Anthropic", category="Versatile",
        input_per_million=3.0, cached_input_per_million=0.3, cache_write_per_million=3.75, output_per_million=15.0,
    ),
    "claude-opus-4.5": ModelPricing(
        id="claude-opus-4.5", display_name="Claude Opus 4.5", provider="Anthropic", category="Powerful",
        input_per_million=5.0, cached_input_per_million=0.5, cache_write_per_million=6.25, output_per_million=25.0,
    ),
    "claude-opus-4.6": ModelPricing(
        id="claude-opus-4.6", display_name="Claude Opus 4.6", provider="Anthropic", category="Powerful",
        input_per_million=5.0, cached_input_per_million=0.5, cache_write_per_million=6.25, output_per_million=25.0,
    ),
    "claude-opus-4.7": ModelPricing(
        id="claude-opus-4.7", display_name="Claude Opus 4.7", provider="Anthropic", category="Powerful",
        input_per_million=5.0, cached_input_per_million=0.5, cache_write_per_million=6.25, output_per_million=25.0,
    ),
    # ── Google ──────────────────────────────────────────────────────────────
    "gemini-2.5-pro": ModelPricing(
        id="gemini-2.5-pro", display_name="Gemini 2.5 Pro", provider="Google", category="Powerful",
        input_per_million=1.25, cached_input_per_million=0.125, output_per_million=10.0,
    ),
    "gemini-3-flash": ModelPricing(
        id="gemini-3-flash", display_name="Gemini 3 Flash", provider="Google", category="Lightweight",
        input_per_million=0.5, cached_input_per_million=0.05, output_per_million=3.0,
    ),
    "gemini-3.1-pro": ModelPricing(
        id="gemini-3.1-pro", display_name="Gemini 3.1 Pro", provider="Google", category="Powerful",
        input_per_million=2.0, cached_input_per_million=0.2, output_per_million=12.0,
    ),
    # ── xAI ─────────────────────────────────────────────────────────────────
    "grok-code-fast-1": ModelPricing(
        id="grok-code-fast-1", display_name="Grok Code Fast 1", provider="xAI", category="Lightweight",
        input_per_million=0.2, cached_input_per_million=0.02, output_per_million=1.5,
    ),
    # ── GitHub fine-tuned ────────────────────────────────────────────────────
    "raptor-mini": ModelPricing(
        id="raptor-mini", display_name="Raptor mini", provider="GitHub", category="Versatile",
        input_per_million=0.25, cached_input_per_million=0.025, output_per_million=2.0,
    ),
    "goldeneye": ModelPricing(
        id="goldeneye", display_name="Goldeneye", provider="GitHub", category="Powerful",
        input_per_million=1.25, cached_input_per_million=0.125, output_per_million=10.0,
    ),
}

# Sorted cheapest-output-first for UI display
MODEL_PRICING_LIST: list[ModelPricing] = sorted(
    MODEL_PRICING.values(), key=lambda m: m.output_per_million
)

# ---------------------------------------------------------------------------
# Model multipliers for legacy annual Copilot Pro/Pro+ subscribers (request-based)
# Source: https://docs.github.com/en/copilot/reference/copilot-billing/model-multipliers-for-annual-plans
# 
# Subscribers staying on legacy request-based billing after June 1, 2026 will
# see these multipliers applied to base request costs. Multiplier 1.0 = base cost.
# ---------------------------------------------------------------------------

MODEL_MULTIPLIERS: dict[str, float] = {
    # OpenAI models
    "gpt-4.1": 1.0,
    "gpt-5-mini": 0.2,
    "gpt-5.2": 1.25,
    "gpt-5.2-codex": 1.25,
    "gpt-5.3-codex": 1.25,
    "gpt-5.4": 1.5,
    "gpt-5.4-mini": 0.5,
    "gpt-5.4-nano": 0.15,
    "gpt-5.5": 3.0,
    
    # Anthropic models
    "claude-haiku-4.5": 0.5,
    "claude-sonnet-4": 1.5,
    "claude-sonnet-4.5": 1.5,
    "claude-sonnet-4.6": 1.5,
    "claude-opus-4.5": 3.0,
    "claude-opus-4.6": 3.0,
    "claude-opus-4.7": 3.0,
    
    # Google models
    "gemini-2.5-pro": 1.0,
    "gemini-3-flash": 0.3,
    "gemini-3.1-pro": 1.5,
    
    # xAI models
    "grok-code-fast-1": 0.15,
    
    # GitHub fine-tuned
    "raptor-mini": 0.2,
    "goldeneye": 1.0,
}

def get_model_multiplier(model_id: str) -> float:
    """Get request-based billing multiplier for legacy annual plans.
    
    Returns 1.0 as default for unknown models.
    """
    return MODEL_MULTIPLIERS.get(_strip_prefix(model_id), 1.0)

# ---------------------------------------------------------------------------
# Plan allowances
# ---------------------------------------------------------------------------

CopilotPlan = Literal["free", "pro", "pro_plus", "business", "enterprise", "unknown"]
AllowanceType = Literal["individual", "pooled_org", "limited_or_unknown"]


@dataclass(frozen=True)
class PlanAllowance:
    display_name: str
    allowance_type: AllowanceType
    included_credits_per_month: int | None = None       # individual plans
    included_credits_per_user_per_month: int | None = None  # org plans
    included_usd_value: float | None = None


PLAN_ALLOWANCES: dict[str, PlanAllowance] = {
    "pro": PlanAllowance(
        display_name="Copilot Pro",
        allowance_type="individual",
        included_credits_per_month=1000,
        included_usd_value=10.0,
    ),
    "pro_plus": PlanAllowance(
        display_name="Copilot Pro+",
        allowance_type="individual",
        included_credits_per_month=3900,
        included_usd_value=39.0,
    ),
    "business": PlanAllowance(
        display_name="Copilot Business",
        allowance_type="pooled_org",
        included_credits_per_user_per_month=1900,
    ),
    "enterprise": PlanAllowance(
        display_name="Copilot Enterprise",
        allowance_type="pooled_org",
        included_credits_per_user_per_month=3900,
    ),
    "free": PlanAllowance(
        display_name="Copilot Free",
        allowance_type="limited_or_unknown",
    ),
    "unknown": PlanAllowance(
        display_name="Not selected",
        allowance_type="limited_or_unknown",
    ),
}

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _strip_prefix(model_id: str) -> str:
    """Strip 'copilot/' prefix if present, returning bare model key."""
    return model_id.removeprefix("copilot/")


def get_model_pricing(model_id: str) -> ModelPricing | None:
    """Return pricing for a model_id (with or without 'copilot/' prefix)."""
    return MODEL_PRICING.get(_strip_prefix(model_id))


# ---------------------------------------------------------------------------
# Calculation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelCostRow:
    model_id: str
    display_name: str
    provider: str
    observed_prompt_tokens: int
    observed_output_tokens: int
    monthly_prompt_tokens: float
    monthly_output_tokens: float
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    monthly_credits: int
    # Optional fields for cached token support (usage-based billing)
    cached_input_tokens: int = 0
    cached_read_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0
    # Billing model identifier
    billing_model: BillingModel = "usage_based"


@dataclass
class CostSummary:
    days_observed: int
    total_monthly_usd: float
    total_monthly_credits: int
    rows: list[ModelCostRow] = field(default_factory=list)
    # Trend fields (populated by compute_trend separately)
    trend_delta_pct: int | None = None
    trend_label: str | None = None


@dataclass
class PlanImpact:
    plan_id: str
    status: Literal[
        "within_allowance",
        "over_allowance_within_budget",
        "over_allowance_exceeds_budget",
        "pooled_org",
        "estimate_only",
    ]
    estimated_credits: int
    included_credits: int | None
    extra_budget_credits: int
    overage_credits: int
    estimated_extra_usd: float
    is_within_allowance: bool | None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def compute_monthly_cost(
    token_rows: list[dict],
    days_observed: int,
) -> CostSummary:
    """Compute estimated monthly USD cost from aggregated token rows.

    Args:
        token_rows: list of dicts with keys:
            ``model_id``, ``prompt_tokens``, ``output_tokens``
        days_observed: number of days the token_rows span (used for
            normalising to a 30-day month).

    Returns:
        :class:`CostSummary` with per-model breakdown and totals.
    """
    days = max(1, days_observed)
    scale = ESTIMATION_MONTH_DAYS / days  # >1 when observing less than 30 days

    rows: list[ModelCostRow] = []
    total_usd = 0.0
    total_credits = 0

    for row in token_rows:
        raw_model = row.get("model_id") or row.get("model") or ""
        prompt_obs = int(row.get("prompt_tokens", 0) or 0)
        output_obs = int(row.get("output_tokens", 0) or 0)

        monthly_prompt = prompt_obs * scale
        monthly_output = output_obs * scale

        pricing = get_model_pricing(raw_model)
        if pricing is None:
            # Unknown model: use 0 cost but still record the row
            rows.append(ModelCostRow(
                model_id=raw_model,
                display_name=_strip_prefix(raw_model) or "unknown",
                provider="Unknown",
                observed_prompt_tokens=prompt_obs,
                observed_output_tokens=output_obs,
                monthly_prompt_tokens=monthly_prompt,
                monthly_output_tokens=monthly_output,
                input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
                monthly_credits=0,
            ))
            continue

        input_usd = (monthly_prompt / PER_TOKEN_UNIT) * pricing.input_per_million
        output_usd = (monthly_output / PER_TOKEN_UNIT) * pricing.output_per_million
        row_usd = input_usd + output_usd
        row_credits = math.ceil(row_usd / AI_CREDIT_USD_VALUE)

        total_usd += row_usd
        total_credits += row_credits

        rows.append(ModelCostRow(
            model_id=raw_model,
            display_name=pricing.display_name,
            provider=pricing.provider,
            observed_prompt_tokens=prompt_obs,
            observed_output_tokens=output_obs,
            monthly_prompt_tokens=monthly_prompt,
            monthly_output_tokens=monthly_output,
            input_cost_usd=input_usd,
            output_cost_usd=output_usd,
            total_cost_usd=row_usd,
            monthly_credits=row_credits,
        ))

    rows.sort(key=lambda r: r.total_cost_usd, reverse=True)
    return CostSummary(
        days_observed=days,
        total_monthly_usd=total_usd,
        total_monthly_credits=math.ceil(total_usd / AI_CREDIT_USD_VALUE),
        rows=rows,
    )


def compute_monthly_cost_request_based(
    token_rows: list[dict],
    days_observed: int,
    base_per_request_usd: float = 0.5,
) -> CostSummary:
    """Compute monthly cost under legacy request-based billing with model multipliers.
    
    For annual Copilot Pro/Pro+ subscribers staying on legacy billing after June 1, 2026.
    Applies model multipliers to a base per-request cost.
    
    Args:
        token_rows: list of dicts with ``model_id`` (required), plus optional
            ``prompt_tokens``, ``output_tokens`` for analytics only
        days_observed: number of days for normalization
        base_per_request_usd: base cost per request (default $0.50)
    
    Returns:
        :class:`CostSummary` with per-model breakdown and totals.
    """
    days = max(1, days_observed)
    scale = ESTIMATION_MONTH_DAYS / days
    
    rows: list[ModelCostRow] = []
    total_usd = 0.0
    total_credits = 0
    
    # Group by model to count requests
    model_request_counts: dict[str, int] = {}
    for row in token_rows:
        raw_model = row.get("model_id") or row.get("model") or "unknown"
        model_request_counts[raw_model] = model_request_counts.get(raw_model, 0) + 1
    
    for raw_model, request_count in model_request_counts.items():
        prompt_obs = int(sum(
            int(r.get("prompt_tokens", 0) or 0) 
            for r in token_rows 
            if (r.get("model_id") or r.get("model")) == raw_model
        ))
        output_obs = int(sum(
            int(r.get("output_tokens", 0) or 0)
            for r in token_rows
            if (r.get("model_id") or r.get("model")) == raw_model
        ))
        
        monthly_requests = request_count * scale
        monthly_prompt = prompt_obs * scale
        monthly_output = output_obs * scale
        
        multiplier = get_model_multiplier(raw_model)
        pricing = get_model_pricing(raw_model)
        
        # Cost = base_rate × multiplier × request_count (normalized to month)
        row_usd = base_per_request_usd * multiplier * monthly_requests
        row_credits = math.ceil(row_usd / AI_CREDIT_USD_VALUE)
        
        total_usd += row_usd
        total_credits += row_credits
        
        display_name = pricing.display_name if pricing else _strip_prefix(raw_model)
        provider = pricing.provider if pricing else "Unknown"
        
        rows.append(ModelCostRow(
            model_id=raw_model,
            display_name=display_name,
            provider=provider,
            observed_prompt_tokens=prompt_obs,
            observed_output_tokens=output_obs,
            monthly_prompt_tokens=monthly_prompt,
            monthly_output_tokens=monthly_output,
            input_cost_usd=row_usd * 0.5,  # Split for display
            output_cost_usd=row_usd * 0.5,
            total_cost_usd=row_usd,
            monthly_credits=row_credits,
            billing_model="request_based_legacy",
        ))
    
    rows.sort(key=lambda r: r.total_cost_usd, reverse=True)
    return CostSummary(
        days_observed=days,
        total_monthly_usd=total_usd,
        total_monthly_credits=math.ceil(total_usd / AI_CREDIT_USD_VALUE),
        rows=rows,
    )


def compute_monthly_cost_with_cache(
    token_rows: list[dict],
    days_observed: int,
    cache_hit_rate: float = 0.0,
) -> CostSummary:
    """Compute monthly cost under usage-based billing with prompt caching support.
    
    Calculates costs for input, output, cached input reads, and cache writes.
    
    Args:
        token_rows: list of dicts with keys:
            ``model_id``, ``prompt_tokens``, ``output_tokens``,
            optionally ``cached_input_tokens``, ``cache_write_tokens``
        days_observed: number of days for normalization
        cache_hit_rate: estimated cache hit rate (0.0-1.0) for calculating savings
    
    Returns:
        :class:`CostSummary` with per-model breakdown including cache costs.
    """
    days = max(1, days_observed)
    scale = ESTIMATION_MONTH_DAYS / days
    
    rows: list[ModelCostRow] = []
    total_usd = 0.0
    total_credits = 0
    
    for row in token_rows:
        raw_model = row.get("model_id") or row.get("model") or ""
        prompt_obs = int(row.get("prompt_tokens", 0) or 0)
        output_obs = int(row.get("output_tokens", 0) or 0)
        cached_obs = int(row.get("cached_input_tokens", 0) or 0)
        cache_write_obs = int(row.get("cache_write_tokens", 0) or 0)
        
        monthly_prompt = prompt_obs * scale
        monthly_output = output_obs * scale
        monthly_cached = cached_obs * scale
        monthly_cache_write = cache_write_obs * scale
        
        pricing = get_model_pricing(raw_model)
        if pricing is None:
            rows.append(ModelCostRow(
                model_id=raw_model,
                display_name=_strip_prefix(raw_model) or "unknown",
                provider="Unknown",
                observed_prompt_tokens=prompt_obs,
                observed_output_tokens=output_obs,
                monthly_prompt_tokens=monthly_prompt,
                monthly_output_tokens=monthly_output,
                input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
                monthly_credits=0,
                cached_input_tokens=int(cached_obs),
                billing_model="usage_based",
            ))
            continue
        
        # Calculate costs for each component
        input_usd = (monthly_prompt / PER_TOKEN_UNIT) * pricing.input_per_million
        output_usd = (monthly_output / PER_TOKEN_UNIT) * pricing.output_per_million
        
        # Cached input reads are cheaper (90% discount typical, varies by model)
        cached_read_usd = (monthly_cached / PER_TOKEN_UNIT) * pricing.cached_input_per_million
        
        # Cache writes cost more (Anthropic models have explicit cache_write cost)
        cache_write_usd = (monthly_cache_write / PER_TOKEN_UNIT) * pricing.cache_write_per_million if pricing.cache_write_per_million > 0 else 0.0
        
        row_usd = input_usd + output_usd + cached_read_usd + cache_write_usd
        row_credits = math.ceil(row_usd / AI_CREDIT_USD_VALUE)
        
        total_usd += row_usd
        total_credits += row_credits
        
        rows.append(ModelCostRow(
            model_id=raw_model,
            display_name=pricing.display_name,
            provider=pricing.provider,
            observed_prompt_tokens=prompt_obs,
            observed_output_tokens=output_obs,
            monthly_prompt_tokens=monthly_prompt,
            monthly_output_tokens=monthly_output,
            input_cost_usd=input_usd,
            output_cost_usd=output_usd,
            total_cost_usd=row_usd,
            monthly_credits=row_credits,
            cached_input_tokens=int(cached_obs),
            cached_read_cost_usd=cached_read_usd,
            cache_write_cost_usd=cache_write_usd,
            billing_model="usage_based",
        ))
    
    rows.sort(key=lambda r: r.total_cost_usd, reverse=True)
    return CostSummary(
        days_observed=days,
        total_monthly_usd=total_usd,
        total_monthly_credits=math.ceil(total_usd / AI_CREDIT_USD_VALUE),
        rows=rows,
    )


def compute_monthly_cost_v2(
    token_rows: list[dict],
    days_observed: int,
    billing_model: BillingModel = "usage_based",
) -> CostSummary:
    """Compute monthly cost, routing to the appropriate billing model calculator.
    
    Args:
        token_rows: list of token usage dicts
        days_observed: observation window in days
        billing_model: "usage_based" (default, June 1 2026+) or 
                       "request_based_legacy" (annual plans staying on old model)
    
    Returns:
        :class:`CostSummary` with costs under the selected model.
    """
    if billing_model == "request_based_legacy":
        return compute_monthly_cost_request_based(token_rows, days_observed)
    else:  # usage_based (default)
        # Check if rows have cached token data
        has_cache_data = any(
            row.get("cached_input_tokens", 0) or row.get("cache_write_tokens", 0)
            for row in token_rows
        )
        if has_cache_data:
            return compute_monthly_cost_with_cache(token_rows, days_observed)
        else:
            return compute_monthly_cost(token_rows, days_observed)


def compute_plan_impact(
    summary: CostSummary,
    plan_id: str,
    extra_budget_usd: float = 0.0,
) -> PlanImpact:
    """Compare estimated monthly credits against a Copilot plan allowance."""
    allowance = PLAN_ALLOWANCES.get(plan_id, PLAN_ALLOWANCES["unknown"])
    extra_budget_credits = max(0, math.ceil(extra_budget_usd / AI_CREDIT_USD_VALUE))
    estimated = summary.total_monthly_credits
    warnings: list[str] = []

    if allowance.allowance_type == "individual" and allowance.included_credits_per_month is not None:
        included = allowance.included_credits_per_month
        overage = max(0, estimated - included)
        extra_usd = overage * AI_CREDIT_USD_VALUE
        within = estimated <= included

        if within:
            status = "within_allowance"
        elif overage <= extra_budget_credits:
            status = "over_allowance_within_budget"
        else:
            status = "over_allowance_exceeds_budget"

        return PlanImpact(
            plan_id=plan_id,
            status=status,
            estimated_credits=estimated,
            included_credits=included,
            extra_budget_credits=extra_budget_credits,
            overage_credits=overage,
            estimated_extra_usd=extra_usd,
            is_within_allowance=within,
            warnings=warnings,
        )

    if allowance.allowance_type == "pooled_org":
        per_user = allowance.included_credits_per_user_per_month
        if plan_id == "business":
            warnings.append(
                "Copilot Business: 1,900 credits per assigned user per month, "
                "pooled at the billing entity level. This shows your personal usage "
                "equivalent, not your organisation's final bill."
            )
        else:
            warnings.append(
                "Copilot Enterprise: 3,900 credits per assigned user per month, "
                "pooled at the billing entity level. This shows your personal usage "
                "equivalent, not your organisation's final bill."
            )
        return PlanImpact(
            plan_id=plan_id,
            status="pooled_org",
            estimated_credits=estimated,
            included_credits=per_user,
            extra_budget_credits=extra_budget_credits,
            overage_credits=0,
            estimated_extra_usd=0.0,
            is_within_allowance=None,
            warnings=warnings,
        )

    # free / unknown
    if plan_id == "free":
        warnings.append(
            "Copilot Free has limited usage. This estimate shows token-based value "
            "but may not reflect Free-plan limits exactly."
        )
    else:
        warnings.append("Select your Copilot plan to see personalised plan impact.")

    return PlanImpact(
        plan_id=plan_id,
        status="estimate_only",
        estimated_credits=estimated,
        included_credits=None,
        extra_budget_credits=extra_budget_credits,
        overage_credits=0,
        estimated_extra_usd=0.0,
        is_within_allowance=None,
        warnings=warnings,
    )


def compute_trend(
    tokens_last_30d: int,
    tokens_last_90d: int,
) -> tuple[int, str] | None:
    """Compare last-30d tokens vs 3-month average.

    Returns ``(delta_pct, label)`` or ``None`` when insufficient data.
    Requires both windows to be non-zero.
    """
    if tokens_last_90d == 0:
        return None
    long_term_monthly = tokens_last_90d / 3
    if long_term_monthly == 0:
        return None
    delta_pct = round(((tokens_last_30d - long_term_monthly) / long_term_monthly) * 100)
    if delta_pct >= 0:
        label = f"Your last 30 days are {delta_pct}% higher than your 3-month average."
    else:
        label = f"Your last 30 days are {abs(delta_pct)}% lower than your 3-month average."
    return delta_pct, label
