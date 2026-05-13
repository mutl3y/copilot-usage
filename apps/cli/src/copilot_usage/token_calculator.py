"""Calculate token usage from local Copilot session events.

This module estimates token consumption by:
1. Counting actual tokens in user/assistant messages using tiktoken
2. Estimating tool overhead based on tool invocations
3. Accumulating context usage per session

This provides a local-first estimate since actual token metrics are not
stored by GitHub in the local Copilot cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from loguru import logger as log


@dataclass
class TokenUsageEstimate:
    """Estimated token usage for a session or conversation turn."""

    session_id: str
    turn_id: str | None = None
    user_tokens: int = 0
    assistant_tokens: int = 0
    tool_overhead_tokens: int = 0
    model: str = "unknown"
    timestamp: str = ""

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens on-demand."""
        return (
            self.user_tokens + self.assistant_tokens + self.tool_overhead_tokens
        )


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Lazy-load tiktoken encoder for token counting.

    Uses cl100k_base (same as GPT-4 / Claude / Copilot models).
    Falls back gracefully if tiktoken unavailable.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # noqa: BLE001
        log.debug(f"tiktoken unavailable, using character heuristic: {e}")
        return None


def _count_tokens(text: str | None) -> int:
    """Count tokens in text using tiktoken, or estimate from character count."""
    if not text:
        return 0

    tokenizer = _get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:  # noqa: BLE001
            log.debug(f"Token counting failed: {e}, falling back to heuristic")

    # Fallback: ~1 token per 4 characters (rough estimate for English)
    return max(1, len(text) // 4)


def _estimate_tool_overhead(tool_name: str) -> int:
    """Estimate token overhead for tool invocation.

    Tools consume tokens for:
    - Tool definition/description in system prompt
    - Tool invocation parameters
    - Tool result parsing
    """
    # Base overhead per tool call
    base = 100

    # Additional overhead by tool type (rough estimates)
    overhead_map = {
        "code_execution": 200,  # High overhead for code results
        "file_read": 150,  # File content parsing
        "file_write": 100,  # File operations
        "search": 120,  # Search result parsing
        "browser": 250,  # Rich result content
    }

    for tool_type, overhead in overhead_map.items():
        if tool_type in tool_name.lower():
            return base + overhead

    return base


class SessionTokenCalculator:
    """Calculate token usage for a Copilot session from local event files."""

    def __init__(self, session_dir: Path):
        """Initialize calculator for a session directory.

        Args:
            session_dir: Path to ~/.copilot/session-state/{session-id}
        """
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self.events_file = self.session_dir / "events.jsonl"
        self.metadata_file = self.session_dir / "vscode.metadata.json"
        self._lock = Lock()

    def calculate_session_usage(self) -> TokenUsageEstimate:
        """Calculate total token usage for the entire session.

        Returns:
            TokenUsageEstimate with aggregated counts across all turns
        """
        if not self.events_file.exists():
            return TokenUsageEstimate(session_id=self.session_id)

        estimate = TokenUsageEstimate(session_id=self.session_id)
        model = self._load_model_from_metadata()
        estimate.model = model

        try:
            with open(self.events_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        self._process_event(event, estimate)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"Error reading session {self.session_id}: {e}",
            )

        return estimate

    def calculate_turn_usage(self, turn_id: str) -> TokenUsageEstimate:
        """Calculate token usage for a specific turn/request.

        Args:
            turn_id: Turn ID to aggregate events for

        Returns:
            TokenUsageEstimate for that turn only
        """
        if not self.events_file.exists():
            return TokenUsageEstimate(session_id=self.session_id, turn_id=turn_id)

        estimate = TokenUsageEstimate(session_id=self.session_id, turn_id=turn_id)
        model = self._load_model_from_metadata()
        estimate.model = model

        try:
            with open(self.events_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        # Only process events for this turn
                        if self._get_turn_id_from_event(event) == turn_id:
                            self._process_event(event, estimate)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"Error reading session {self.session_id}, turn {turn_id}: {e}",
            )

        return estimate

    def _process_event(self, event: dict, estimate: TokenUsageEstimate) -> None:
        """Process a single event and accumulate token counts."""
        event_type = event.get("type", "")
        data = event.get("data", {})

        if event_type == "user.message":
            content = data.get("content", "")
            tokens = _count_tokens(content)
            estimate.user_tokens += tokens

        elif event_type == "assistant.message":
            content = data.get("content", "")
            tokens = _count_tokens(content)
            estimate.assistant_tokens += tokens

        elif event_type == "tool.execution_start":
            tool_name = data.get("toolName", "unknown")
            estimate.tool_overhead_tokens += _estimate_tool_overhead(tool_name)

        elif event_type == "tool.execution_complete":
            # Tool results can be large; count if present
            result = data.get("result", "")
            if result:
                tokens = _count_tokens(str(result))
                # Tool results are part of context; add modest token count
                estimate.tool_overhead_tokens += min(tokens // 2, 500)

        # Track timestamp from first event
        if not estimate.timestamp and "timestamp" in event:
            estimate.timestamp = event["timestamp"]

    def _load_model_from_metadata(self) -> str:
        """Load selected model from session metadata."""
        if self.metadata_file.exists():
            try:
                metadata = json.loads(self.metadata_file.read_text())
                return metadata.get("selectedModel", "unknown")
            except Exception as e:  # noqa: BLE001
                log.debug(f"Failed to read metadata: {e}")
        return "unknown"

    @staticmethod
    def _get_turn_id_from_event(event: dict) -> str | None:
        """Extract turn ID from event if present."""
        data = event.get("data", {})
        return data.get("turnId") or data.get("parentId")


def calculate_all_sessions(
    sessions_dir: Path | None = None,
) -> list[TokenUsageEstimate]:
    """Calculate token usage for all sessions in ~/.copilot/session-state.

    Args:
        sessions_dir: Path to session state directory. Defaults to ~/.copilot/session-state

    Returns:
        List of TokenUsageEstimate per session
    """
    if sessions_dir is None:
        sessions_dir = Path.home() / ".copilot" / "session-state"

    if not sessions_dir.exists():
        log.warning(f"Sessions directory not found: {sessions_dir}")
        return []

    estimates = []
    for session_dir in sorted(sessions_dir.glob("*")):
        if not session_dir.is_dir():
            continue

        calculator = SessionTokenCalculator(session_dir)
        try:
            estimate = calculator.calculate_session_usage()
            if estimate.total_tokens > 0:
                estimates.append(estimate)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Failed to calculate usage for {session_dir.name}: {e}")

    return estimates


if __name__ == "__main__":
    # Quick test: calculate usage for all sessions
    print("Calculating token usage for all Copilot sessions...\n")

    estimates = calculate_all_sessions()

    total_user = 0
    total_assistant = 0
    total_tool = 0

    for est in estimates:
        print(
            f"Session {est.session_id[:8]}…"
            f"\n  Model: {est.model}"
            f"\n  User tokens: {est.user_tokens}"
            f"\n  Assistant tokens: {est.assistant_tokens}"
            f"\n  Tool overhead: {est.tool_overhead_tokens}"
            f"\n  Total: {est.total_tokens}\n"
        )
        total_user += est.user_tokens
        total_assistant += est.assistant_tokens
        total_tool += est.tool_overhead_tokens

    grand_total = total_user + total_assistant + total_tool
    print(
        f"Aggregated Usage:\n"
        f"  User:      {total_user:,} tokens\n"
        f"  Assistant: {total_assistant:,} tokens\n"
        f"  Tools:     {total_tool:,} tokens\n"
        f"  TOTAL:     {grand_total:,} tokens"
    )
