"""Parse VS Code debug-logs for actual Copilot usage metrics.

Debug-logs contain detailed telemetry of Copilot API requests including
actual token counts, model information, and performance metrics.
This is more complete than session-state since it captures ALL requests.

IMPORTANT: Token Interpretation
- input_tokens: Includes full conversation history/context sent to API
- output_tokens: Actual response tokens from the model
- GitHub typically bills on output tokens only or cached input
- Each user interaction may spawn multiple API requests (retries, internal calls)
- Total tokens in debug-logs ≠ GitHub usage dashboard (different counting methods)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from loguru import logger as log


@dataclass
class DebugLogRequest:
    """A Copilot API request with actual token metrics."""

    timestamp_ms: int
    duration_ms: int
    session_id: str
    request_id: str | None = None
    model: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "unknown"
    error_message: str | None = None
    debug_name: str | None = None  # e.g., "panel/editAgent"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def parse_debug_log_file(
    file_path: Path,
    workspace_id: str,
    session_id: str,
) -> Generator[DebugLogRequest, None, None]:
    """Parse a debug-log JSONL file and yield requests with token data.

    Args:
        file_path: Path to debug-log JSONL file
        workspace_id: VS Code workspace identifier
        session_id: Copilot session ID

    Yields:
        DebugLogRequest objects with token metrics
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    event = json.loads(line)

                    # Only process llm_request events (actual API calls)
                    if event.get("type") != "llm_request":
                        continue

                    attrs = event.get("attrs", {})

                    request = DebugLogRequest(
                        timestamp_ms=event.get("ts", 0),
                        duration_ms=event.get("dur", 0),
                        session_id=session_id,
                        request_id=event.get("spanId"),
                        model=attrs.get("model", "unknown"),
                        input_tokens=attrs.get("inputTokens", 0) or 0,
                        output_tokens=attrs.get("outputTokens", 0) or 0,
                        status=event.get("status", "unknown"),
                        debug_name=attrs.get("debugName"),
                    )

                    # Track errors
                    if request.status != "ok":
                        if "error" in event:
                            request.error_message = str(event.get("error"))
                        log.debug(
                            f"Failed request in {file_path.name}:{line_no} "
                            f"- status={request.status}, model={request.model}"
                        )

                    yield request

                except json.JSONDecodeError as e:
                    log.warning(
                        f"Invalid JSON in {file_path.name}:{line_no}: {e}"
                    )
                except Exception as e:  # noqa: BLE001
                    log.debug(
                        f"Error processing line {line_no} in {file_path.name}: {e}"
                    )

    except OSError as e:
        log.warning(f"Cannot read debug-log file {file_path}: {e}")


def discover_debug_log_files(
    storage_root: Path | None = None,
) -> list[tuple[str, str, Path]]:
    """Discover all debug-log JSONL files in VS Code workspaceStorage.

    Returns:
        List of (workspace_id, session_id, file_path) tuples
    """
    if storage_root is None:
        storage_root = Path.home() / ".vscode-server/data/User/workspaceStorage"

    if not storage_root.exists():
        log.warning(f"workspaceStorage not found: {storage_root}")
        return []

    files = []
    for workspace_dir in storage_root.glob("*"):
        if not workspace_dir.is_dir():
            continue

        workspace_id = workspace_dir.name
        debug_logs = workspace_dir / "GitHub.copilot-chat" / "debug-logs"

        if not debug_logs.exists():
            continue

        for session_dir in debug_logs.glob("*"):
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name

            for jsonl_file in session_dir.glob("*.jsonl"):
                files.append((workspace_id, session_id, jsonl_file))

    log.info(f"Discovered {len(files)} debug-log files")
    return files


def calculate_debug_log_usage(
    storage_root: Path | None = None,
) -> dict:
    """Calculate total Copilot usage from all debug-logs.

    Returns:
        Dict with aggregated usage metrics
    """
    files = discover_debug_log_files(storage_root)

    total_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    model_counts = {}
    error_count = 0

    for workspace_id, session_id, file_path in files:
        for request in parse_debug_log_file(file_path, workspace_id, session_id):
            total_requests += 1
            total_input_tokens += request.input_tokens
            total_output_tokens += request.output_tokens

            model = request.model
            if model not in model_counts:
                model_counts[model] = 0
            model_counts[model] += 1

            if request.status != "ok":
                error_count += 1

    return {
        "total_requests": total_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "model_counts": model_counts,
        "error_count": error_count,
        "file_count": len(files),
    }


if __name__ == "__main__":
    usage = calculate_debug_log_usage()
    print("\nDebug-log Usage Summary:")
    print(f"  Requests: {usage['total_requests']:,}")
    print(f"  Input tokens: {usage['total_input_tokens']:,}")
    print(f"  Output tokens: {usage['total_output_tokens']:,}")
    print(f"  Total tokens: {usage['total_tokens']:,}")
    print(f"  Models: {len(usage['model_counts'])}")
    print(f"  Files: {usage['file_count']}")
    print(f"  Errors: {usage['error_count']}")
