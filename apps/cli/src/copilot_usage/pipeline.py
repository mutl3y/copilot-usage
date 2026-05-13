"""Orchestrate an incremental scan: discover → parse → ingest → aggregate → badges."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import duckdb
from loguru import logger as log

from copilot_usage.aggregator import rebuild_aggregates
from copilot_usage.badges import export_badges
from copilot_usage.discovery import (
    discover_all_session_files,
    get_changed_files,
    update_file_index,
)
from copilot_usage.ingest import ingest_parsed_file
from copilot_usage.parser import parse_jsonl, parse_legacy_json, calculate_tokens_from_session_events
from copilot_usage import debug_logs_parser

ProgressCallback = Callable[[str, float | None], None]  # (message, progress_pct)


def run_scan(
    con: duckdb.DuckDBPyConnection,
    *,
    storage_root=None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Execute a full incremental scan pipeline. Returns stats dict."""
    t0 = time.perf_counter()

    def _emit(msg: str, pct: float | None = None):
        log.info(msg)
        if on_progress:
            on_progress(msg, pct)

    _emit("Starting scan…", 0)

    # 1. Start scan run
    con.execute("INSERT INTO scan_runs (files_checked, files_parsed) VALUES (0, 0)")
    scan_id = con.execute("SELECT MAX(scan_id) FROM scan_runs").fetchone()[0]

    # 2. Discover all session files (single directory walk)
    _emit("Discovering session files…", 5)
    all_jsonl, all_legacy = discover_all_session_files(storage_root)
    _emit(f"  Found {len(all_jsonl)} JSONL + {len(all_legacy)} legacy JSON files", 15)
    all_files = all_jsonl + all_legacy

    # 3. Upsert workspaces (even if no changed files, we still want the mapping)
    _emit("Registering workspaces…", 18)
    seen_ws: set[str] = set()
    for ws_id, ws_path, _ in all_files:
        if ws_id not in seen_ws:
            con.execute(
                """INSERT INTO workspaces (workspace_id, workspace_path)
                   VALUES (?, ?)
                   ON CONFLICT (workspace_id) DO UPDATE SET workspace_path = excluded.workspace_path""",
                [ws_id, ws_path],
            )
            seen_ws.add(ws_id)

    # 4. Incremental diff
    _emit("Calculating incremental diff…", 20)
    changed, deleted = get_changed_files(con, all_files)
    _emit(f"  {len(changed)} changed, {len(deleted)} deleted", 25)

    # 5. Parse changed files in parallel, then ingest sequentially
    total_events = 0
    parsed_paths = []
    affected_ws: set[str] = set()
    n_files = len(changed)

    def _parse_one(item):
        ws_id, ws_path, path = item
        log.info("  Parsing {}…", path.name)
        if path.suffix == ".json":
            return parse_legacy_json(path, ws_id, ws_path)
        return parse_jsonl(path, ws_id, ws_path)

    parsed_files = []
    if n_files > 0:
        _emit(f"Parsing {n_files} file(s)…", 25)
        workers = min(n_files, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_parse_one, item): item for item in changed}
            for i, fut in enumerate(as_completed(futures)):
                pct = 25 + ((i + 1) / n_files) * 40  # 25% → 65%
                ws_id, ws_path, path = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse {}: {}", path.name, exc)
                    result = None
                _emit(f"Parsed [{i + 1}/{n_files}] {path.name}", pct)
                if result is not None:
                    parsed_files.append((ws_id, path, result))

        _emit("Ingesting events…", 68)
        for ws_id, path, pf in parsed_files:
            n = ingest_parsed_file(con, pf)
            total_events += n
            parsed_paths.append(path)
            affected_ws.add(ws_id)

    # 5b. Fallback: if normal parsing yielded zero events, try token calculator
    if total_events == 0 and len(all_jsonl) > 0:
        _emit("No events found; attempting fallback token calculation…", 70)
        try:
            from pathlib import Path

            session_state_dir = Path.home() / ".copilot" / "session-state"
            if session_state_dir.exists():
                for session_dir in session_state_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    session_id = session_dir.name
                    event = calculate_tokens_from_session_events(session_id, session_dir)
                    if event:
                        # Create a minimal ParsedFile to ingest the fallback event
                        from copilot_usage.parser import ParsedFile

                        pf = ParsedFile(
                            source_path=session_dir / "events.jsonl",
                            workspace_id="",  # Will be backfilled
                            workspace_path="",
                            data_source="session_state_fallback",
                        )
                        pf.anchor = None
                        pf.requests = [event]
                        n = ingest_parsed_file(con, pf)
                        total_events += n
                        affected_ws.add("")  # Mark for aggregate rebuild
        except Exception as e:  # noqa: BLE001
            log.warning(f"Fallback token calculation failed: {e}")

    # 5c. Ingest from debug-logs (actual API call metrics)
    _emit("Scanning debug-logs for actual usage metrics…", 75)
    try:
        from copilot_usage.parser import RequestEvent, ParsedFile

        debug_log_files = debug_logs_parser.discover_debug_log_files(storage_root)
        if debug_log_files:
            events_from_debug = 0
            
            # Batch requests by file to avoid deleting within same file
            for workspace_id, session_id, file_path in debug_log_files:
                file_requests = []
                request_counter = 0
                
                for request in debug_logs_parser.parse_debug_log_file(
                    file_path, workspace_id, session_id
                ):
                    # Convert debug-log request to RequestEvent format
                    # Use a unique index per request in this file
                    event = RequestEvent(
                        chat_session_id=request.session_id,
                        request_index=request_counter,  # Counter ensures uniqueness
                        request_id=request.request_id,  # Preserve the spanId
                        model_id=f"copilot/{request.model}",
                        timestamp_ms=request.timestamp_ms,
                        prompt_tokens=request.input_tokens,
                        output_tokens=request.output_tokens,
                        tool_call_rounds=0,
                        tokens_estimated=False,  # These are actual tokens!
                    )
                    file_requests.append(event)
                    request_counter += 1
                
                # Ingest all requests from this file in one batch
                if file_requests:
                    pf = ParsedFile(
                        source_path=file_path,
                        workspace_id=workspace_id,
                        workspace_path="",  # Not available from debug-logs
                        data_source="debug_logs",
                    )
                    pf.anchor = None
                    pf.requests = file_requests
                    n = ingest_parsed_file(con, pf)
                    total_events += n
                    events_from_debug += len(file_requests)
                    affected_ws.add(workspace_id)

                    # Update progress occasionally
                    if events_from_debug % 500 == 0:
                        pct = 75 + ((events_from_debug / 10000) * 5)  # 75% → 80%
                        _emit(
                            f"Ingesting debug-log events… ({events_from_debug:,})",
                            pct,
                        )

            if events_from_debug > 0:
                _emit(f"  Ingested {events_from_debug:,} events from debug-logs", 80)
    except Exception as e:  # noqa: BLE001
        log.warning(f"Debug-logs ingest failed: {e}")

    # 6. Update file index
    _emit("Updating file index…", 85)
    update_file_index(con, parsed_paths, deleted, scan_id)

    # 7. Rebuild aggregates (incremental when possible)
    _emit("Rebuilding aggregates…", 90)
    rebuild_aggregates(con, affected_ws or None)

    # 8. Export badges
    _emit("Exporting badges…", 96)
    export_badges(con)

    # 9. Finalize scan run
    elapsed = time.perf_counter() - t0
    con.execute(
        """UPDATE scan_runs
           SET finished_at = now(), files_checked = ?, files_parsed = ?
           WHERE scan_id = ?""",
        [len(all_files), len(changed), scan_id],
    )

    stats = {
        "scan_id": scan_id,
        "files_total": len(all_files),
        "files_jsonl": len(all_jsonl),
        "files_legacy_json": len(all_legacy),
        "files_parsed": len(changed),
        "files_deleted": len(deleted),
        "events_ingested": total_events,
        "elapsed_s": round(elapsed, 2),
    }
    _emit(
        f"Scan #{scan_id} complete: {len(changed)} files parsed, "
        f"{total_events} events ingested in {elapsed:.2f}s",
        100,
    )
    return stats
