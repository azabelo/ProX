#!/usr/bin/env python3
"""
View a **character-granular** diff between the input document and the OpenRouter output
for global row ``n``, using the same YAML resolution as ``print_nth_input_output.py``.

**Diff algorithm:** Python's ``difflib.SequenceMatcher`` on the two full strings (Unicode
code points / Python ``str`` elements), with ``autojunk=False``, using ``get_opcodes()``.
Chunk delimiters from ``run_openrouter_chunked`` (``### OPENROUTER_CHUNK_PROGRAM ###``) are
replaced by a placeholder **before** diffing so they are not counted as insertions; in the
HTML they render as **yellow** neutral markers, not red/green change.

Each opcode is one of ``equal | replace | delete | insert`` over index ranges in the
input and output strings. This is Ratcliff/Obershelp–style *gestalt* pattern matching
(not the same as Git's Myers line diff). ``autojunk=False`` disables heuristics that
skip detailed matching on large repeated regions, which usually yields finer (and
slower) character-level blocks for long web text.

**No HTML file is written** (nothing under ``.cache/`` or the repo). The page is served
once from an ephemeral ``127.0.0.1`` HTTP server (in-memory HTML), the default browser
is opened—similar in spirit to ``plotly``'s ``show()`` opening a transient window. The
local server shuts down after a short idle period; **the browser tab does not close
automatically** (browsers do not allow that from Python), but **reload will fail** once
the server has stopped.

Usage::

  python openrouter_data_generation/view_nth_input_output_diff.py \\
    openrouter_data_generation/example_openrouter_rewrite.yaml 7

  python openrouter_data_generation/view_nth_input_output_diff.py config.yaml 1 --one-based
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import sys
import threading
import time
import webbrowser
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_MOD_DIR = Path(__file__).resolve().parent

# Must match ``run_openrouter_chunked.PROGRAM_CHUNK_SEPARATOR``
PROGRAM_CHUNK_SEPARATOR = "\n\n### OPENROUTER_CHUNK_PROGRAM ###\n\n"
# Private-use placeholder for delimiter during diff (skipped if present in either string)
_DELIM_PLACEHOLDER = "\ufffc"


def _load_print_nth_module() -> Any:
    name = "print_nth_input_output"
    path = _MOD_DIR / "print_nth_input_output.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _emit_after_for_diff(fragment: str, *, delim_mode: bool) -> str:
    """Escape ``after`` slice; if ``delim_mode``, expand placeholders to yellow delimiter spans."""
    if not delim_mode:
        return html.escape(fragment)
    esc_sep = html.escape(PROGRAM_CHUNK_SEPARATOR)
    sep_html = f'<span class="chunk-delimiter">{esc_sep}</span>'
    return sep_html.join(html.escape(p) for p in fragment.split(_DELIM_PLACEHOLDER))


def _build_merged_diff_html(before: str, after: str) -> str:
    """Single stream: deletions then insertions for ``replace`` (like interleaved diff)."""
    delim_mode = (
        bool(PROGRAM_CHUNK_SEPARATOR)
        and (PROGRAM_CHUNK_SEPARATOR in after)
        and (_DELIM_PLACEHOLDER not in before)
        and (_DELIM_PLACEHOLDER not in after)
    )
    after_for_sm = after.replace(PROGRAM_CHUNK_SEPARATOR, _DELIM_PLACEHOLDER) if delim_mode else after

    sm = SequenceMatcher(None, before, after_for_sm, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(f'<span class="eq">{html.escape(before[i1:i2])}</span>')
        elif tag == "delete":
            parts.append(f'<del>{html.escape(before[i1:i2])}</del>')
        elif tag == "insert":
            frag = after_for_sm[j1:j2]
            parts.append(f'<ins>{_emit_after_for_diff(frag, delim_mode=delim_mode)}</ins>')
        elif tag == "replace":
            parts.append(f'<del>{html.escape(before[i1:i2])}</del>')
            frag = after_for_sm[j1:j2]
            parts.append(f'<ins>{_emit_after_for_diff(frag, delim_mode=delim_mode)}</ins>')
    return "".join(parts)


def _build_full_html_document(
    *,
    before: str,
    after: str,
    meta: dict[str, Any],
    truncated: bool,
) -> str:
    merged = _build_merged_diff_html(before, after)
    title = f"diff row {meta['global_idx']} (local {meta['local_row']})"
    warn = (
        '<p class="warn">Strings were truncated for diffing; see script <code>--max-chars</code>.</p>'
        if truncated
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         margin: 1rem 1.5rem; line-height: 1.45; background: #1e1e1e; color: #d4d4d4; }}
  .meta {{ color: #9cdcfe; margin-bottom: 1rem; font-size: 0.85rem; white-space: pre-wrap; }}
  .warn {{ color: #f9c74f; }}
  pre.diff {{ white-space: pre-wrap; word-break: break-word; margin: 0; }}
  del {{ background: rgba(244, 67, 54, 0.35); text-decoration: line-through; }}
  ins {{ background: rgba(76, 175, 80, 0.45); text-decoration: none; }}
  .chunk-delimiter {{ background: rgba(255, 213, 79, 0.75); color: #1a1a1a; border-radius: 3px; padding: 0 2px; }}
  .eq {{ }}
  .algo {{ color: #858585; font-size: 0.8rem; margin-top: 1.5rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
{warn}
<div class="meta">{html.escape(str(meta["yaml_config"]))}
input: {html.escape(str(meta["shard_path"]))}
output: {html.escape(str(meta["output_parquet"]))}</div>
<pre class="diff">{merged}</pre>
<p class="algo">Algorithm: <code>difflib.SequenceMatcher(..., autojunk=False).get_opcodes()</code> on input vs output with chunk delimiters replaced by a placeholder (U+FFFC) so they are not diff insertions; yellow spans restore the delimiter for display.</p>
</body></html>"""


def _serve_ephemeral_html(
    html_doc: str,
    *,
    open_browser: bool,
    wait_first_request_sec: float,
    keep_alive_after_first_sec: float,
    bind_port: int,
) -> None:
    """Serve HTML from RAM on localhost; optionally open browser; then shut down."""
    body = html_doc.encode("utf-8")
    got_main = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                got_main.set()
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", bind_port), _Handler)
    host, port = httpd.server_address[:2]
    url = f"http://{host}:{port}/"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        if open_browser:
            webbrowser.open(url)
        print(f"Ephemeral diff server: {url}", flush=True)
        if not got_main.wait(timeout=wait_first_request_sec):
            print("[warn] No GET / received; open the URL manually if needed.", file=sys.stderr)
        else:
            time.sleep(keep_alive_after_first_sec)
    finally:
        httpd.shutdown()
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_config", type=Path, help="Path to run_openrouter_chunked YAML.")
    ap.add_argument("n", type=int, help="Global row index (0-based unless --one-based).")
    ap.add_argument("--one-based", action="store_true", help="Interpret n as 1-based.")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=800_000,
        help="Max combined chars (input+output) before truncation for diff (default 800000).",
    )
    ap.add_argument("--no-browser", action="store_true", help="Do not open the default web browser.")
    ap.add_argument(
        "--keep-alive-sec",
        type=float,
        default=3.0,
        help="Seconds to keep serving after the first GET / (default 3).",
    )
    ap.add_argument(
        "--wait-sec",
        type=float,
        default=90.0,
        help="Max seconds to wait for the browser to request / before giving up (default 90).",
    )
    ap.add_argument("--port", type=int, default=0, help="Bind port (0 = ephemeral).")
    args = ap.parse_args()

    pn = _load_print_nth_module()
    try:
        source_text, out_text, meta = pn.load_input_output_for_yaml_row(
            args.yaml_config, args.n, one_based=args.one_based
        )
    except IndexError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    lim = args.max_chars
    truncated = False
    if len(source_text) + len(out_text) > lim:
        half = lim // 2
        source_text = source_text[:half]
        out_text = out_text[:half]
        truncated = True

    html_doc = _build_full_html_document(
        before=source_text, after=out_text, meta=meta, truncated=truncated
    )

    print(
        "Diff algorithm: difflib.SequenceMatcher (Ratcliff/Obershelp–style), "
        "character-level get_opcodes(), autojunk=False",
        flush=True,
    )

    if not args.no_browser:
        _serve_ephemeral_html(
            html_doc,
            open_browser=True,
            wait_first_request_sec=args.wait_sec,
            keep_alive_after_first_sec=args.keep_alive_sec,
            bind_port=args.port,
        )
    else:
        print("Built HTML in memory only (--no-browser); not serving.", flush=True)


if __name__ == "__main__":
    main()
