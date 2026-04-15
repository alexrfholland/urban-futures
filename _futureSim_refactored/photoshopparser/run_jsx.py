"""Open a PSB in Adobe Photoshop and execute an ExtendScript against it.

Uses osascript + JXA to drive Photoshop from the command line. Reads the
.jsx file as text and passes it to `doJavascript`, which avoids the
`do javascript (POSIX file ...)` path that returns error -1728 for some
script files.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.run_jsx \\
        --psb _data-refactored/_psds/psd-live/parade_single-state_yr180.psb \\
        --jsx _futureSim_refactored/photoshopparser/relink_psb.jsx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

APP_NAME = "Adobe Photoshop 2026"

JXA_TEMPLATE = r"""
ObjC.import('Foundation');

var psbPath = {psb_literal};
var jsxPath = {jsx_literal};

var ps = Application({app_literal});
ps.includeStandardAdditions = true;
ps.activate();

// Open the PSB (no-op if already open). Empty path = no open.
if (psbPath.length > 0) {{
    ps.open(Path(psbPath));
}}

// Read the JSX source as a UTF-8 string.
var err = $();
var nsSrc = $.NSString.stringWithContentsOfFileEncodingError(jsxPath, $.NSUTF8StringEncoding, err);
if (nsSrc.isNil()) {{
    throw new Error("failed to read jsx: " + err.localizedDescription.js);
}}
var src = nsSrc.js;

// Execute in Photoshop's ExtendScript engine.
var result = ps.doJavascript(src);
result === undefined ? "" : String(result);
"""


def _js_literal(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psb", default=None, type=Path,
                    help="Optional PSB to open before running the JSX. "
                         "Omit when the script creates its own document.")
    ap.add_argument("--jsx", required=True, type=Path)
    ap.add_argument("--app", default=APP_NAME,
                    help=f"Photoshop app name (default: {APP_NAME!r})")
    args = ap.parse_args()

    psb_str = ""
    if args.psb is not None:
        psb = args.psb.resolve()
        if not psb.exists():
            print(f"ERROR: psb missing: {psb}", file=sys.stderr)
            return 1
        psb_str = str(psb)
    jsx = args.jsx.resolve()
    if not jsx.exists():
        print(f"ERROR: jsx missing: {jsx}", file=sys.stderr)
        return 1

    script = JXA_TEMPLATE.format(
        psb_literal=_js_literal(psb_str),
        jsx_literal=_js_literal(str(jsx)),
        app_literal=_js_literal(args.app),
    )

    proc = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script],
        capture_output=True, text=True,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
