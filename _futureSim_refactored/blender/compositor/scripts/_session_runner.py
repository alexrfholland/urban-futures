"""Multi-run session driver: executes N compositor runs inside one Blender process.

Invoked as:
    blender --background --factory-startup --python _session_runner.py -- <manifest.json>

Manifest schema (JSON array of run entries):
    [
        {
            "runner_module": "render_current_ao",
            "tag": "ao",
            "output_dir": "<absolute path>",
            "env": {
                "COMPOSITOR_EXISTING_EXR": "<abs>",
                "COMPOSITOR_PATHWAY_EXR": "<abs>",
                "COMPOSITOR_PRIORITY_EXR": "<abs>",
                ...any other env vars the runner reads
            }
        },
        ...
    ]

For each entry:
  - set os.environ with the entry's env (and COMPOSITOR_OUTPUT_DIR = output_dir)
  - import (or reload) runner_module and call its main()
  - count *.png in output_dir, record result

Writes <manifest>.result.json on completion. Exit code 1 if any run failed.

Amortizes Blender cold start (~20s) across all runs in the chunk. bpy.data is
reset by run_fast_render's bpy.ops.wm.open_mainfile, so runs are isolated from
one another.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _parse_argv() -> Path:
    if "--" not in sys.argv:
        raise SystemExit("expected `-- <manifest.json>` after blender args")
    idx = sys.argv.index("--")
    extras = sys.argv[idx + 1:]
    if not extras:
        raise SystemExit("manifest.json path missing after --")
    return Path(extras[0]).resolve()


def main() -> int:
    manifest_path = _parse_argv()
    runs = json.loads(manifest_path.read_text())

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    saved_env = os.environ.copy()
    results: list[dict] = []

    t_session_start = time.time()
    print(f"[session] manifest: {manifest_path}")
    print(f"[session] {len(runs)} runs queued")

    for i, run in enumerate(runs, start=1):
        tag = run.get("tag") or run["runner_module"]
        output_dir = Path(run["output_dir"])
        print(f"\n[session] --- run {i}/{len(runs)}: {tag} ---")
        print(f"[session]   runner : {run['runner_module']}")
        print(f"[session]   out    : {output_dir}")

        # reset env to baseline, apply this run's env
        os.environ.clear()
        os.environ.update(saved_env)
        for k, v in run.get("env", {}).items():
            os.environ[k] = v
        os.environ["COMPOSITOR_OUTPUT_DIR"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        ok = False
        err = ""
        try:
            mod = importlib.import_module(run["runner_module"])
            mod.main()
            ok = True
        except SystemExit as e:
            ok = (e.code in (0, None))
            err = f"SystemExit({e.code})"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        pngs = sorted(output_dir.glob("*.png"))
        elapsed = time.time() - t0
        run_ok = ok and len(pngs) > 0
        status = (
            f"OK ({len(pngs)} PNGs)" if run_ok
            else f"FAIL ({err or 'no PNGs written'})"
        )
        print(f"[session]   {status} in {elapsed:.1f}s")
        results.append({
            "tag": tag,
            "runner_module": run["runner_module"],
            "output_dir": str(output_dir),
            "ok": run_ok,
            "pngs": len(pngs),
            "elapsed_s": round(elapsed, 2),
            "error": err,
        })

    os.environ.clear()
    os.environ.update(saved_env)

    result_path = manifest_path.with_suffix(".result.json")
    result_path.write_text(json.dumps(results, indent=2))
    elapsed_total = time.time() - t_session_start
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[session] done: {n_ok}/{len(results)} OK in {elapsed_total:.1f}s")
    print(f"[session] results -> {result_path}")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
