"""Batch driver: render all 15 compositor runs for 4.10 / parade_timeline @ 8k.

Per COMPOSITOR_SYNC_CONTRACT.md:
  Inputs  : _data-refactored/blenderv2/output/4.10/parade_timeline/
  Outputs : _data-refactored/compositor/outputs/4.10/parade_timeline/<family>__<ts>/

Per COMPOSITOR_RUN.md / template contract:
  Canonical blends own the graph.
  Runners are thin wrappers over _fast_runner_core (animation=True).
  This driver only sets env vars and shells out to Blender headless.

Runs (15 total):
  ao, normals, resources, bioenvelope, shading, base  (6 state-agnostic)
  mist x (positive, trending)                         (2)
  depth_outliner x (positive, trending, priority)     (3)
  proposal_and_interventions x (positive, trending)   (2)
  proposal_only x (positive, trending)                (2) [future-ready]
  proposal_outline x (positive, trending)             (2) [future-ready]
  proposal_colored_depth_outlines x (positive, trending) (2) [future-ready]
  size_outline x (positive, trending, priority)       (3) [future-ready]
  sizes_single_input x (positive, trending, priority) (3) [future-ready]

Usage:
  uv run python _futureSim_refactored/blender/compositor/scripts/batch_parade_timeline_4_10.py
  uv run python ... --only ao,normals           # subset
  uv run python ... --dry-run                   # print plan without rendering
  uv run python ... --res 8k64s                 # switch resolution
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
SCRIPTS_DIR = _THIS.parent

SIM_ROOT_DEFAULT = "4.10"
EXR_FAMILY_DEFAULT = "parade_timeline"

# These are populated by configure() before plan_runs() runs.
SIM_ROOT = SIM_ROOT_DEFAULT
EXR_FAMILY = EXR_FAMILY_DEFAULT
CASE = EXR_FAMILY_DEFAULT
EXR_DIR = REPO_ROOT / "_data-refactored" / "blenderv2" / "output" / SIM_ROOT / EXR_FAMILY
OUTPUT_ROOT = (
    REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / SIM_ROOT / EXR_FAMILY
)

BLENDER_BIN_DEFAULT = Path("C:/Program Files/Blender Foundation/Blender 4.2/blender.exe")


def configure(sim_root: str, exr_family: str) -> None:
    """Set the module-level paths from --sim-root / --exr-family."""
    global SIM_ROOT, EXR_FAMILY, CASE, EXR_DIR, OUTPUT_ROOT
    SIM_ROOT = sim_root
    EXR_FAMILY = exr_family
    CASE = exr_family
    EXR_DIR = (
        REPO_ROOT / "_data-refactored" / "blenderv2" / "output" / SIM_ROOT / EXR_FAMILY
    )
    OUTPUT_ROOT = (
        REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / SIM_ROOT / EXR_FAMILY
    )


def exr(view_layer: str, res: str) -> Path:
    return EXR_DIR / f"{CASE}__{view_layer}__{res}.exr"


@dataclass
class Run:
    """One compositor_run: family + runner script + env + output dir tag."""

    family: str
    runner: str
    env: dict[str, str]
    tag: str = ""

    def run_name(self, ts: str) -> str:
        if self.tag:
            return f"{self.family}__{ts}__{self.tag}"
        return f"{self.family}__{ts}"


def plan_runs(res: str) -> list[Run]:
    existing_pos = str(exr("existing_condition_positive", res))
    existing_trend = str(exr("existing_condition_trending", res))
    positive = str(exr("positive_state", res))
    trending = str(exr("trending_state", res))
    priority = str(exr("positive_priority_state", res))
    bio_pos = str(exr("bioenvelope_positive", res))
    bio_trend = str(exr("bioenvelope_trending", res))

    # The "positive_state / trending_state / priority" branch loops.
    state_branches = [
        ("positive", positive),
        ("trending", trending),
    ]
    state_branches_with_priority = state_branches + [("priority", priority)]

    runs: list[Run] = []

    # 3-input: ao, normals
    runs.append(Run("ao", "render_current_ao.py", {
        "COMPOSITOR_EXISTING_EXR": existing_pos,
        "COMPOSITOR_PATHWAY_EXR": positive,
        "COMPOSITOR_PRIORITY_EXR": priority,
    }))
    runs.append(Run("normals", "render_current_normals.py", {
        "COMPOSITOR_EXISTING_EXR": existing_pos,
        "COMPOSITOR_PATHWAY_EXR": positive,
        "COMPOSITOR_PRIORITY_EXR": priority,
    }))

    # 3-input: resources (positive / priority / trending)
    runs.append(Run("resources", "render_current_resources.py", {
        "COMPOSITOR_PATHWAY_EXR": positive,
        "COMPOSITOR_PRIORITY_EXR": priority,
        "COMPOSITOR_TRENDING_EXR": trending,
    }))

    # 3-input: bioenvelope (existing / bio_pos / bio_trend)
    runs.append(Run("bioenvelope", "render_current_bioenvelope.py", {
        "COMPOSITOR_EXISTING_EXR": existing_pos,
        "COMPOSITOR_BIOENVELOPE_EXR": bio_pos,
        "COMPOSITOR_BIOENVELOPE_TRENDING_EXR": bio_trend,
    }))

    # 7-input: shading
    runs.append(Run("shading", "render_current_shading.py", {
        "COMPOSITOR_EXISTING_EXR": existing_pos,
        "COMPOSITOR_EXISTING_TRENDING_EXR": existing_trend,
        "COMPOSITOR_PATHWAY_EXR": positive,
        "COMPOSITOR_PRIORITY_EXR": priority,
        "COMPOSITOR_BIOENVELOPE_EXR": bio_pos,
        "COMPOSITOR_BIOENVELOPE_TRENDING_EXR": bio_trend,
    }))

    # base — per-reroute runner, single input (existing_condition_positive)
    runs.append(Run("base", "render_current_base.py", {
        "COMPOSITOR_EXISTING_EXR": existing_pos,
    }))

    # intervention_int x (positive bio, trending bio) — bioenvelope EXR input
    for tag, bio_exr in [("positive", bio_pos), ("trending", bio_trend)]:
        runs.append(Run("intervention_int", "render_current_intervention_int.py", {
            "COMPOSITOR_EXR": bio_exr,
        }, tag=tag))

    # mist x (positive, trending)
    for tag, state_exr in state_branches:
        runs.append(Run("mist", "render_current_mist.py", {
            "COMPOSITOR_EXR": state_exr,
        }, tag=tag))

    # depth_outliner x (positive, trending, priority)
    for tag, state_exr in state_branches_with_priority:
        runs.append(Run("depth_outliner", "render_current_depth_outliner.py", {
            "COMPOSITOR_EXR": state_exr,
        }, tag=tag))

    # proposal_and_interventions x (positive, trending)
    for tag, state_exr in state_branches:
        runs.append(Run(
            "proposal_and_interventions",
            "render_current_proposal_and_interventions.py",
            {"COMPOSITOR_EXR": state_exr},
            tag=tag,
        ))

    # proposal_only x (positive, trending)
    for tag, state_exr in state_branches:
        runs.append(Run("proposal_only", "render_current_proposal_only.py", {
            "COMPOSITOR_EXR": state_exr,
        }, tag=tag))

    # proposal_outline x (positive, trending)
    for tag, state_exr in state_branches:
        runs.append(Run("proposal_outline", "render_current_proposal_outline.py", {
            "COMPOSITOR_EXR": state_exr,
        }, tag=tag))

    # proposal_colored_depth_outlines x (positive, trending)
    for tag, state_exr in state_branches:
        runs.append(Run(
            "proposal_colored_depth_outlines",
            "render_current_proposal_colored_depth_outlines.py",
            {"COMPOSITOR_EXR": state_exr},
            tag=tag,
        ))

    # size_outline x (positive, trending, priority)
    for tag, state_exr in state_branches_with_priority:
        runs.append(Run("size_outline", "render_current_size_outline.py", {
            "COMPOSITOR_EXR": state_exr,
        }, tag=tag))

    # sizes_single_input x (positive, trending, priority)
    for tag, state_exr in state_branches_with_priority:
        runs.append(Run(
            "sizes_single_input",
            "render_current_sizes_single_input.py",
            {"COMPOSITOR_EXR": state_exr},
            tag=tag,
        ))

    return runs


def verify_inputs(runs: list[Run]) -> None:
    missing: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for _, path in run.env.items():
            if path in seen:
                continue
            seen.add(path)
            if not Path(path).exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing EXR inputs:\n  " + "\n  ".join(missing)
        )


def _run_to_manifest_entry(run: Run, ts: str) -> dict:
    """Convert a Run into a manifest entry for _session_runner.py."""
    run_name = run.run_name(ts)
    out_dir = OUTPUT_ROOT / run_name
    return {
        "runner_module": run.runner.replace(".py", ""),
        "tag": f"{run.family}" + (f"/{run.tag}" if run.tag else ""),
        "output_dir": str(out_dir),
        "env": dict(run.env),
    }


def _shard_runs(entries: list[dict], n_workers: int) -> list[list[dict]]:
    """Round-robin shard to balance heterogeneous run durations across workers."""
    chunks: list[list[dict]] = [[] for _ in range(n_workers)]
    for i, entry in enumerate(entries):
        chunks[i % n_workers].append(entry)
    return [c for c in chunks if c]


def run_parallel(runs: list[Run], blender_bin: Path, ts: str, n_workers: int,
                 dry_run: bool) -> list[tuple[Run, bool]]:
    """Dispatch runs across N parallel Blender sessions via _session_runner.py."""
    entries = [_run_to_manifest_entry(r, ts) for r in runs]
    chunks = _shard_runs(entries, n_workers)
    n_workers = len(chunks)

    session_runner = SCRIPTS_DIR / "_session_runner.py"
    if not session_runner.exists():
        raise FileNotFoundError(f"session runner missing: {session_runner}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"parade_batch_{ts}_"))
    manifests: list[Path] = []
    for i, chunk in enumerate(chunks):
        m = tmp_dir / f"manifest_{i}.json"
        m.write_text(json.dumps(chunk, indent=2))
        manifests.append(m)

    print(f"\n=== parallel: {len(entries)} runs across {n_workers} worker(s)")
    for i, chunk in enumerate(chunks):
        tags = ", ".join(e["tag"] for e in chunk)
        print(f"    worker {i}: {len(chunk)} runs -> {tags}")
    if dry_run:
        for m in manifests:
            m.unlink()
        tmp_dir.rmdir()
        return [(r, True) for r in runs]

    for entry in entries:
        Path(entry["output_dir"]).mkdir(parents=True, exist_ok=True)

    procs: list[tuple[subprocess.Popen, Path, int]] = []
    t0 = time.time()
    for i, manifest in enumerate(manifests):
        cmd = [
            str(blender_bin),
            "--background",
            "--factory-startup",
            "--python",
            str(session_runner),
            "--",
            str(manifest),
        ]
        log_path = tmp_dir / f"worker_{i}.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"    spawning worker {i} -> log {log_path}")
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((p, log_path, i))

    for p, log_path, i in procs:
        rc = p.wait()
        print(f"    worker {i} exit rc={rc} (log: {log_path})")

    elapsed = time.time() - t0
    print(f"    all workers done in {elapsed:.1f}s")

    tag_to_ok: dict[str, bool] = {}
    for manifest in manifests:
        result_file = manifest.with_suffix(".result.json")
        if not result_file.exists():
            for entry in json.loads(manifest.read_text()):
                tag_to_ok[entry["tag"]] = False
            continue
        for r in json.loads(result_file.read_text()):
            tag_to_ok[r["tag"]] = bool(r["ok"])

    results: list[tuple[Run, bool]] = []
    for run in runs:
        tag = f"{run.family}" + (f"/{run.tag}" if run.tag else "")
        results.append((run, tag_to_ok.get(tag, False)))
    return results


def run_one(run: Run, blender_bin: Path, ts: str, dry_run: bool) -> bool:
    run_name = run.run_name(ts)
    out_dir = OUTPUT_ROOT / run_name
    runner_path = SCRIPTS_DIR / run.runner

    env = os.environ.copy()
    env["COMPOSITOR_OUTPUT_DIR"] = str(out_dir)
    env.update(run.env)

    cmd = [
        str(blender_bin),
        "--background",
        "--factory-startup",
        "--python",
        str(runner_path),
    ]

    print(f"\n=== [{run.family}{'/' + run.tag if run.tag else ''}] -> {run_name}")
    print(f"    runner : {run.runner}")
    print(f"    out    : {out_dir}")
    for k, v in sorted(run.env.items()):
        print(f"    {k:40s} {Path(v).name}")
    if dry_run:
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=env, check=False)
    elapsed = time.time() - t0
    pngs = sorted(out_dir.glob("*.png"))
    ok = result.returncode == 0 and len(pngs) > 0
    if result.returncode != 0:
        status = f"FAIL(rc={result.returncode})"
    elif not pngs:
        status = "FAIL(no PNGs — Blender swallowed an exception?)"
    else:
        status = f"OK ({len(pngs)} PNGs)"
    print(f"    {status} in {elapsed:.1f}s")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated family names to include (default: all)")
    parser.add_argument("--skip", type=str, default="",
                        help="comma-separated family names to exclude")
    parser.add_argument("--res", type=str, default="8k", choices=["8k", "8k64s"],
                        help="EXR resolution tag (default: 8k)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan without rendering")
    parser.add_argument("--blender-bin", type=Path, default=BLENDER_BIN_DEFAULT)
    parser.add_argument("--timestamp", type=str, default="",
                        help="override run timestamp (default: now)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="parallel worker count (>=2 enables session-wrapped "
                             "parallel mode; default 1 = one subprocess per run)")
    parser.add_argument("--sim-root", type=str, default=SIM_ROOT_DEFAULT,
                        help=f"sim_root (default: {SIM_ROOT_DEFAULT})")
    parser.add_argument("--exr-family", type=str, default=EXR_FAMILY_DEFAULT,
                        help=f"exr_family (default: {EXR_FAMILY_DEFAULT})")
    args = parser.parse_args(argv)

    configure(args.sim_root, args.exr_family)

    if not args.dry_run and not args.blender_bin.exists():
        print(f"ERROR: blender binary not found: {args.blender_bin}", file=sys.stderr)
        return 2

    if not EXR_DIR.exists():
        print(f"ERROR: EXR dir missing: {EXR_DIR}", file=sys.stderr)
        return 2

    runs = plan_runs(args.res)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if only:
        runs = [r for r in runs if r.family in only]
    if skip:
        runs = [r for r in runs if r.family not in skip]
    if not runs:
        print("No runs selected.", file=sys.stderr)
        return 2

    verify_inputs(runs)

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch: sim_root={SIM_ROOT} exr_family={EXR_FAMILY} res={args.res} ts={ts}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"{len(runs)} runs planned")

    if args.parallel >= 2:
        n = min(args.parallel, len(runs))
        results = run_parallel(runs, args.blender_bin, ts, n, args.dry_run)
    else:
        results = []
        for run in runs:
            ok = run_one(run, args.blender_bin, ts, args.dry_run)
            results.append((run, ok))

    print("\n=== SUMMARY ===")
    for run, ok in results:
        name = f"{run.family}" + (f"/{run.tag}" if run.tag else "")
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")

    fails = [r for r, ok in results if not ok]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
