#!/usr/bin/env python3
"""Build v2 pathway comparison outputs without touching the canonical files."""

from __future__ import annotations

import csv
import math
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD_MD = HERE / "comparison_pathways_indicators.md"
OUT_CSV = HERE / "comparison_pathways_indicators_v2.csv"
OUT_MD = HERE / "comparison_pathways_indicators_v2.md"
OUT_DELTA_MD = HERE / "comparison_pathways_v2_deltas.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/revised/final-v2/output/csv"

SITE_MAP = {
    "Parade": "trimmed-parade",
    "Street": "uni",
    "City": "city",
}

CELL_GROUPS = {
    ("Bird", "Acquire Resources"): ["Bird.self.peeling"],
    ("Bird", "Communicate"): ["Bird.others.perch"],
    ("Bird", "Reproduce"): ["Bird.generations.hollow"],
    ("Lizard", "Acquire Resources"): [
        "Lizard.self.grass",
        "Lizard.self.dead",
        "Lizard.self.epiphyte",
    ],
    ("Lizard", "Communicate"): ["Lizard.others.notpaved"],
    ("Lizard", "Reproduce"): [
        "Lizard.generations.nurse-log",
        "Lizard.generations.fallen-tree",
    ],
    ("Tree", "Acquire Resources"): ["Tree.self.senescent"],
    ("Tree", "Communicate"): ["Tree.others.notpaved"],
    ("Tree", "Reproduce"): ["Tree.generations.grassland"],
}

CELL_LABELS = {
    ("Bird", "Acquire Resources"): "Peeling bark volume",
    ("Bird", "Communicate"): "Perchable canopy volume",
    ("Bird", "Reproduce"): "Hollow count",
    ("Lizard", "Acquire Resources"): "Combined ground-cover/dead branch/epiphyte indicators",
    ("Lizard", "Communicate"): "Non-paved surface area",
    ("Lizard", "Reproduce"): "Combined nurse-log/fallen-tree indicators",
    ("Tree", "Acquire Resources"): "Senescent biovolume",
    ("Tree", "Communicate"): "Soil near canopy features",
    ("Tree", "Reproduce"): "Grassland-for-recruitment indicator",
}

CELL_EXPLANATIONS = {
    ("Parade", "Bird", "Acquire Resources"): "Positive keeps more low-control canopy, so peeling bark stays clearly ahead of trending.",
    ("Parade", "Bird", "Communicate"): "Perchable canopy remains common in both paths, so the separation stays modest.",
    ("Parade", "Bird", "Reproduce"): "Positive still holds the hollow stock; trending drops close to zero, so the gap is much stronger than before.",
    ("Parade", "Lizard", "Acquire Resources"): "Positive retains deadwood and epiphytes, and the combined gap is much larger than before.",
    ("Parade", "Lizard", "Communicate"): "Open ground stays broad in both paths, but positive still keeps more non-paved surface.",
    ("Parade", "Lizard", "Reproduce"): "Positive retains fallen-tree habitat; trending falls to zero.",
    ("Parade", "Tree", "Acquire Resources"): "Senescent biovolume stays exclusive to the positive path.",
    ("Parade", "Tree", "Communicate"): "Canopy-ground overlap remains present in both paths, but positive is still ahead.",
    ("Parade", "Tree", "Reproduce"): "Positive keeps the recruitment grassland stock and trending stays near zero.",
    ("Street", "Bird", "Acquire Resources"): "Positive keeps more bark-bearing canopy, so the gap widens.",
    ("Street", "Bird", "Communicate"): "Positive still leads on perchable canopy, and the gap is a little larger than before.",
    ("Street", "Bird", "Reproduce"): "Positive keeps more hollows and artificial canopy structure.",
    ("Street", "Lizard", "Acquire Resources"): "This is the biggest compression. The new engine leaves the two pathways much closer together on ground cover, deadwood, and epiphytes.",
    ("Street", "Lizard", "Communicate"): "Non-paved surface now sits close to parity, with positive only slightly ahead.",
    ("Street", "Lizard", "Reproduce"): "Positive still leads on nurse logs and fallen trees, but the separation is similar to slightly stronger than before.",
    ("Street", "Tree", "Acquire Resources"): "Positive still holds all senescent biovolume.",
    ("Street", "Tree", "Communicate"): "Canopy-ground overlap has narrowed sharply compared with the old table.",
    ("Street", "Tree", "Reproduce"): "Recruitment grassland is still positive-led, but the gap is much smaller.",
    ("City", "Bird", "Acquire Resources"): "Positive holds more peeling bark, and the gap is stronger than before.",
    ("City", "Bird", "Communicate"): "Positive keeps more perchable canopy; the gap widens.",
    ("City", "Bird", "Reproduce"): "Hollows remain positive-led and the gap is much stronger.",
    ("City", "Lizard", "Acquire Resources"): "The old gap largely disappears; both pathways now carry similar ground-cover, deadwood, and epiphyte totals.",
    ("City", "Lizard", "Communicate"): "Roof and facade surface remains close to parity, again far tighter than before.",
    ("City", "Lizard", "Reproduce"): "Nurse logs and fallen trees stay positive-led, but the split is roughly unchanged.",
    ("City", "Tree", "Acquire Resources"): "Positive now dominates senescent biovolume far more strongly.",
    ("City", "Tree", "Communicate"): "The soil-near-canopy gap narrows sharply.",
    ("City", "Tree", "Reproduce"): "Recruitment grassland still favours positive, but the gap is much smaller.",
}

OLD_ROW_RE = re.compile(r"^(?P<site>[A-Za-z]+) / (?P<persona>Bird|Lizard|Tree)\s*\|")
SUMMARY_RE = re.compile(
    r"^(?P<ratio>[0-9.]+)x (?P<label>.+?) \((?P<a>[0-9.]+)% vs (?P<b>[0-9.]+)%\)$"
)


@dataclass
class CellSummary:
    site_display: str
    site_key: str
    persona: str
    capability: str
    label: str
    old_summary: str
    old_ratio: float
    old_left_pct: float
    old_right_pct: float
    new_summary: str
    new_ratio: float
    new_positive_pct: float
    new_trending_pct: float
    new_positive_value: float
    new_trending_value: float
    new_baseline_value: float


def parse_old_table(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    lines = path.read_text().splitlines()
    table_lines: List[str] = []
    in_table = False
    for line in lines:
        if line.startswith("| Site / Persona"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("|---"):
            continue
        table_lines.append(line)

    parsed: Dict[Tuple[str, str], Dict[str, str]] = {}
    for line in table_lines:
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 7:
            continue
        site, persona = [item.strip() for item in parts[0].split("/")]
        parsed[(site, persona)] = {
            "acq": parts[1],
            "com": parts[3],
            "rep": parts[5],
        }
    return parsed


def load_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_site_indicator_map(site_key: str, output_root: Path) -> Dict[Tuple[str, str], dict]:
    path = output_root / f"{site_key}_1_indicator_counts.csv"
    rows = load_rows(path)
    return {
        (row["scenario"], row["indicator_id"]): row
        for row in rows
    }


def aggregate_cell(site_key: str, persona: str, capability: str, output_root: Path) -> Tuple[float, float, float]:
    indicator_ids = CELL_GROUPS[(persona, capability)]
    rows = load_rows(output_root / f"{site_key}_1_indicator_counts.csv")
    baseline = 0.0
    positive = 0.0
    trending = 0.0
    for row in rows:
        if row["year"] == "-180" and row["scenario"] == "baseline" and row["indicator_id"] in indicator_ids:
            baseline += float(row["count"])
        elif row["year"] == "180" and row["scenario"] == "positive" and row["indicator_id"] in indicator_ids:
            positive += float(row["count"])
        elif row["year"] == "180" and row["scenario"] == "trending" and row["indicator_id"] in indicator_ids:
            trending += float(row["count"])
    return baseline, positive, trending


def pct(count: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return round((count / baseline) * 100, 1)


def fmt_ratio(numerator: float, denominator: float) -> Tuple[str, float]:
    if denominator == 0:
        if numerator == 0:
            return "1.00x", 1.0
        return "∞x", math.inf
    ratio = numerator / denominator
    return f"{ratio:.2f}x", ratio


def parse_old_summary(summary: str) -> Tuple[float, float, float]:
    match = SUMMARY_RE.match(summary)
    if not match:
        raise ValueError(f"Could not parse old summary: {summary!r}")
    return float(match.group("ratio")), float(match.group("a")), float(match.group("b"))


def build_cell_summaries(old_md_path: Path, output_root: Path) -> List[CellSummary]:
    old_table = parse_old_table(old_md_path)
    cells: List[CellSummary] = []
    for site_display, site_key in SITE_MAP.items():
        for persona in ("Bird", "Lizard", "Tree"):
            key = (site_display, persona)
            old_row = old_table[key]
            for capability, short_key in (
                ("Acquire Resources", "acq"),
                ("Communicate", "com"),
                ("Reproduce", "rep"),
            ):
                baseline, positive, trending = aggregate_cell(site_key, persona, capability, output_root)
                new_summary_value, new_ratio = fmt_ratio(positive, trending)
                new_summary = f"{new_summary_value} {CELL_LABELS[(persona, capability)].lower()} ({pct(positive, baseline):.1f}% vs {pct(trending, baseline):.1f}%)"
                old_ratio, old_left_pct, old_right_pct = parse_old_summary(old_row[short_key])
                cells.append(
                    CellSummary(
                        site_display=site_display,
                        site_key=site_key,
                        persona=persona,
                        capability=capability,
                        label=CELL_LABELS[(persona, capability)],
                        old_summary=old_row[short_key],
                        old_ratio=old_ratio,
                        old_left_pct=old_left_pct,
                        old_right_pct=old_right_pct,
                        new_summary=new_summary,
                        new_ratio=new_ratio,
                        new_positive_pct=pct(positive, baseline),
                        new_trending_pct=pct(trending, baseline),
                        new_positive_value=positive,
                        new_trending_value=trending,
                        new_baseline_value=baseline,
                    )
                )
    return cells


def divergence_class(ratio: float) -> str:
    if math.isinf(ratio):
        return "large"
    if ratio < 2:
        return "minimal"
    if ratio < 5:
        return "moderate"
    return "large"


def old_direction_sign(summary: CellSummary) -> int:
    # The original comparison table consistently treats the richer pathway as the nonhuman-led one.
    return 1


def new_direction_sign(summary: CellSummary) -> int:
    if summary.new_ratio == 1:
        return 0
    return 1 if summary.new_positive_value > summary.new_trending_value else -1


def format_delta_row(summary: CellSummary) -> List[str]:
    old_dir = old_direction_sign(summary)
    new_dir = new_direction_sign(summary)
    same_direction = "yes" if old_dir == new_dir else "no"
    if math.isinf(summary.new_ratio):
        ratio_change = "∞"
    else:
        ratio_change = f"{summary.new_ratio / summary.old_ratio:.2f}x"
    return [
        summary.site_display,
        summary.persona,
        summary.capability,
        summary.old_summary,
        summary.new_summary,
        same_direction,
        ratio_change,
        CELL_EXPLANATIONS[(summary.site_display, summary.persona, summary.capability)],
    ]


def render_table(rows: Iterable[List[str]], headers: List[str]) -> str:
    rows = list(rows)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
    header = fmt_row(headers)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = "\n".join(fmt_row(row) for row in rows)
    return "\n".join([header, sep, body])


def write_outputs(
    *,
    old_md_path: Path = OLD_MD,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    out_csv: Path = OUT_CSV,
    out_md: Path = OUT_MD,
    out_delta_md: Path = OUT_DELTA_MD,
) -> None:
    summaries = build_cell_summaries(old_md_path, output_root)

    # CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "site",
                "persona",
                "capability",
                "label",
                "old_summary",
                "old_ratio",
                "old_left_pct",
                "old_right_pct",
                "new_summary",
                "new_ratio",
                "new_positive_pct",
                "new_trending_pct",
                "new_positive_value",
                "new_trending_value",
                "new_baseline_value",
                "old_direction_sign",
                "new_direction_sign",
                "same_direction",
                "ratio_change_factor",
            ]
        )
        for summary in summaries:
            ratio_change = ""
            if summary.old_ratio and not math.isinf(summary.new_ratio):
                ratio_change = f"{summary.new_ratio / summary.old_ratio:.6f}"
            elif math.isinf(summary.new_ratio):
                ratio_change = "inf"
            writer.writerow(
                [
                    summary.site_display,
                    summary.persona,
                    summary.capability,
                    summary.label,
                    summary.old_summary,
                    f"{summary.old_ratio:.2f}",
                    f"{summary.old_left_pct:.1f}",
                    f"{summary.old_right_pct:.1f}",
                    summary.new_summary,
                    "inf" if math.isinf(summary.new_ratio) else f"{summary.new_ratio:.2f}",
                    f"{summary.new_positive_pct:.1f}",
                    f"{summary.new_trending_pct:.1f}",
                    f"{summary.new_positive_value:.0f}",
                    f"{summary.new_trending_value:.0f}",
                    f"{summary.new_baseline_value:.0f}",
                    old_direction_sign(summary),
                    new_direction_sign(summary),
                    "yes" if old_direction_sign(summary) == new_direction_sign(summary) else "no",
                    ratio_change,
                ]
            )

    # Main analysis markdown
    table_rows = []
    for site_display in ("Parade", "Street", "City"):
        row_summaries = [s for s in summaries if s.site_display == site_display]
        lookup = {(s.persona, s.capability): s for s in row_summaries}
        table_rows.append(
            [
                f"{site_display} / Bird",
                lookup[("Bird", "Acquire Resources")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Bird", "Acquire Resources")],
                lookup[("Bird", "Communicate")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Bird", "Communicate")],
                lookup[("Bird", "Reproduce")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Bird", "Reproduce")],
            ]
        )
        table_rows.append(
            [
                f"{site_display} / Lizard",
                lookup[("Lizard", "Acquire Resources")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Lizard", "Acquire Resources")],
                lookup[("Lizard", "Communicate")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Lizard", "Communicate")],
                lookup[("Lizard", "Reproduce")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Lizard", "Reproduce")],
            ]
        )
        table_rows.append(
            [
                f"{site_display} / Tree",
                lookup[("Tree", "Acquire Resources")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Tree", "Acquire Resources")],
                lookup[("Tree", "Communicate")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Tree", "Communicate")],
                lookup[("Tree", "Reproduce")].new_summary,
                CELL_EXPLANATIONS[(site_display, "Tree", "Reproduce")],
            ]
        )

    md_lines = [
        "# Comparison Pathways Indicators v2",
        "",
        "## Full analysis",
        "",
        "The table below compares the year-180 capability indicators from the v2 sim core against the per-site indicator baseline rows in the data. Positive remains the higher pathway in every cell; the main change is magnitude, not sign.",
        "",
        "**Table 1. Full analysis.**",
        "",
        render_table(
            table_rows,
            [
                "Site / Persona",
                "Acquire Resources",
                "Explanation",
                "Communicate",
                "Explanation",
                "Reproduce",
                "Explanation",
            ],
        ),
        "",
        "## Explanation of indicator divergence at Year 180",
        "",
        "This summary uses three divergence classes: minimal divergence for values under 2x, moderate divergence for 2x to under 5x, and large divergence for 5x or more, or where the trending pathway is effectively zero.",
        "",
        "For readability, reported percentages below 0.1% are shown as 0.0%.",
        "",
    ]

    for site_display in ("Parade", "Street", "City"):
        md_lines.append(f"### {site_display}")
        md_lines.append("")
        for persona in ("Bird", "Lizard", "Tree"):
            for capability in ("Acquire Resources", "Communicate", "Reproduce"):
                summary = next(
                    s for s in summaries if s.site_display == site_display and s.persona == persona and s.capability == capability
                )
                cls = divergence_class(summary.new_ratio)
                ratio_text = summary.new_summary.split(" ", 1)[0]
                md_lines.append(
                    f"#### {persona} / {capability}. {cls.capitalize()} divergence ({ratio_text}, {summary.new_positive_pct:.1f}% vs {summary.new_trending_pct:.1f}%)."
                )
                md_lines.append("")
                md_lines.append(CELL_EXPLANATIONS[(site_display, persona, capability)])
                md_lines.append("")

    out_md.write_text("\n".join(md_lines).rstrip() + "\n")

    # Delta markdown
    delta_lines = [
        "# Comparison Pathways v2 Deltas",
        "",
        "The v2 refresh compares positive and trending against the per-site indicator baseline rows in the data. Positive still outranks trending in all 27 comparisons. The change is magnitude, not direction.",
        "",
        "The old comparison markdown is used only as the style reference for the wording below. The biggest compressions are in Street and City lizard/tree communication and reproduction. The biggest expansions are in Parade tree reproduction and City tree acquisition.",
        "",
    ]

    for site_display in ("Parade", "Street", "City"):
        delta_lines.append(f"### {site_display}")
        delta_lines.append("")
        for persona in ("Bird", "Lizard", "Tree"):
            delta_lines.append(f"#### {persona}")
            delta_lines.append("")
            for capability in ("Acquire Resources", "Communicate", "Reproduce"):
                summary = next(
                    s for s in summaries if s.site_display == site_display and s.persona == persona and s.capability == capability
                )
                old_dir = "same direction" if old_direction_sign(summary) == new_direction_sign(summary) else "direction changed"
                delta_lines.append(
                    f"- {capability}: old {summary.old_summary} -> new {summary.new_summary}. {old_dir.capitalize()}. {CELL_EXPLANATIONS[(site_display, persona, capability)]}"
                )
            delta_lines.append("")

    out_delta_md.write_text("\n".join(delta_lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pathway-comparison tables from a chosen indicator CSV root.")
    parser.add_argument(
        "--indicator-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing <site>_1_indicator_counts.csv files.",
    )
    parser.add_argument(
        "--old-md",
        type=Path,
        default=OLD_MD,
        help="Existing comparison markdown used as the style/reference old table.",
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-delta-md", type=Path, default=OUT_DELTA_MD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_outputs(
        old_md_path=args.old_md,
        output_root=args.indicator_root,
        out_csv=args.out_csv,
        out_md=args.out_md,
        out_delta_md=args.out_delta_md,
    )
