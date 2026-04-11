from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
PSD_PATH = Path(
    os.environ.get(
        "EDGE_LAB_TIMESLICE_PSD_PATH",
        str(REPO_ROOT / "_outputs-refactored" / "Blender" / "timeslices" / "timeslice-parade.psd"),
    )
).expanduser()
SIZES_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_TRENDING_SIZES_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "edge_lab_final_template_parade_8k_network_20260402" / "current" / "sizes"),
    )
).expanduser()
PROPOSALS_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_TRENDING_PROPOSALS_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "edge_lab_final_template_parade_8k_network_20260402" / "current" / "proposals" / "trending"),
    )
).expanduser()
PHOTOSHOP_APP = os.environ.get("EDGE_LAB_PHOTOSHOP_APP", "Adobe Photoshop 2026")


def require(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def file_spec(name: str, path: Path) -> dict[str, str]:
    return {"name": name, "path": require(path)}


def build_config() -> dict[str, object]:
    tree_sizes = [
        file_spec("small", SIZES_ROOT / "trending_small.png"),
        file_spec("medium", SIZES_ROOT / "trending_medium.png"),
        file_spec("large", SIZES_ROOT / "trending_large.png"),
        file_spec("senescing", SIZES_ROOT / "trending_senescing.png"),
        file_spec("snag", SIZES_ROOT / "trending_snag.png"),
        file_spec("fallen", SIZES_ROOT / "trending_fallen.png"),
        file_spec("artificial", SIZES_ROOT / "trending_artificial.png"),
    ]

    proposals = [
        {
            "name": "proposal-deploy-structure",
            "layers": [
                file_spec("adapt-utility-pole", PROPOSALS_ROOT / "proposal-deploy-structure-adapt-utility-pole.png"),
                file_spec("translocated-log", PROPOSALS_ROOT / "proposal-deploy-structure-translocated-log.png"),
                file_spec("upgrade-feature", PROPOSALS_ROOT / "proposal-deploy-structure-upgrade-feature.png"),
            ],
        },
        {
            "name": "proposal-decay",
            "layers": [
                file_spec("buffer-feature", PROPOSALS_ROOT / "proposal-decay-buffer-feature.png"),
                file_spec("brace-feature", PROPOSALS_ROOT / "proposal-decay-brace-feature.png"),
            ],
        },
        {
            "name": "proposal-recruit",
            "layers": [
                file_spec("buffer-feature", PROPOSALS_ROOT / "proposal-recruit-buffer-feature.png"),
                file_spec("rewild-ground", PROPOSALS_ROOT / "proposal-recruit-rewild-ground.png"),
            ],
        },
        {
            "name": "proposal-colonise",
            "layers": [
                file_spec("rewild-ground", PROPOSALS_ROOT / "proposal-colonise-rewild-ground.png"),
                file_spec("enrich-envelope", PROPOSALS_ROOT / "proposal-colonise-enrich-envelope.png"),
                file_spec("roughen-envelope", PROPOSALS_ROOT / "proposal-colonise-roughen-envelope.png"),
            ],
        },
        {
            "name": "proposal-release-control",
            "layers": [
                file_spec("rejected", PROPOSALS_ROOT / "proposal-release-control-rejected.png"),
                file_spec("reduce-pruning", PROPOSALS_ROOT / "proposal-release-control-reduce-pruning.png"),
                file_spec("eliminate-pruning", PROPOSALS_ROOT / "proposal-release-control-eliminate-pruning.png"),
            ],
        },
    ]

    return {
        "psd_path": require(PSD_PATH),
        "tree_sizes": tree_sizes,
        "proposals": proposals,
    }


def build_jsx(config: dict[str, object]) -> str:
    config_literal = json.dumps(config)
    return f"""#target photoshop
app.displayDialogs = DialogModes.NO;

function findTopLevelGroup(doc, name) {{
    for (var i = 0; i < doc.layerSets.length; i++) {{
        if (doc.layerSets[i].name === name) {{
            return doc.layerSets[i];
        }}
    }}
    return null;
}}

function clearGroup(group) {{
    while (group.layers.length > 0) {{
        group.layers[0].remove();
    }}
}}

function importInto(doc, parent, spec) {{
    var src = app.open(new File(spec.path));
    var imported = src.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
    src.close(SaveOptions.DONOTSAVECHANGES);
    imported.move(parent, ElementPlacement.INSIDE);
    imported.name = spec.name;
    imported.visible = true;
    return imported;
}}

var config = {config_literal};
var file = new File(config.psd_path);
var doc = app.open(file);

var trending = findTopLevelGroup(doc, "trending");
if (trending === null) {{
    trending = doc.layerSets.add();
    trending.name = "trending";
}}
clearGroup(trending);

var proposalsGroup = trending.layerSets.add();
proposalsGroup.name = "proposals";
for (var p = config.proposals.length - 1; p >= 0; p--) {{
    var proposal = config.proposals[p];
    var proposalSet = proposalsGroup.layerSets.add();
    proposalSet.name = proposal.name;
    for (var j = proposal.layers.length - 1; j >= 0; j--) {{
        importInto(doc, proposalSet, proposal.layers[j]);
    }}
}}

var sizesGroup = trending.layerSets.add();
sizesGroup.name = "tree sizes";
for (var i = config.tree_sizes.length - 1; i >= 0; i--) {{
    importInto(doc, sizesGroup, config.tree_sizes[i]);
}}

var psdOpts = new PhotoshopSaveOptions();
psdOpts.layers = true;
psdOpts.embedColorProfile = true;
psdOpts.maximizeCompatibility = true;
doc.saveAs(file, psdOpts, true, Extension.LOWERCASE);
"""


def run() -> None:
    config = build_config()
    with tempfile.TemporaryDirectory(prefix="edge_lab_trending_psd_") as tmpdir:
        jsx_path = Path(tmpdir) / "insert_trending.jsx"
        jsx_path.write_text(build_jsx(config))
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "{PHOTOSHOP_APP}"',
                "-e",
                "with timeout of 7200 seconds",
                "-e",
                f'do javascript file POSIX file "{jsx_path}"',
                "-e",
                "end timeout",
                "-e",
                "end tell",
            ],
            check=True,
        )
    print(f"[insert_trending_into_timeslice_psd] Updated {PSD_PATH}")


if __name__ == "__main__":
    run()
