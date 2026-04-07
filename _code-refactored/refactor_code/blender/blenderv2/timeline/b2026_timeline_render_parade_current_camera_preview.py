from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_render_parade_cleaned_pack as render_pack


SCENE_NAME = "parade"
VIEW_LAYER_NAME = "pathway_state"
OUTPUT_DIR = Path(r"D:\2026 Arboreal Futures\data\renders\timeslices\parade\preview")


def next_version_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    versions = []
    for path in OUTPUT_DIR.glob("v*.png"):
        try:
            versions.append(int(path.stem[1:]))
        except ValueError:
            continue
    next_version = max(versions, default=0) + 1
    return OUTPUT_DIR / f"v{next_version}.png"


def main():
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")

    render_pack.configure_common_render_settings(scene)
    render_pack.restore_production_materials(scene)

    output_path = next_version_path()
    output_dir = output_path.parent

    rendered = render_pack.render_view_layer(
        scene,
        VIEW_LAYER_NAME,
        output_dir,
        file_format="PNG",
        color_depth="8",
        resolution_percentage=50,
        samples=16,
        suffix=output_path.stem,
    )

    if rendered != output_path:
        rendered.replace(output_path)
        rendered = output_path

    digest = hashlib.sha256(rendered.read_bytes()).hexdigest()
    print(f"PREVIEW {rendered} sha256={digest}")


if __name__ == "__main__":
    main()
