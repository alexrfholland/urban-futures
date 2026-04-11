from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_futureSim_refactored"))

from _futureSim_refactored.paths import hook_tree_ply_library_dir


TREE_PLY_FOLDER = hook_tree_ply_library_dir()

RESOURCE_PROPERTY_BY_RAW_VALUE_ZERO_BASED = {
    0: "resource_other",
    1: "resource_dead branch",
    2: "resource_peeling bark",
    3: "resource_perch branch",
    4: "resource_epiphyte",
    5: "resource_fallen log",
    6: "resource_hollow",
}

RESOURCE_PROPERTY_BY_RAW_VALUE_ONE_BASED = {
    1: "resource_other",
    2: "resource_dead branch",
    3: "resource_peeling bark",
    4: "resource_perch branch",
    5: "resource_epiphyte",
    6: "resource_fallen log",
    7: "resource_hollow",
}

RESOURCE_PROPERTY_ORDER = [
    "resource_hollow",
    "resource_epiphyte",
    "resource_dead branch",
    "resource_perch branch",
    "resource_peeling bark",
    "resource_fallen log",
    "resource_other",
]


def parse_ascii_ply(path):
    lines = path.read_text(encoding="latin1").splitlines()
    header = []
    vertex_rows = []
    tail_rows = []
    vertex_count = None
    in_vertex = False
    property_names = []
    end_header_index = None

    for index, line in enumerate(lines):
        header.append(line)
        stripped = line.strip()
        if stripped.startswith("format "):
            if stripped != "format ascii 1.0":
                raise ValueError(f"{path} is not an ascii 1.0 PLY")
        elif stripped.startswith("element vertex "):
            vertex_count = int(stripped.split()[-1])
            in_vertex = True
        elif stripped.startswith("element ") and not stripped.startswith("element vertex "):
            in_vertex = False
        elif in_vertex and stripped.startswith("property "):
            parts = stripped.split()
            property_names.append(" ".join(parts[2:]))
        elif stripped == "end_header":
            end_header_index = index
            break

    if end_header_index is None or vertex_count is None:
        raise ValueError(f"Failed to parse header for {path}")

    data_lines = lines[end_header_index + 1 :]
    if len(data_lines) < vertex_count:
        raise ValueError(f"{path} does not contain enough vertex rows")

    vertex_rows = [row.split() for row in data_lines[:vertex_count]]
    tail_rows = data_lines[vertex_count:]

    return {
        "header_lines": header[:-1],
        "property_names": property_names,
        "vertex_rows": vertex_rows,
        "tail_rows": tail_rows,
        "vertex_count": vertex_count,
    }


def build_rewritten_ply(parsed, path):
    property_names = parsed["property_names"]
    vertex_rows = parsed["vertex_rows"]

    existing_resource_properties = [name for name in property_names if name.startswith("resource_")]
    if existing_resource_properties:
        return None

    int_resource_index = property_names.index("int_resource")
    raw_values = [int(float(row[int_resource_index])) for row in vertex_rows]
    counts = Counter(raw_values)

    mapping = (
        RESOURCE_PROPERTY_BY_RAW_VALUE_ZERO_BASED
        if 0 in counts
        else RESOURCE_PROPERTY_BY_RAW_VALUE_ONE_BASED
    )
    used_resource_properties = [
        resource_name
        for resource_name in RESOURCE_PROPERTY_ORDER
        if any(mapping.get(value) == resource_name for value in counts)
    ]

    if not used_resource_properties:
        raise ValueError(f"No resource properties could be derived for {path}")

    new_header = []
    inserted = False
    for line in parsed["header_lines"]:
        new_header.append(line)
        if line.strip().startswith("property ") and not inserted:
            # wait until the final vertex property line before inserting
            pass

    # Rebuild header so new properties sit at the end of the vertex property block.
    rebuilt_header = []
    in_vertex = False
    for index, line in enumerate(parsed["header_lines"]):
        stripped = line.strip()
        rebuilt_header.append(line)
        if stripped.startswith("element vertex "):
            in_vertex = True
            continue
        if in_vertex and stripped.startswith("property "):
            next_is_non_vertex_property = (
                index + 1 >= len(parsed["header_lines"])
                or not parsed["header_lines"][index + 1].strip().startswith("property ")
            )
            if next_is_non_vertex_property:
                for resource_name in used_resource_properties:
                    rebuilt_header.append(f"property float {resource_name}")
                in_vertex = False
        elif stripped.startswith("element ") and not stripped.startswith("element vertex "):
            in_vertex = False

    new_vertex_rows = []
    for row in vertex_rows:
        raw_value = int(float(row[int_resource_index]))
        resource_name = mapping.get(raw_value)
        extras = []
        for candidate in used_resource_properties:
            extras.append("1.00000000" if candidate == resource_name else "0.00000000")
        new_vertex_rows.append(row + extras)

    content_lines = rebuilt_header + ["end_header"]
    content_lines.extend(" ".join(row) for row in new_vertex_rows)
    content_lines.extend(parsed["tail_rows"])

    return {
        "content": "\n".join(content_lines) + "\n",
        "used_resource_properties": used_resource_properties,
        "value_counts": counts,
    }


def main():
    modified = []
    skipped = []
    for path in sorted(TREE_PLY_FOLDER.glob("*.ply")):
        parsed = parse_ascii_ply(path)
        rewritten = build_rewritten_ply(parsed, path)
        if rewritten is None:
            skipped.append(path.name)
            continue
        path.write_text(rewritten["content"], encoding="latin1")
        modified.append(
            {
                "name": path.name,
                "used_resource_properties": rewritten["used_resource_properties"],
                "value_counts": dict(sorted(rewritten["value_counts"].items())),
            }
        )

    print(f"Modified PLY files: {len(modified)}")
    for item in modified:
        print(item["name"])
        print(f"  used_resource_properties: {item['used_resource_properties']}")
        print(f"  int_resource_value_counts: {item['value_counts']}")

    print(f"Skipped PLY files already containing resource_* properties: {len(skipped)}")
    return modified


if __name__ == "__main__":
    main()
