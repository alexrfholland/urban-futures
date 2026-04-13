"""
Fast extraction of individual point-data arrays from legacy binary VTK PolyData
files, without loading the full mesh into pyvista.

Usage:
    from _futureSim_refactored.sim.vtk_extract import extract_point_array

    arr = extract_point_array("path/to/file.vtk", "resource_hollow")
    # Returns numpy array (numeric) or list[str] (string arrays)

    # List available arrays without reading data:
    from _futureSim_refactored.sim.vtk_extract import list_arrays
    for name, dtype, n in list_arrays("path/to/file.vtk"):
        print(f"{name}: {dtype} x {n}")
"""
from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from urllib.parse import unquote as _url_decode

# ---------------------------------------------------------------------------
# Known column registry — dtype string as written by VTK binary writer.
# Lets us skip arrays by byte-size without parsing headers we've already seen.
# Keys are array names; values are VTK dtype strings.
# ---------------------------------------------------------------------------
KNOWN_COLUMNS: dict[str, str] = {
    # simulation bookkeeping
    "sim_Nodes":                              "vtktypeint64",
    "sim_Turns":                              "vtktypeint64",
    # scenario flags (float64 encoded as 0/1/-1)
    "scenario_rewildingEnabled":              "double",
    "scenario_rewildGroundRecruitZone":       "double",
    "scenario_nodeRewildRecruitZone":         "double",
    "scenario_underCanopyRecruitZone":        "double",
    "scenario_underCanopyLinkedRecruitZone":  "double",
    "sim_averageResistance":                  "double",
    # scenario strings
    "scenario_under-node-treatment":          "string",
    "scenario_bioEnvelope":                   "string",
    "scenario_outputs":                       "string",
    # proposals
    "proposal_decay":                         "string",
    "proposal_decay_intervention":            "string",
    "proposal_release_control":               "string",
    "proposal_release_control_intervention":  "string",
    "proposal_deploy_structure":              "string",
    "proposal_deploy_structure_intervention": "string",
    "proposal_recruit":                       "string",
    "proposal_recruit_intervention":          "string",
    "proposal_colonise":                      "string",
    "proposal_colonise_intervention":         "string",
    # resources & stats (float64, NaN for non-tree voxels)
    "resource_hollow":                        "double",
    "resource_epiphyte":                      "double",
    "resource_dead branch":                   "double",
    "resource_perch branch":                  "double",
    "resource_peeling bark":                  "double",
    "resource_fallen log":                    "double",
    "resource_other":                         "double",
    "stat_hollow":                            "double",
    "stat_epiphyte":                          "double",
    "stat_dead branch":                       "double",
    "stat_perch branch":                      "double",
    "stat_peeling bark":                      "double",
    "stat_fallen log":                        "double",
    "stat_other":                             "double",
    # forest attributes
    "forest_precolonial":                     "string",
    "forest_size":                            "string",
    "forest_control":                         "string",
    # recruitment
    "recruit_isNewTree":                      "string",
    "recruit_hasbeenReplanted":               "string",
    "recruit_mechanism":                      "string",
    "recruit_year":                           "double",
    "recruit_mortality_rate":                 "double",
    "recruit_mortality_cohort":               "double",
    # masks & indicators (unsigned_char / bool)
    "maskForRewilding":                       "unsigned_char",
    "indicator_Bird_self_peeling":            "unsigned_char",
    "indicator_Bird_others_perch":            "unsigned_char",
    "indicator_Bird_generations_hollow":      "unsigned_char",
    "indicator_Lizard_self_grass":            "unsigned_char",
    "indicator_Lizard_self_dead":             "unsigned_char",
    "indicator_Lizard_self_epiphyte":         "unsigned_char",
    "indicator_Lizard_others_notpaved":       "unsigned_char",
    "indicator_Lizard_generations_nurse-log": "unsigned_char",
    "indicator_Lizard_generations_fallen-tree": "unsigned_char",
    "indicator_Tree_self_senescent":          "unsigned_char",
    "indicator_Tree_others_notpaved":         "unsigned_char",
    "indicator_Tree_generations_grassland":   "unsigned_char",
    # search layers
    "search_bioavailable":                    "string",
    "search_design_action":                   "string",
    "search_urban_elements":                  "string",
    # blender proposal encodings (int16)
    "blender_proposal-decay":                 "short",
    "blender_proposal-release-control":       "short",
    "blender_proposal-recruit":               "short",
    "blender_proposal-colonise":              "short",
    "blender_proposal-deploy-structure":      "short",
}

# ---------------------------------------------------------------------------
# VTK binary dtype → (byte_size, numpy_dtype)
# ---------------------------------------------------------------------------
_FIXED_DTYPES: dict[str, tuple[int, str]] = {
    "vtktypeint64":  (8, ">i8"),
    "double":        (8, ">f8"),
    "float":         (4, ">f4"),
    "int":           (4, ">i4"),
    "unsigned_char": (1, "u1"),
    "short":         (2, ">i2"),
}


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _skip_strings(data: bytes, offset: int, n: int) -> int:
    """Advance past *n* length-prefixed strings. Returns offset after the last."""
    pos = offset
    for _ in range(n):
        b = data[pos]
        if 0xC0 <= b <= 0xFF:          # single-byte prefix: length = b - 0xC0
            pos += 1 + (b - 0xC0)
        elif b == 0xD9:                 # str8: next byte is length
            pos += 2 + data[pos + 1]
        elif b == 0xDA:                 # str16: next 2 bytes are length (big-endian)
            pos += 3 + int.from_bytes(data[pos + 1:pos + 3], "big")
        else:
            raise ValueError(f"Unknown string prefix 0x{b:02X} at offset {pos}")
    return pos


def _read_strings(data: bytes, offset: int, n: int) -> list[str]:
    """Decode *n* length-prefixed strings starting at *offset*."""
    pos = offset
    result: list[str] = []
    for _ in range(n):
        b = data[pos]
        if 0xC0 <= b <= 0xFF:
            length = b - 0xC0
            result.append(data[pos + 1:pos + 1 + length].decode("ascii"))
            pos += 1 + length
        elif b == 0xD9:
            length = data[pos + 1]
            result.append(data[pos + 2:pos + 2 + length].decode("ascii"))
            pos += 2 + length
        elif b == 0xDA:
            length = int.from_bytes(data[pos + 1:pos + 3], "big")
            result.append(data[pos + 3:pos + 3 + length].decode("ascii"))
            pos += 3 + length
        else:
            raise ValueError(f"Unknown string prefix 0x{b:02X} at offset {pos}")
    return result


def _find_field_start(data: bytes) -> tuple[int, int]:
    """Locate 'POINT_DATA ... FIELD FieldData N' and return (scan_pos, n_fields)."""
    pd_pos = data.index(b"POINT_DATA")
    fd_match = re.search(rb"FIELD FieldData (\d+)\n", data[pd_pos:])
    if fd_match is None:
        raise ValueError("No FIELD FieldData section found after POINT_DATA")
    return pd_pos + fd_match.end(), int(fd_match.group(1))


def _parse_header_at(data: bytes, pos: int) -> tuple[str, int, int, str, int]:
    """Parse an array header line at *pos*.

    Returns (name, ncomp, ntuples, dtype_str, blob_start).
    VTK encodes spaces in array names as ``%20``; this function decodes them.
    """
    line_end = data.index(b"\n", pos)
    header = data[pos:line_end].decode("ascii")
    parts = header.split()
    name = _url_decode(parts[0])  # e.g. "resource_dead%20branch" → "resource_dead branch"
    return name, int(parts[1]), int(parts[2]), parts[3], line_end + 1


def _skip_array(data: bytes, blob_start: int, ncomp: int, ntuples: int, dtype: str) -> int:
    """Return offset just past the array's data blob."""
    if dtype == "string":
        return _skip_strings(data, blob_start, ntuples)
    byte_size = _FIXED_DTYPES[dtype][0]
    return blob_start + ncomp * ntuples * byte_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_arrays(vtk_path: str | Path) -> list[tuple[str, str, int]]:
    """Return [(name, dtype, n_tuples), ...] for every point-data field array.

    Only reads/parses headers — does not decode any data blobs.
    """
    data = Path(vtk_path).read_bytes()
    pos, n_fields = _find_field_start(data)
    result: list[tuple[str, str, int]] = []

    for _ in range(n_fields):
        while data[pos:pos + 1] == b"\n":
            pos += 1
        name, ncomp, ntuples, dtype, blob_start = _parse_header_at(data, pos)
        result.append((name, dtype, ntuples))
        pos = _skip_array(data, blob_start, ncomp, ntuples, dtype)

    return result


def extract_point_array(
    vtk_path: str | Path,
    target: str,
) -> np.ndarray | list[str] | None:
    """Extract a single point-data array by name from a legacy binary VTK file.

    Strategy:
      1. If *target* is in KNOWN_COLUMNS, use the registered dtype to validate
         against the file header and skip preceding arrays efficiently.
      2. Otherwise, fall back to sequential header parsing + skip.

    Returns:
      - numpy array (native-endian) for numeric types
      - list[str] for string arrays
      - None if the array is not found
    """
    data = Path(vtk_path).read_bytes()
    pos, n_fields = _find_field_start(data)

    for _ in range(n_fields):
        # skip inter-array newlines
        while data[pos:pos + 1] == b"\n":
            pos += 1

        name, ncomp, ntuples, dtype, blob_start = _parse_header_at(data, pos)

        if name == target:
            # — Found it —
            if dtype == "string":
                return _read_strings(data, blob_start, ntuples)

            if dtype not in _FIXED_DTYPES:
                raise ValueError(f"Unsupported dtype '{dtype}' for array '{name}'")

            byte_size, np_dtype = _FIXED_DTYPES[dtype]
            total = ncomp * ntuples * byte_size
            raw = data[blob_start:blob_start + total]
            arr = np.frombuffer(raw, dtype=np_dtype)
            if ncomp > 1:
                arr = arr.reshape(ntuples, ncomp)
            # Convert to native byte order for downstream use
            return arr.astype(arr.dtype.newbyteorder("="))

        # — Not the target, skip past —
        # Use known dtype if available (validates against file header)
        skip_dtype = dtype
        if name in KNOWN_COLUMNS:
            known = KNOWN_COLUMNS[name]
            if known != dtype:
                # Registry mismatch — trust the file header, but warn
                import warnings
                warnings.warn(
                    f"vtk_extract: KNOWN_COLUMNS says '{name}' is {known}, "
                    f"but file header says {dtype}. Using file header."
                )

        pos = _skip_array(data, blob_start, ncomp, ntuples, skip_dtype)

    return None


def extract_multiple(
    vtk_path: str | Path,
    targets: list[str],
) -> dict[str, np.ndarray | list[str]]:
    """Extract several arrays in a single pass (one file read, sequential scan).

    Returns a dict mapping each found target name to its data.
    Missing targets are silently omitted.
    """
    data = Path(vtk_path).read_bytes()
    pos, n_fields = _find_field_start(data)
    remaining = set(targets)
    result: dict[str, np.ndarray | list[str]] = {}

    for _ in range(n_fields):
        if not remaining:
            break

        while data[pos:pos + 1] == b"\n":
            pos += 1

        name, ncomp, ntuples, dtype, blob_start = _parse_header_at(data, pos)

        if name in remaining:
            if dtype == "string":
                result[name] = _read_strings(data, blob_start, ntuples)
            elif dtype in _FIXED_DTYPES:
                byte_size, np_dtype = _FIXED_DTYPES[dtype]
                total = ncomp * ntuples * byte_size
                arr = np.frombuffer(data[blob_start:blob_start + total], dtype=np_dtype)
                if ncomp > 1:
                    arr = arr.reshape(ntuples, ncomp)
                result[name] = arr.astype(arr.dtype.newbyteorder("="))
            remaining.discard(name)

        pos = _skip_array(data, blob_start, ncomp, ntuples, dtype)

    return result


# ---------------------------------------------------------------------------
# CLI — quick inspection / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from collections import Counter

    if len(sys.argv) < 2:
        print("Usage: python -m _futureSim_refactored.sim.vtk_extract <vtk_path> [array_name]")
        sys.exit(1)

    vtk_path = sys.argv[1]

    if len(sys.argv) == 2:
        # List mode
        print(f"{'Array Name':<50s} {'dtype':<16s} {'n_tuples':>10s}")
        print("-" * 78)
        for name, dtype, n in list_arrays(vtk_path):
            print(f"{name:<50s} {dtype:<16s} {n:>10,d}")
    else:
        target = sys.argv[2]
        result = extract_point_array(vtk_path, target)
        if result is None:
            print(f"Array '{target}' not found.")
            sys.exit(1)
        if isinstance(result, list):
            counts = Counter(result)
            print(f"{target}: {len(result):,} strings, {len(counts)} unique")
            for v, c in counts.most_common(15):
                print(f"  [{len(v):>3d}] \"{v}\" x{c:,}")
        else:
            print(f"{target}: shape={result.shape}, dtype={result.dtype}")
            uniq = np.unique(result[~np.isnan(result)]) if np.issubdtype(result.dtype, np.floating) else np.unique(result)
            print(f"  unique: {len(uniq)}, sample: {result[:5]}")
