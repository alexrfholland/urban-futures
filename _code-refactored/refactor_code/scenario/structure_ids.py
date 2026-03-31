from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


_INT31_MAX = np.iinfo(np.int32).max


def _normalize_part(value: object) -> str:
    if value is None or pd.isna(value):
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "none"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6f}"
    return str(value).strip()


def _stable_int31(parts: Iterable[object], used_ids: set[int]) -> int:
    base = "::".join(_normalize_part(part) for part in parts)
    salt = 0
    while True:
        digest = hashlib.sha256(f"{base}::{salt}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) & _INT31_MAX
        if value == 0:
            salt += 1
            continue
        if value not in used_ids:
            used_ids.add(value)
            return value
        salt += 1


def _coerce_structure_id(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def collect_structure_ids(*dfs: pd.DataFrame | None) -> set[int]:
    used_ids: set[int] = set()
    for df in dfs:
        if df is None or df.empty or "structureID" not in df.columns:
            continue
        for value in df["structureID"].tolist():
            numeric = _coerce_structure_id(value)
            if numeric is not None:
                used_ids.add(numeric)
    return used_ids


def assign_tree_structure_ids(
    df: pd.DataFrame | None,
    *,
    site: str,
    scenario: str | None,
    used_ids: set[int] | None = None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    df = df.copy()
    if "structureID" not in df.columns:
        df["structureID"] = np.nan

    used_ids = set() if used_ids is None else used_ids

    for index, row in df.iterrows():
        current = _coerce_structure_id(row.get("structureID"))
        if current is not None:
            used_ids.add(current)
            df.at[index, "structureID"] = current
            continue

        if bool(row.get("isNewTree", False)):
            structure_id = _stable_int31(
                (
                    site,
                    scenario or "baseline",
                    "tree",
                    "recruit",
                    row.get("recruit_intervention_type"),
                    row.get("recruit_year"),
                    row.get("recruit_source_id"),
                    row.get("voxel_index"),
                    row.get("x"),
                    row.get("y"),
                    row.get("z"),
                ),
                used_ids,
            )
        else:
            structure_id = _stable_int31(
                (
                    site,
                    "tree",
                    "initial",
                    row.get("NodeID"),
                    row.get("tree_number"),
                    row.get("debugNodeID"),
                    row.get("x"),
                    row.get("y"),
                    row.get("z"),
                ),
                used_ids,
            )

        df.at[index, "structureID"] = structure_id

    df["structureID"] = pd.to_numeric(df["structureID"], errors="coerce").astype("Int64")
    return df


def assign_baseline_tree_structure_ids(
    df: pd.DataFrame | None,
    *,
    site: str,
    used_ids: set[int] | None = None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    df = df.copy()
    if "structureID" not in df.columns:
        df["structureID"] = np.nan

    used_ids = set() if used_ids is None else used_ids

    for index, row in df.iterrows():
        current = _coerce_structure_id(row.get("structureID"))
        if current is not None:
            used_ids.add(current)
            df.at[index, "structureID"] = current
            continue

        structure_id = _stable_int31(
            (
                site,
                "baseline-tree",
                row.get("tree_number"),
                row.get("NodeID"),
                row.get("size"),
                row.get("control"),
                row.get("precolonial"),
                row.get("x"),
                row.get("y"),
                row.get("z"),
            ),
            used_ids,
        )
        df.at[index, "structureID"] = structure_id

    df["structureID"] = pd.to_numeric(df["structureID"], errors="coerce").astype("Int64")
    return df


def assign_log_structure_ids(
    df: pd.DataFrame | None,
    *,
    site: str,
    used_ids: set[int] | None = None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    df = df.copy()
    if "structureID" not in df.columns:
        df["structureID"] = np.nan

    used_ids = set() if used_ids is None else used_ids

    for index, row in df.iterrows():
        current = _coerce_structure_id(row.get("structureID"))
        if current is not None:
            used_ids.add(current)
            df.at[index, "structureID"] = current
            continue

        structure_id = _stable_int31(
            (
                site,
                "log",
                row.get("logNo"),
                row.get("NodeID"),
                row.get("debugNodeID"),
                row.get("voxelID"),
                row.get("x"),
                row.get("y"),
                row.get("z"),
            ),
            used_ids,
        )
        df.at[index, "structureID"] = structure_id

    df["structureID"] = pd.to_numeric(df["structureID"], errors="coerce").astype("Int64")
    return df


def assign_pole_structure_ids(
    df: pd.DataFrame | None,
    *,
    site: str,
    used_ids: set[int] | None = None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    df = df.copy()
    if "structureID" not in df.columns:
        df["structureID"] = np.nan

    used_ids = set() if used_ids is None else used_ids

    for index, row in df.iterrows():
        current = _coerce_structure_id(row.get("structureID"))
        if current is not None:
            used_ids.add(current)
            df.at[index, "structureID"] = current
            continue

        structure_id = _stable_int31(
            (
                site,
                "pole",
                row.get("pole_number"),
                row.get("NodeID"),
                row.get("debugNodeID"),
                row.get("voxel_index"),
                row.get("x"),
                row.get("y"),
                row.get("z"),
            ),
            used_ids,
        )
        df.at[index, "structureID"] = structure_id

    df["structureID"] = pd.to_numeric(df["structureID"], errors="coerce").astype("Int64")
    return df


def replacement_structure_ids(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    site: str,
    scenario: str,
    absolute_year: int,
    used_ids: set[int] | None = None,
) -> list[int]:
    used_ids = set() if used_ids is None else used_ids

    replacement_ids: list[int] = []
    for _, row in df.loc[mask].iterrows():
        replacement_ids.append(
            _stable_int31(
                (
                    site,
                    scenario,
                    "tree",
                    "replace",
                    absolute_year,
                    row.get("structureID"),
                    row.get("NodeID"),
                    row.get("tree_number"),
                    row.get("x"),
                    row.get("y"),
                    row.get("z"),
                ),
                used_ids,
            )
        )
    return replacement_ids
