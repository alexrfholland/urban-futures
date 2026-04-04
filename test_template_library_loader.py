from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from final import a_resource_distributor_dataframes as loader


def _make_template_df(
    *,
    precolonial: bool = False,
    size: str = "fallen",
    control: str = "improved-tree",
    tree_id: int = 1,
    x_offset: float = 0.0,
) -> pd.DataFrame:
    template = pd.DataFrame(
        {
            "x": [x_offset],
            "y": [0.0],
            "z": [0.0],
            "resource_fallen log": [1],
            "stat_fallen log": [1],
            "resource": ["fallen log"],
        }
    )
    return pd.DataFrame(
        [
            {
                "precolonial": precolonial,
                "size": size,
                "control": control,
                "tree_id": tree_id,
                "template": template,
            }
        ]
    )


class TemplateLibraryLoaderTests(unittest.TestCase):
    def test_load_full_template_table_prefers_overrides_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "variant" / "trees"
            template_root.mkdir(parents=True)

            expected = _make_template_df(tree_id=7, x_offset=7.0)
            expected.to_pickle(template_root / "template-library.overrides-applied.pkl")

            loaded, path = loader._load_full_template_table(template_root)

            self.assertEqual(path.name, "template-library.overrides-applied.pkl")
            self.assertEqual(int(loaded.iloc[0]["tree_id"]), 7)
            self.assertEqual(float(loaded.iloc[0]["template"].iloc[0]["x"]), 7.0)

    def test_load_full_template_table_reads_base_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "base" / "trees"
            template_root.mkdir(parents=True)

            expected = _make_template_df(tree_id=3, x_offset=3.0)
            expected.to_pickle(template_root / "template-library.base.pkl")

            loaded, path = loader._load_full_template_table(template_root)

            self.assertEqual(path.name, "template-library.base.pkl")
            self.assertEqual(int(loaded.iloc[0]["tree_id"]), 3)
            self.assertEqual(float(loaded.iloc[0]["template"].iloc[0]["x"]), 3.0)

    def test_load_full_template_table_applies_selected_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_root = tmpdir_path / "tree_libraries" / "base" / "trees"
            base_root.mkdir(parents=True)
            variant_root = tmpdir_path / "variant" / "trees"
            variant_root.mkdir(parents=True)

            base = pd.concat(
                [
                    _make_template_df(tree_id=1, x_offset=1.0),
                    _make_template_df(tree_id=2, x_offset=2.0),
                ],
                ignore_index=True,
            )
            overrides = _make_template_df(tree_id=2, x_offset=22.0)

            base.to_pickle(base_root / "template-library.base.pkl")
            overrides.to_pickle(variant_root / "template-library.selected-overrides.pkl")

            original = os.environ.get("TREE_TEMPLATE_BASE_ROOT")
            os.environ["TREE_TEMPLATE_BASE_ROOT"] = str(base_root)
            try:
                loaded, path = loader._load_full_template_table(variant_root)
            finally:
                if original is None:
                    os.environ.pop("TREE_TEMPLATE_BASE_ROOT", None)
                else:
                    os.environ["TREE_TEMPLATE_BASE_ROOT"] = original

            self.assertEqual(path.name, "template-library.selected-overrides.pkl")
            self.assertEqual(sorted(loaded["tree_id"].tolist()), [1, 2])
            loaded = loaded.sort_values("tree_id").reset_index(drop=True)
            self.assertEqual(int(loaded.iloc[1]["tree_id"]), 2)
            self.assertEqual(float(loaded.iloc[1]["template"].iloc[0]["x"]), 22.0)

    def test_template_root_uses_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "variant" / "trees"
            original = os.environ.get("TREE_TEMPLATE_ROOT")
            os.environ["TREE_TEMPLATE_ROOT"] = str(template_root)
            try:
                resolved = loader._template_root()
            finally:
                if original is None:
                    os.environ.pop("TREE_TEMPLATE_ROOT", None)
                else:
                    os.environ["TREE_TEMPLATE_ROOT"] = original

            self.assertEqual(resolved, template_root)


if __name__ == "__main__":
    unittest.main()
