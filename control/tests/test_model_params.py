"""model/params.yaml の整合性テスト。

寸法や関節定義がずれると、sim と実機が一致しなくなる。
機械が確認できることは機械に確認させる。
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PARAMS_PATH = Path(__file__).resolve().parents[2] / "model" / "params.yaml"


@pytest.fixture(scope="module")
def params():
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_params_file_exists():
    assert PARAMS_PATH.exists(), "model/params.yaml が見つかりません"


def test_joint_count_matches_declaration(params):
    """joints の要素数と robot.num_joints が一致すること。"""
    assert len(params["joints"]) == params["robot"]["num_joints"]


def test_joint_ids_are_sequential(params):
    """ID が 0 から連番であること（配列の添字として使うため）。"""
    ids = [j["id"] for j in params["joints"]]
    assert ids == list(range(len(ids)))


def test_joint_names_are_unique(params):
    names = [j["name"] for j in params["joints"]]
    assert len(names) == len(set(names))


@pytest.mark.xfail(
    reason="Phase 0 完了まで未確定。埋め終わったら xfail を外すこと",
    strict=False,
)
def test_no_unfilled_values(params):
    """TODO（null）が残っていないこと。Phase 0 の完了条件。"""
    assert params["meta"]["status"] == "fixed"
    assert params["robot"]["total_mass"] is not None
    for joint in params["joints"]:
        assert joint["limit_lower"] is not None, f"{joint['name']} の可動域が未設定"
