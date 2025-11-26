import pandas as pd
import os
from pathlib import Path
from osg.pandas import read_file, read_directory, select_quantiles


def test_read_file_sets_sources(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x,y\n1,2\n")

    df = read_file(str(p))
    assert "sources" in df.attrs
    assert str(p) in df.attrs["sources"]


def test_read_directory_concatenate(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.csv").write_text("x,y\n1,2\n")
    (d / "b.csv").write_text("x,y\n3,4\n")

    df = read_directory(str(d))
    # two rows from each file -> total 4 rows
    assert len(df) == 2
    # sources should be present
    assert "sources" in df.attrs


def test_select_quantiles_simple():
    df = pd.DataFrame({"val": [1, 2, 3]})
    res = select_quantiles(df, q=[0.0, 0.5, 1.0], columns=["val"], compress=True)
    # index names are (column, q)
    assert ("val", 0.5) in res.index
    # median value should be 2
    assert res.loc[("val", 0.5), "value"] == 2
