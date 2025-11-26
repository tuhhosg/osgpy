import os
import pandas as pd
import pytest
import sys
print(sys.path)
from osg.dataref import PgfKeyDict, DatarefDict


# ------------------------------------------------------------
# 1. Parsing existing file
# ------------------------------------------------------------

def test_parse_basic(tmp_path):
    fn = tmp_path / "data.tex"
    fn.write_text(
        "\\pgfkeyssetvalue{/versuchung/a}{1}\n"
        "\\pgfkeyssetvalue{/versuchung/b}{hello}\n"
    )

    d = PgfKeyDict(filename=str(fn))
    assert d["a"] == "1"
    assert d["b"] == "hello"
    assert len(d) == 2


# ------------------------------------------------------------
# 2. Formatting + flush()
# ------------------------------------------------------------

def test_flush_roundtrip(tmp_path):
    fn = tmp_path / "out.tex"
    fn.write_text("")  # start empty

    d = PgfKeyDict(filename=str(fn))
    d["x"] = "42"
    d["y/z"] = "A"

    d.flush()

    text = fn.read_text()
    assert "\\pgfkeyssetvalue{/versuchung/x}{42}" in text
    assert "\\pgfkeyssetvalue{/versuchung/y/z}{A}" in text

    # re-import
    d2 = PgfKeyDict(filename=str(fn))
    assert d2["x"] == "42"
    assert d2["y/z"] == "A"


# ------------------------------------------------------------
# 3. PrefixForPgfKeyDict
# ------------------------------------------------------------

def test_prefix_view(tmp_path):
    fn = tmp_path / "data.tex"
    fn.write_text("")

    d = PgfKeyDict(filename=str(fn))
    p = d.prefixed_with("foo/")

    p["a"] = "1"
    p["b"] = "2"

    assert d["foo/a"] == "1"
    assert d["foo/b"] == "2"

    del p["a"]
    assert "foo/a" not in d


# ------------------------------------------------------------
# 4. pandas-Series import
# ------------------------------------------------------------

def test_pandas_series(tmp_path):
    fn = tmp_path / "data.tex"
    fn.write_text("")

    d = PgfKeyDict(filename=str(fn))

    s = pd.Series({"count": 2.0, "mean": 1.25})
    d.pandas(s, prefix="speedup")

    assert d["speedup/count"] == 2.0
    assert d["speedup/mean"] == 1.25


# ------------------------------------------------------------
# 5. pandas-DataFrame import incl. names=[]
# ------------------------------------------------------------

def test_pandas_dataframe(tmp_path):
    fn = tmp_path / "data.tex"
    fn.write_text("")

    df = pd.DataFrame(
        {"th": [1, 4], "speedup": [1.0, 1.5]},
    ).set_index("th")

    d = PgfKeyDict(filename=str(fn))
    d.pandas(df, names=["th"])

    # keys become th=1/speedup => 1.0
    assert d["th=1/speedup"] == 1.0
    assert d["th=4/speedup"] == 1.5


# ------------------------------------------------------------
# 6. DatarefDict uses "drefset"
# ------------------------------------------------------------

def test_datarefdict_format(tmp_path):
    fn = tmp_path / "data.tex"
    fn.write_text("")

    d = DatarefDict(filename=str(fn), key="/foo")
    d["a"] = "9"
    d.flush()

    text = fn.read_text()
    # now the formatter uses \drefset{/foo/a}{9}
    assert "\\drefset{/foo/a}{9}" in text
