from pathlib import Path
import pandas as pd
from osg.flamegraph import sample_tree


def test_from_counts_and_to_table():
    fn = Path(__file__).parent.parent / 'data' / 'perf-example.counts'
    # ensure file exists for the test environment
    assert fn.exists()

    tree = sample_tree.from_counts(fn)
    assert tree.count > 0

    tbl = tree.to_table()
    assert isinstance(tbl, pd.DataFrame)
    assert 'name' in tbl.columns
