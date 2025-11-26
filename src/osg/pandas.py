import pandas as pd
from collections.abc import Iterable
from pathlib import Path
from fnmatch import fnmatch
import os

def read_file(fn, *args, **kwargs):
    """Read a single CSV file into a pandas.DataFrame and record its source.

    Parameters
    - fn: path-like or str
    - *args, **kwargs: passed to ``pandas.read_csv``

    Returns
    - pandas.DataFrame with a ``.attrs['sources']`` list containing the filename
    """
    df = pd.read_csv(fn, *args, **kwargs)
    df.attrs['sources'] = [fn]
    return df

def read_directory(dirname, read=read_file, fn_match=None, fn_not_match=None,
                   fn_cols=(), fn_prefix='file_', **kwargs):
    """Read all files from a directory and concatenate them into one DataFrame.

    Parameters
    - dirname: directory path to scan
    - read: callable used to read a single file (defaults to :func:`read_file`)
    - fn_match/fn_not_match: optional filename filters (glob style)
    - fn_cols, fn_prefix: attributes from Path objects to inject as columns
    - **kwargs passed to the reader function

    Returns
    - pandas.DataFrame consisting of concatenated frames. The DataFrame will have
      an ``.attrs['sources']`` key with the list of read files.
    """

    dfs = []
    files = []

    for fn in Path(os.path.expanduser(dirname)).rglob("*"):
        if not fn.is_file():
            continue
        if fn_match is not None and not fnmatch(fn.name, fn_match):
            continue
        if fn_not_match is not None and fnmatch(fn.name, fn_not_match):
            continue

        # Read the Data
        df = read(fn, **kwargs)
        # Remove all spaces in column headers, nobody needs spaces
        df.columns = df.columns.str.strip()

        # Import Columns from Path object
        for col in fn_cols:
            df[fn_prefix + col] = getattr(fn, col)

        dfs.append(df)
        files.append(str(fn))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    df.attrs['sources'] = files
    return df


def missing_rows(data, collapse=True, known_missing=None):
    """Return the missing index combinations for a DataFrame with a (possibly) MultiIndex.

    The function computes the cartesian product of all observed values in each index
    level and returns the combinations that are missing in the provided data.

    Parameters
    - data: pandas.DataFrame with a meaningful Index or MultiIndex
    - collapse: bool or list of columns to collapse full-columns into a '*' marker
    - known_missing: optional DataFrame of known-missing entries to subtract

    Returns
    - DataFrame of missing index combinations, or None when nothing is missing
    """

    dims = data.reset_index()[data.index.names].apply(set)
    full_index = pd.MultiIndex.from_product(dims)
    missing_index = full_index.difference(data.index)
    if len(missing_index) == 0:
        return None
    df = pd.DataFrame(missing_index.to_list(), columns=dims.index)

    # Reorder Columns by fullness
    def is_full(s):
        return len(set(s))/len(dims[s.name])

    names = df.apply(is_full).sort_values().index.to_list()

    df = df[df.apply(is_full).sort_values().index.to_list()]

    if not collapse:
        return df

    def repl_full(s):
        if set(s) == dims[s.name]:
            return frozenset({'*'})
        return frozenset(s.to_list())

    for idx in range(len(df.columns)-1, -1,-1):
        col = df.columns[idx]
        X = repl_full
        if type(collapse) is list and col not in collapse:
            X = frozenset
        others = df.columns.to_list()
        others.remove(col)
        df = df.groupby(others).agg(X).reset_index()

    ret = []
    for _, x in df[names].iterrows():
        ret.extend(pd.MultiIndex.from_product(x).to_list())

    df = pd.DataFrame(ret,columns=names)

    # If we have known missing, remove them
    if known_missing is not None:
        df = (pd.merge(df, known_missing, indicator=True, how='outer')
              .query('_merge=="left_only"')
              .drop('_merge', axis=1))
        if len(df) == 0:
            return None
    return df


def select_quantiles(df, q=[0.0, 0.5, 1.0], columns=None, compress=True,
                     q_col='q', q_labels=False, column_col='column',
                     value_col='value', cardinality_col='cardinality',
                     index_cols=True, value_cols=True):
    """Select rows at given quantiles for one or more columns.

    This helper returns representative rows for requested quantiles. It is
    designed to work with DataFrames (or Series) where selecting the row that
    has the quantile value is useful for reporting or plotting.

    Parameters
    - df: DataFrame or Series
    - q: sequence of quantiles in [0,1]
    - columns: which columns to inspect (defaults to all)
    - compress: if True, multiple matching rows are collapsed and cardinality reported
    - q_labels, q_col, column_col, value_col, cardinality_col: naming options
    - index_cols, value_cols: whether to include index/other columns in the result

    Returns
    - DataFrame indexed by (column, q) with requested columns
    """

    if type(df) is pd.Series:
        df = pd.DataFrame(data=df)
        # For a series it makes no sense to also include the values as
        # this is already in the value_col
        value_cols = False

    # Get those rows from df that are at the quantile q in the column col.
    def get_rows(df_sort, col, q):
        idx = int(q * (len(df_sort)-1))
        val = df_sort.iloc[idx][col]
        rows = df_sort[df_sort[col] == val].copy()
        if compress:
            cardinality = len(rows)
            rows = rows.iloc[:1]

        index_cols = rows.reset_index()

        r = pd.DataFrame(data=index_cols)
        if q_labels:
            if   q == 0:   label = 'min'
            elif q == 0.5: label = 'median'
            elif q == 1.0: label = 'max'
            else:          label = f'{int(q*100)}%'
            r[q_col] = label
        else:
            r[q_col]      = q
        r[column_col] = col
        r[value_col]  = val

        if compress:
            r[cardinality_col] = cardinality
        return r

    # Get the rows for every given quantile
    ret = []
    columns = columns or df.columns
    for col in columns:
        df_sort = df[df[col].notna()].sort_values(col)
        for _q in q:
            ret += [get_rows(df_sort, col, _q)]

    # Concatenate the individual rows
    ret = pd.concat(ret).set_index([column_col, q_col])

    # Select those columns, the user requested
    ordered = [value_col]
    if index_cols:
        if  isinstance(index_cols, Iterable):
            ordered += list(index_cols)
        else:
            ordered += list([n for n in df.index.names if n])

    if value_cols:
        if  isinstance(value_cols, Iterable):
            ordered += list(value_cols)
        else:
            ordered += list(df.columns)
    if compress:
        ordered = [cardinality_col] + ordered

    return ret[ordered]


def reorder_by(column, categories, dropna=True):
    """Return a transformation that casts ``column`` to an ordered categorical.

    Useful for data pipelines (e.g., with ``plydata``). The returned function
    takes a DataFrame and returns a copy where ``column`` is converted to an
    ordered ``Categorical`` with categories in the provided order.
    """
    def helper(df):
        df = df.copy()
        df[column] = df[column].astype('category').cat.set_categories(categories, ordered=True)
        if dropna:
            df = df[df[column].notna()]
        return df
    return helper


def mapvalues(column, keys=None, values=None, na_action=None, keep_order=True, **kwargs):
    """Returns a function that:
       1. Takes an DataFrame
       2. Selects the given column
       3. Maps the column-contents according to key->value mapping (or kwargs dict)
       4. Returns a pandas.Series

    Can be used with plydata:

    >>> df >> define(xyz_label=mapvalues('xyz', x='System 1', y='System 2'))
    """
    translate_dict = None
    if keys is not None:
        assert values is not None and len(keys) == len(values)
        translate_dict = dict(zip(keys, values))
    else:
        assert kwargs is not None
        translate_dict = kwargs

    def mapper(df):
        series = df[column].map(translate_dict, na_action=na_action)
        if values is not None and keep_order:
            series = series.astype('category').cat.set_categories(values, ordered=True)
        return series
    return mapper

