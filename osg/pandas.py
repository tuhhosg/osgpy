import pandas as pd
from collections.abc import Iterable
from pathlib import Path
from fnmatch import fnmatch
import os

def read_file(fn, *args, **kwargs):
    df = pd.read_csv(fn, *args, **kwargs)
    df.attrs['sources'] = [fn]
    return df

def read_directory(dirname, read=read_file,
                   # fnmatch filters for the filename
                   fn_match=None, fn_not_match=None,
                   # Take parameters from the filename
                   fn_cols=(), fn_prefix='file_',
                   # Arguments for the read function
                   **kwargs):
    dfs = []
    files = []
    for fn in Path(os.path.expanduser(dirname)).iterdir():
        if fn_match is not None and not fnmatch(fn, fn_match):
            continue
        if fn_not_match is not None and fnmatch(fn, fn_not_match):
            continue

        # Read the Data
        df =  read(fn, **kwargs)
        # Remove all spaces in column headers, nobody needs spaces
        df.columns = df.columns.str.strip()

        # Import Columns from Path object
        for col in fn_cols:
            df[fn_prefix + col] = getattr(fn, col)

        dfs.append(df)

    df = pd.concat(dfs)
    df.attrs['sources'] = files
    return df


def missing_rows(data, collapse=True, known_missing=None):
    """Find missing measurements in large datasets.

    For each index column of the data frame `data`, we collect all
    observed values and interpret them as the possible values for this
    dimension. The function returns a dataframe that summarizes which
    values are missing for a full product of the measurements.

    collapse: Boolean or column names to collapse

    known_missing: If you know that some of your results are missing,
          you can give a dataframe that is removed from the result.

    >>> missing_rows(...)
       benchmark    cachesize icache CPs
    0  ghostscript        128      *   *
    1  ghostscript        256      *   *

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
    """Select the rows that are at the given quantile (e.g, min, median, max) of the resepective column.

    df              : DataFrame or Series to inspect
    q               : Which quantile rows to report
    q_labels        : Generate human readable labels instead of q values
    columns         : For which columns are the quantiles searched?
    compress        : If multiple rows are at a given quantile, select the first (random) and report the cardinality.
    q_col           : column name for the selected quantile
    column_col      : column name for the inspected column
    value_col       : column name for the quantile value
    cardinality_col : column name for the cardinality
    index_cols      : Are the index columns included in the result?
    value_cols      : Are the other columns included in the result?


    >>> df
                                     uniform   genetic  original_mse
    benchmark icache cachesize CPs
    adpcm_c   False  2         2    0.499697  0.499801  5.084280e+08
                               3    0.666426  0.666493  5.084280e+08
                               4    0.749660  0.749781  5.084280e+08
                               5    0.799851  0.799958  5.084280e+08
                               6    0.832983  0.833108  5.084280e+08
    ...                                  ...       ...           ...
    typeset   True   64        12   0.946788  0.983514  4.774040e+09
                               13   0.955670  0.985238  4.774040e+09
                               14   0.953097  0.986249  4.774040e+09
                               15   0.956851  0.987022  4.774040e+09
                               16   0.962734  0.987791  4.774040e+09

    >>> select_quantiles(df, q=[0,0.25,0.5,0.75,1.0], columns=['uniform', 'genetic'], value_cols=False)
                  cardinality     value   benchmark  icache  cachesize  CPs
    column  q
    uniform 0.00          130  0.000000  rijndael_e   False         64   11
            0.25            1  0.661176      jpeg_d    True          2    3
            0.50            1  0.828055     bitcnts   False          4   16
            0.75            1  0.898309     adpcm_c    True          8   15
            1.00            1  0.973457       pgp_e   False         64   15
    genetic 0.00            1  0.329892      jpeg_d   False         64    2
            0.25            1  0.856994   basicmath   False          8    9
            0.50            1  0.922826      jpeg_d    True          2   13
            0.75            1  0.969853       qsort   False         64   11
            1.00            1  0.999749     bitcnts    True          8   16
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
    """Returns a function that reorders an column as an ordered category:

       Can be used with plydata:

       df >> do(reorder_by('foo', ['a, 'c', 'b']))"""
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

