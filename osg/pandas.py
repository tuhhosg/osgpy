import pandas as pd

def missing_rows(data, collapse=True):
    """Find missing measurements in large datasets.

    For each index column of the data frame `data`, we collect all
    observed values and interpret them as the possible values for this
    dimension. The function returns a dataframe that summarizes which
    values are missing for a full product of the measurements.

    collaps: Boolean or column names to collapse

    >>> missing_rows(...)
       benchmark    cachesize icache CPs
    0  ghostscript        128      *   *
    1  ghostscript        256      *   *

    """
    dims = data.reset_index()[data.index.names].apply(set)
    full_index = pd.MultiIndex.from_product(dims)
    missing_index = full_index.difference(data.index)
    df = pd.DataFrame(missing_index.to_list(), columns=dims.index)

    # Reorder Columns by fullness
    def is_full(s):
        return len(set(s))/len(set(dims[s.name]))

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

    return pd.DataFrame(ret,columns=names)
