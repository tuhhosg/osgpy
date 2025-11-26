import re

# THIS IS A MODIFIED CLONE of versuchung.tex.{PgfKey,Dataref}Dict

class PgfKeyDict(dict):
    """
    PgfKeyDict creates one PgfKey definition per stored key

    >>> from osg.dataref import PgfKeyDict
    >>> pgf = PgfKeyDict("/tmp/test.tex")
    >>> pgf["abcd"] = 23
    >>> pgf.flush()  # flush method of File
    >>> print(open("/tmp/test.tex").read())
    \\pgfkeyssetvalue{/versuchung/abcd}{23}

    In the TeX source you can do something like::

      \\usepackage{tikz}
      \\pgfkeys{/pgf/number format/.cd,fixed,precision=1}
      [...]
      \\newcommand{\\versuchung}[1]{\\pgfkeysvalueof{/versuchung/#1}}
      \\versuchung{abcd}

    """

    def __init__(self, filename = "data.tex", pgfkey = "/versuchung", setmacro="pgfkeyssetvalue"):
        dict.__init__(self)

        self.__pgfkey = pgfkey
        self.format_string = "\\" + setmacro + "{%s/%s}{%s}"
        self.filename = filename

        with open(self.filename) as fd:
            self._parse(fd.read())

    def _parse(self, value):
        regex = self.format_string %(self.__pgfkey,"([^{}]*)", "([^{}]*)")
        regex = regex.replace('\\', r'\\')
        for line in value.split("\n"):
            m = re.search(regex, line)
            if m:
                self[m.groups()[0]] = m.groups()[1]
        return self

    def _format(self):
        v = []
        last_base_key = None
        for key in sorted(self):
            value = self[key]
            if "/" in key:
                base_key = key[:key.rindex("/")]
            else:
                base_key = None
            if last_base_key and last_base_key != base_key:
                v.append("")
            last_base_key = base_key
            v.append(self.format_string % (self.__pgfkey, key, value))

        return "\n".join(v) + "\n"

    def flush(self):
        value = self._format()
        with open(self.filename, "w+") as fd:
            fd.write(value)

    class PrefixForPgfKeyDict:
        def __init__(self, prefix, d):
            self.prefix = prefix
            self.d = d
        def __getitem__(self, key):
            return self.d[self.prefix + key]
        def __setitem__(self, key, value):
            self.d[self.prefix + key] = value
        def __delitem__(self, key):
            del self.d[self.prefix + key]
        def prefixed_with(self, prefix):
            return self.d.PrefixForPgfKeyDict(self.prefix + prefix, self.d)

    def prefixed_with(self, prefix):
        return self.PrefixForPgfKeyDict(prefix, self)

    def pandas(self, df, prefix="", names=None, verbose=False):
        """Import pandas.DataFrame or pandas.Series as keys

        This functions imports all cells from a DataFrame or Series as
        individual keys. The key is generated from the index and the
        columns (lines first, columns second). MultiIndex is supported
        for both dimension. For example, for the following DataFrame

           >>> import pandas as pd
           >>> df = pd.DataFrame([[1,1,1], [4, 1.5, 4]], columns=['th', 'speedup', 'load'])
           >>> df = df.set_index('th')
           >>> df
               speedup  load
           th
           1       1.0     1
           4       1.5     4
           >>> from versuchung.tex import PgfKeyDict
           >>> pgf = PgfKeyDict("/tmp/test.tex")
           >>> pgf.pandas(df, names=['th'], verbose=True)
           th=1/speedup => 1.0
           th=1/load => 1.0
           th=4/speedup => 1.5
           th=4/load => 4.0

        With the ``names`` parameter, you can control whether a level
        should be prefixed with the index name. In the above example,
        ``th=`` is the result of the names parameter. A useful pattern
        is to ``describe()`` a column:

          >>> pgf.pandas(df.speedup.describe(), prefix="speedup", verbose=True)
          speedup/count => 2.0
          speedup/mean => 1.25
          speedup/std => 0.3535533905932738
          speedup/min => 1.0
          speedup/25 percent => 1.125
          speedup/50 percent => 1.25
          speedup/75 percent => 1.375
          speedup/max => 1.5
        """
        import pandas as pd

        if prefix: prefix += "/"
        else:      prefix = ""

        def wrap_list(seq):
            if hasattr(seq, "__iter__") and not type(seq) is str:
                return seq
            return [seq]

        def zip_extend(a, b):
            a_len, b_len = len(a), len(b)
            ret_len = max(a_len, b_len)
            return zip(list(a) + [None] * (ret_len - a_len), list(b) + [None] * (ret_len - b_len))

        def fmt(name, key):
            ret = []
            for k, v in zip_extend(wrap_list(name), wrap_list(key)):
                # print(names, k, hasattr(names, "__contains__") and k in names)
                if k is not None and names == True or (hasattr(names, "__contains__") and k in names):
                    ret.append(f"{k}={v}")
                else:
                    ret.append(str(v))
            return "/".join(ret)

        if isinstance(df, pd.DataFrame):
            for index, row in df.iterrows():
                pp = fmt(df.index.name or df.index.names, index)
                self.pandas(row, prefix=prefix+pp, names=names, verbose=verbose)
        elif isinstance(df, pd.Series):
            for key in df.index:
                pp = fmt(df.index.name or df.index.names, key)
                dref_key = (prefix + pp).replace("%", " percent")
                if verbose: print(f"{dref_key} => {df.loc[key]}")
                self[dref_key] = df.loc[key]
        else:
            raise ValueError("Please supply a pandas.DataFrame or pandas.Series")


class DatarefDict(PgfKeyDict):
    """

    DatarefDict is like :class:`~osgpy.dataref.PgfKeyDict`, but generates keys for dataref <https://ctan.org/pkg/dataref>.

    In LaTeX, you can reference dataref keys or calculate with them::

        \\dref{/base}
        \\drefrel[percent of=/base]{/derived}

    """

    def __init__(self, filename = "data.tex", key = ""):
        PgfKeyDict.__init__(self, filename, key, "drefset")

if __name__ == '__main__':
    import sys
    print(PgfKeyDict(sys.argv[1]))
