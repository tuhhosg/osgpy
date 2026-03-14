import re
import matplotlib.colors as mcolors
import pandas as pd

srared    = (208/255., 38/255., 38/255.)
safegreen = "#31a354"

def mix_color(a, scale=1.0, b='white'):
    """Mixes two colors in RGB-space: `a * scale + (1-scale) * b`"""
    a   = mcolors.to_rgb(a)
    b   = mcolors.to_rgb(b)
    ret = tuple([a[i] * scale + (1-scale) * b[i] for i in range(0, 3)])
    return mcolors.to_hex(ret)


def latex_color(spec):
    """Converts LaTeX-like color mixes ("green!50") to an RGB

    >>> L("red!20")
    '#ffcccc'
    """

    assert '!' in spec, "There must be at least one ! in the spec"
    a, rest = spec.split("!", 1)
    while rest:
        m = re.match("([0-9]{1,3})(?:!([^!]+))?(?:!(.*))?$", rest)
        scale, b, rest = m.groups()
        a = mix_color(a, float(scale) / 100.0, b or "white")
    return a

def aes_direct(data=None, **kwargs):
    """
    Facilitates the definition of a standalone plotnine layer using direct coordinate values.

    This helper streamlines the inclusion of annotations and specific geometric 
    elements (e.g., highlighting regions with geom_rect) by converting raw 
    coordinates into a localized dataset. By establishing identity mappings 
    and disabling aesthetic inheritance, it allows for the precise placement of 
    elements without requiring an external DataFrame or affecting global 
    aesthetic scales.

    Parameters
    ----------
    data : dict, optional
        A collection of values to be treated as a single data row.
    **kwargs
        Named arguments representing aesthetic dimensions (e.g., xmin, ymax) 
        and their corresponding values.

    Returns
    -------
    dict
        A configuration dictionary intended for keyword unpacking (**) into 
        a geom layer, containing 'inherit_aes', 'mapping', and 'data'.

    Example
    -------
    >>> # Highlight a specific region without modifying the primary dataset
    >>> ggplot(df, aes(x='x', y='y')) + \
    ...     geom_rect(**aes_direct(xmin=5, xmax=10, ymin=0, ymax=20), 
    ...               fill='red', alpha=0.2)
    """
    df = []
    if data is not None and type(data) is dict:
        df += [data]
    if kwargs:
        df += [kwargs]
    df = pd.DataFrame(data=df)
    return dict(
        inherit_aes=False,
        mapping=aes(**{str(k):str(k) for k in df.columns}),
        data=df
    )
