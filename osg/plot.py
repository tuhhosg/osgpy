import re
import matplotlib.colors as mcolors

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
