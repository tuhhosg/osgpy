import pytest
from osg.plot import latex_color, mix_color


def test_mix_color_basic():
    # mixing a color with white should lighten it
    light = mix_color("#ff0000", 0.5, "#ffffff")
    assert isinstance(light, str)
    assert light.startswith("#")


def test_latex_color_red20():
    # The documented example: red!20 -> light red
    c = latex_color("red!20")
    assert c.lower() == "#ffcccc"
