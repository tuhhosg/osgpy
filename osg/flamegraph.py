import sys
import pandas as pd
from plotnine import geom_rect, geom_text, aes
import plotnine as p9
from fnmatch import fnmatch
from plydata import *
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.backends.backend_agg import get_hinting_flag
from collections import OrderedDict
from pathlib import Path
import numpy as np

class sample_tree:
    def __init__(self, other=None):
        if other is not None:
            self.count = other.count
            self.attrs = other.attrs.copy()
            self.children = other.children.copy()
        else:
            self.count = 0
            self.attrs = {}
            self.children = OrderedDict()

    def dump(self, fd=sys.stderr, cur_depth=0):
        """Dump tree structure to fd"""
        fd.write(f" count={self.count} attrs={self.attrs}\n")
        for name, child in self.children.items():
            fd.write(" " * (cur_depth*2+2) + f"{name}")
            child.dump(fd, cur_depth + 1)

    def size(self):
        """Recursively calculate the number of nodes in this sample tree."""
        return 1 + sum([child.size() for child in self.children.values()])

    def __repr__(self):
        return f"<sample_tree count:{self.count}  attrs={self.attrs} children:{list(self.children.keys())}>"


    @staticmethod
    def from_counts(fn : Path) -> "sample_tree":
        """Reads path-based represenation as produced by the stackcollapse-*.pl
           scripts and transforms them to a tree-based representation."""
        tree = sample_tree()
        with open(fn) as fd:
            for line in fd.readlines():
                path, count = line.rsplit(" ", 1)
                count = int(count)
                ptr = tree
                for p in path.split(";"):
                    if p not in ptr.children:
                        n = sample_tree()
                        ptr.children[p] = n
                    ptr.count += count
                    ptr = ptr.children[p]
                ptr.count += count
        return tree

    def prune(self, cutoff_percent=0.1, cutoff_height=None, pattern=None):
        """Cutoff leafs and subtrees to shrink the sample tree.

           cutoff_percent: remove subtrees that are within less than X percent of all samples.

           cutoff_height: remove subtrees below a certain tree height.

           cutoff_match: remove subtrees whose root matches a certain predicate (pattern or function)
        """
    
        if cutoff_percent is not None:
            cutoff_value = self.count * cutoff_percent/100.0
    
        def recr(old, height):
            new = sample_tree(old)
            for name, child in old.children.items():
                if (cutoff_percent is not None and child.count < cutoff_value) \
                   or (cutoff_height is not None and height > cutoff_height) \
                   or (callable(pattern) and pattern(name, child))\
                   or (pattern is not None and fnmatch(name, pattern)):
                    # Delete the child
                    del new.children[name]
                else:
                    new.children[name] = recr(child, height+1)
            return new
        return recr(self,0)

    def collapse(self, pattern=None):
        """Remove inner nodes that match a certain pattern. The children of removed nodes
           are lifted into their respective parent. If the parent already has a child with
           that name, merge the subtrees."""
    
        new_children = []
        new = sample_tree(self)
        new.children.clear()

        queue = list(self.children.items())
        while queue:
            name, child = queue.pop()
            keep = True
            if callable(pattern) and pattern(name, child):
                keep = False
            if pattern is not None and fnmatch(name, pattern):
                keep = False
    
            if keep:
                if name in new.children:
                    new.children[name].count += child.count
                else:
                    new.children[name] = child
            else:
                queue.extend(child.children.items())
    
        for name, child in new.children.items():
            new.children[name] = child.collapse(pattern)
    
        return new

    def transform(self, func):
        """Copies the tree and calls a visitor function on every tree node. The visitor function takes a name and an sample tree object"""
        def recr(name, node):
            N = sample_tree(node)
            func(name, N)
            for name, child in node.children.items():
                N.children[name] = recr(name, child)
            return N
        return recr("<root>", self)

    def attr_set(self, initial=None, **kwargs):
        """Set attributes according to a given list of patterns. The pattern list is
           given as kwargs with the key being the attribute name and the value being
           a 2- or 3- tuple: (pattern, value, inherit?). For example:

        tree = tree.attr_set(label=[('asm_exc_page_fault', 'pagefault'),
                                    ('nvme_read', 'NVME', False)])

        All nodes below asm_exc_page_fault are labled with "pagefault".
        The node nvme_read is labeled "NVME".
        """
        def recr(name, child, state):
            ret = sample_tree(child)
            ret.attrs.update(state)

            for attr_name, patterns in kwargs.items():
                for pattern in patterns:
                    if fnmatch(name, pattern[0]):
                        ret.attrs[attr_name] = pattern[1]
                        # Give to children
                        if len(pattern) <= 2 or pattern[2]:
                            state[attr_name] = pattern[1]

            for name, child in ret.children.items():
                ret.children[name] = recr(name, child, state.copy())
            return ret
        if initial is None:
            initial = dict()
        return recr('<root>', self, initial)


    def to_table(self):
        """Convert the sample tree to a pandas DataFrame to be consumed with plotnine.

        This function performs the layouting of the flame graph. A simple flamegraph can be produced with:

        >>> (ggplot(tree.to_table(), aes(xmin='xmin', xmax='xmax', ymin='ymin',ymax='ymax',
                                     fill='label', label='name'))
              + geom_rect()
              + geom_text_truncate(show_legend=False))
        """
        total_count = self.count
        data = []
        def recr(name, node, height, x_min, x_max):
            data.append(dict(
                name=name,
                name_hash=name_hash(name),
                count=node.count,
                rect_width=node.count/total_count,
                rect_start=x_min,
                level=height,
                **node.attrs
            ))
            x = x_max
            for name, child in reversed(node.children.items()):
                percent = child.count / total_count
                child_x_max = x
                child_x_min = x - percent
                x = child_x_min
                recr(name, child, height+1, child_x_min, child_x_max)
        recr("<root>", self, -1, 0.0, 1.0)

        return pd.DataFrame(data=data)

    @staticmethod
    def aes(**kwargs):
        return aes(**kwargs).inherit(dict(xmin='rect_start', xmax='rect_start+rect_width', ymin='level',ymax='level+1', label='name'))
    
        

def name_hash(name):
    """The namehash algorithm from flamegraph.pl"""
    vector = 0
    weight = 1
    max = 1
    mod = 10
    for c in name:
        i = ord(c) % mod
        vector += (i / (mod - 1)) * weight
        mod += 1
        max += weight
        weight *= 0.7
        if mod > 12:
            break;

        return (1 - vector / max)


_params = geom_text.DEFAULT_PARAMS.copy()
_params.update(
    dict(
        font_scale=0.8,
        font_max_size=11,
        padding_left=0.005,
        padding_right=0.005,
    )
)


class geom_text_truncate(geom_text):
    """An derived geom that scales and clips a text to fit a certain geom_rect."""
    REQUIRED_AES = {"label", "xmin", "xmax", "ymin", "ymax"}
    DEFAULT_PARAMS = _params

    def draw_panel(self, data, panel_params, coord, ax, **params):
        bb = ax.get_window_extent()

        A, B = panel_params.x.limits
        total_width = B - A
        df = data.copy()
        padding_left = params.get("padding_left")
        padding_right = params.get("padding_right")
        df['x'] = data['xmin'] + padding_left  * total_width
        df['y'] = (data['ymax']  + data['ymin'])/ 2
        df['ha'] = 'left'
        df['va'] = 'center'

        A, B = panel_params.y.limits
        total_height = (B  - A)
        rect_height = (data.ymax - data.ymin)
        df['size'] = (bb.height * (rect_height / total_height) / 1.33 / data.lineheight) * params.get("font_scale")
        df['size'] = df['size'].clip(upper=params.get("font_max_size"))

        df['label_percent'] = ((df.xmax - df.xmin)/total_width) - padding_left - padding_right
        df['label_percent'] = df.label_percent.clip(lower=0)
        df['label_width']   = df.label_percent * bb.width

        dpi = p9.options.get_option('dpi')
        family = p9.options.get_option('base_family')
        props = FontProperties(family=family)
        font = get_font(findfont(props))
        df['label_trunc'] = None
        for idx, row in df.iterrows():
            font.set_size(row['size'], dpi)
            def test(s):
                font.set_text(s, 0, flags=get_hinting_flag())
                text_width, _ = font.get_width_height()
                # print(row.label_width, row.label_percent, text_width, s)
                if row.label_width >= text_width/64:
                    df.at[idx, 'label_trunc'] = s
                    return True
            if not test(row['label']):
                i = len(row['label'])
                while i >= 3 and not test(row['label'][:i] + '..'):
                    i -= 1
        df['label'] = df['label_trunc']
        df = df[~df.label.isna()]

        geom_text.draw_group(df, panel_params, coord, ax, **params)


if __name__ == "__main__":
    import sys
    from plotnine import *
    from pathlib import Path

    fn = Path(__file__).parent.parent / 'data'/ 'perf-example.counts'
    tree = sample_tree.from_counts(fn)
    print("Tree Nodes before pruning:", tree.size())
    tree = tree.prune(cutoff_percent=0.1, cutoff_height=8)
    print("Tree Nodes after pruning:", tree.size())
    tree = tree.attr_set(
        component=[
            ('*geos*', 'GEOS'),
            ('*GEOS*', 'GEOS')
        ],
        initial=dict(component='other')
    )
    table = tree.to_table()
    
    flamegraph = (
        ggplot(table, tree.aes(fill='component'))
        + geom_rect(color='white',size=0.1)
        + geom_text_truncate(show_legend=False)
        + scale_fill_brewer('qual', palette=2)
        + scale_y_continuous(breaks=[])
        + theme(
            figure_size=(6,3),
        )
    )

    flamegraph.save('test.pdf')
