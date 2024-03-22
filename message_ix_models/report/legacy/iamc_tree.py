import re

import pandas as pd

from . import DF_IDX

_col = "Variable"


class Node(object):
    def __init__(
        self,
        name: str,
        parent=None,
        transparent: bool = False,
        df: pd.DataFrame = None,
    ):
        self.parent = parent
        self.name = name
        self.transparent = transparent
        if df is not None:
            df = df.copy()  # gets rid of warning
            if isinstance(df.index, pd.MultiIndex):
                df.reset_index(inplace=True)
            # update because root can be named
            df.loc[:, _col] = self.full_name
            df.set_index(DF_IDX, inplace=True)
        self.df = df
        self.children: list = []
        self._child_by_name: dict = {}

    def __repr__(self):
        return self.full_name

    @property
    def full_name(self):
        if self.parent is None or self.parent.name == "":
            return self.name
        ret = "|".join([self.parent.full_name, self.name])
        return ret

    def node_list(self, fullname=True):
        node_list = []
        for child in self.children:
            node_list += child.node_list(fullname=fullname)
        if fullname:
            node_list += [self.full_name]
        else:
            node_list += [self]
        return node_list

    def add_node(self, namelist, *args, **kwargs):
        if len(namelist) == 0:
            return  # end recursion

        name = namelist[0]
        atbottom = len(namelist) == 1
        exists = name in self._child_by_name
        if atbottom and not exists:
            child = Node(self, name, *args, **kwargs)
            self.add_child(child)
        elif not atbottom:
            if not exists:
                self.add_child(Node(self, name))
            self._child_by_name[name].add_node(namelist[1:], *args, **kwargs)
        elif atbottom and exists:
            msg = "Cannot insert node {}, already exists in children of {}."
            raise ValueError(msg.format(name, self.full_name))

    def rm_child(self, node):
        name = node.name
        node.parent = None
        delattr(self, name)
        del self._child_by_name[name]
        self.children.remove(node)

    def add_child(self, node):
        name = node.name
        node.parent = self
        self.children.append(node)
        self._child_by_name[name] = node
        setattr(self, name, node)

    def sum_sectors(self, mode="overwrite"):
        if len(self.children) == 0:
            return self.df  # bottom of recursion

        if self.df is not None and mode not in ["overwrite", "add"]:
            return self.df  # check writeability

        # children must have a df
        try:
            dfs = [c.sum_sectors(mode=mode) for c in self.children]
            df = pd.concat(dfs, copy=True, sort=True)
        except ValueError as e:
            raise ValueError(e.message + ": No dataframes found in children")

        grp_idx = [x for x in DF_IDX if x != _col]
        df = df.groupby(level=grp_idx).sum()
        df[_col] = self.full_name
        df = df.set_index(_col, append=True).reorder_levels(DF_IDX)

        if mode == "add" and self.df is not None:
            # combine new and old dfs
            assert not self.df.isnull().values.any()
            assert not df.isnull().values.any()
            df = self.df + df
            assert not df.isnull().values.any()

        self.df = df

        return self.df

    def concat_df(self):
        return pd.concat([self.df] + [c.concat_df() for c in self.children], sort=True)


class Tree(object):
    def __init__(self, root: str):
        self.root = Node(parent=None, name=root)

    def add_node(self, namelist, *args, **kwargs):
        if isinstance(namelist, str):
            namelist = namelist.split("|")
        self.root.add_node(namelist, *args, **kwargs)

    def node_list(self, *args, **kwargs):
        return self.root.node_list(*args, **kwargs)

    def find_node(self, fullname):
        namelist = fullname.split("|")
        node = self.root
        for name in namelist:
            node = node._child_by_name[name]
        return node

    def full_df(self, mode="overwrite"):
        self.root.sum_sectors(mode=mode)
        df = self.root.concat_df()
        df.reset_index(inplace=True)
        df = df[df[_col] != ""]  # hack to get rid of empty sectors
        df.set_index(DF_IDX, inplace=True)
        return df.sort_index()


def replace(x: str, base_name: str) -> str:
    return re.sub("^" + base_name, "", x)


def sum_iamc_sectors(
    df: pd.DataFrame, root: str, mode: str = "overwrite", cleanname: bool = True
):
    """Given a dataframe in the internal schema (df_idx), construct a full
    IAMC variable representation

    Parameters
    ----------
    df: sector data frame
    root: root name
    cleanname: remove root name from sector names
    """
    multi_idx = isinstance(df.index, pd.MultiIndex)

    if multi_idx:
        df.reset_index(inplace=True)

    # explicity remove total columns, these will be recalculated
    df = df[df[_col] != ""]

    sectors = sorted(df[_col].unique())
    t = Tree(root=root)
    for sector in sectors:
        t.add_node(sector, df=df[df[_col] == sector])
    df = t.full_df(mode=mode)
    if cleanname:
        df.reset_index(inplace=True)
        df[_col] = df[_col].apply(replace, args=(t.root.name,)).str.lstrip("|")
        df.set_index(DF_IDX, inplace=True)

    return df
