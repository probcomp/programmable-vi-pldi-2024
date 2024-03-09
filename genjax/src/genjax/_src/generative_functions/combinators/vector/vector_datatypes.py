# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import jax.numpy as jnp
import jax.tree_util as jtu
import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.datatypes.generative import select
from genjax._src.core.datatypes.masking import Mask
from genjax._src.core.datatypes.masking import mask
from genjax._src.core.pytree.static_checks import (
    static_check_tree_leaves_have_matching_leading_dim,
)
from genjax._src.core.pytree.utilities import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch


######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.


@dataclass
class IndexSelection(Selection):
    indices: IntArray
    inner: Selection

    def flatten(self):
        return (
            self.indices,
            self.inner,
        ), ()

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, IntArray]):
        idxs = jnp.array(idx)
        return IndexSelection(idxs, AllSelection())

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, List[Int], IntArray], inner: Selection):
        idxs = jnp.array(idx)
        return IndexSelection(idxs, inner)

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, List[Int], IntArray], *inner: Any):
        idxs = jnp.array(idx)
        inner = select(*inner)
        return IndexSelection(idxs, inner)

    @dispatch
    def has_subtree(self, addr: IntArray):
        return jnp.isin(addr, self.indices)

    @dispatch
    def has_subtree(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    def get_subtree(self, addr):
        return self.index_selection.get_subtree(addr)

    def complement(self):
        return ComplementIndexSelection(self)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        sub_tree = rich.tree.Tree(f"[bold](Index,{doc})")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


@dataclass
class ComplementIndexSelection(IndexSelection):
    index_selection: Selection

    def __init__(self, index_selection):
        self.index_selection = index_selection

    def flatten(self):
        return (self.index_selection,), ()

    @dispatch
    def has_subtree(self, addr: IntArray):
        return jnp.logical_not(jnp.isin(addr, self.indices))

    @dispatch
    def has_subtree(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_not(
            jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))
        )

    def get_subtree(self, addr):
        return self.index_selection.get_subtree(addr).complement()

    def complement(self):
        return self.index_selection

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Complement)")
        self.index_selection.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


@dataclass
class IndexChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    def flatten(self):
        return (self.indices, self.inner), ()

    @classmethod
    def convert(cls, chm: ChoiceMap) -> "IndexChoiceMap":
        indices = []
        subtrees = []
        for (k, v) in chm.get_subtrees_shallow():
            if isinstance(k, IntArray):
                indices.append(k)
                subtrees.append(v)
            else:
                raise Exception(
                    f"Failed to convert choice map of type {type(chm)} to IndexChoiceMap."
                )

        inner = tree_stack(subtrees)
        indices = jnp.array(indices)
        return IndexChoiceMap.new(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: IntArray, inner: ChoiceMap) -> ChoiceMap:
        # Promote raw integers (or scalars) to non-null leading dim.
        indices = jnp.array(indices)
        if not indices.shape:
            indices = indices[:, None]

        # Verify that dimensions are consistent before creating an
        # `IndexChoiceMap`.
        _ = static_check_tree_leaves_have_matching_leading_dim((inner, indices))

        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        return IndexChoiceMap(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: List, inner: ChoiceMap) -> ChoiceMap:
        indices = jnp.array(indices)
        return IndexChoiceMap.new(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: Any, inner: Dict) -> ChoiceMap:
        inner = choice_map(inner)
        return IndexChoiceMap.new(indices, inner)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        return IndexChoiceMap(self.indices, self.inner.filter(selection))

    def has_subtree(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    @dispatch
    def filter(
        self,
        selection: IndexSelection,
    ) -> ChoiceMap:
        flags = jnp.isin(selection.indices, self.indices)
        filtered_inner = self.inner.filter(selection.inner)
        masked = mask(flags, filtered_inner)
        return IndexChoiceMap(self.indices, masked)

    def has_subtree(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    @dispatch
    def get_subtree(self, addr: Tuple):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return EmptyChoiceMap()
        (idx, addr) = addr
        subtree = self.inner.get_subtree(addr)
        if isinstance(subtree, EmptyChoiceMap):
            return EmptyChoiceMap()
        else:
            (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
            subtree = jtu.tree_map(lambda v: v[slice_index], subtree)
            return mask(idx in self.indices, subtree)

    @dispatch
    def get_subtree(self, idx: IntArray):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        subtree = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), subtree)

    def get_selection(self):
        return self.inner.get_selection()

    def get_subtrees_shallow(self):
        raise NotImplementedError

    def merge(self, _: ChoiceMap):
        raise Exception("TODO: can't merge IndexChoiceMaps")

    def get_index(self):
        return self.indices

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        sub_tree = rich.tree.Tree(f"[bold](Index,{doc})")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    @dispatch
    def new(
        cls,
        inner: EmptyChoiceMap,
    ) -> EmptyChoiceMap:
        return inner

    @classmethod
    @dispatch
    def new(
        cls,
        inner: ChoiceMap,
    ) -> ChoiceMap:
        # Static assertion: all leaves must have same first dim size.
        static_check_tree_leaves_have_matching_leading_dim(inner)
        return VectorChoiceMap(inner)

    @classmethod
    @dispatch
    def new(
        cls,
        inner: Dict,
    ) -> ChoiceMap:
        chm = choice_map(inner)
        return VectorChoiceMap.new(chm)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        return VectorChoiceMap.new(self.inner.filter(selection))

    @dispatch
    def filter(
        self,
        selection: IndexSelection,
    ) -> Mask:
        filtered = self.inner.filter(selection.inner)
        flags = jnp.logical_and(
            selection.indices >= 0,
            selection.indices
            < static_check_tree_leaves_have_matching_leading_dim(self.inner),
        )

        def _take(v):
            return jnp.take(v, selection.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    @dispatch
    def filter(
        self,
        selection: ComplementIndexSelection,
    ) -> Mask:
        filtered = self.inner.filter(selection.inner.complement())
        flags = jnp.logical_not(
            jnp.logical_and(
                selection.indices >= 0,
                selection.indices
                < static_check_tree_leaves_have_matching_leading_dim(self.inner),
            )
        )

        def _take(v):
            return jnp.take(v, selection.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    def get_selection(self):
        subselection = self.inner.get_selection()
        return subselection

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        return self.inner.get_subtrees_shallow()

    @dispatch
    def merge(self, other: "VectorChoiceMap") -> Tuple[ChoiceMap, ChoiceMap]:
        new, discard = self.inner.merge(other.inner)
        return VectorChoiceMap(new), VectorChoiceMap(discard)

    @dispatch
    def merge(self, other: IndexChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        indices = other.indices

        sliced = jtu.tree_map(lambda v: v[indices], self.inner)
        new, discard = sliced.merge(other.inner)

        def _inner(v1, v2):
            return v1.at[indices].set(v2)

        new = jtu.tree_map(_inner, self.inner, new)

        return VectorChoiceMap(new), IndexChoiceMap(indices, discard)

    @dispatch
    def merge(self, other: EmptyChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return self, other

    ###########
    # Dunders #
    ###########

    def __hash__(self):
        return hash(self.inner)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Vector)")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


#####
# VectorTraceType
#####


@dataclass
class VectorTraceType(TraceType):
    inner: TraceType
    length: int

    def flatten(self):
        return (), (self.inner, self.length)

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return self.inner.get_leaf_value()

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        v = self.inner.get_subtree(addr)
        return VectorTraceType(v, self.length)

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return (k, VectorTraceType(v, self.length))

        return map(lambda args: _inner(*args), self.inner.get_subtrees_shallow())

    def merge(self, _):
        raise Exception("Not implemented.")

    def __subseteq__(self, _):
        return False

    def get_rettype(self):
        return self.inner.get_rettype()


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
index_choice_map = IndexChoiceMap.new
index_select = IndexSelection.new
