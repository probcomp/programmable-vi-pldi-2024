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

import abc
import dataclasses
from dataclasses import dataclass
from enum import Enum

import jax
import jax.core as jc
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.address_tree import AddressLeaf
from genjax._src.core.datatypes.address_tree import AddressTree
from genjax._src.core.datatypes.masking import Mask
from genjax._src.core.datatypes.masking import mask
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pretty_printing import CustomPretty
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.utilities import tree_grad_split
from genjax._src.core.pytree.utilities import tree_zipper
from genjax._src.core.transforms.incremental import tree_diff_no_change
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.transforms.incremental import tree_diff_unknown_change
from genjax._src.core.typing import Any
from genjax._src.core.typing import Bool
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import static_check_is_array
from genjax._src.core.typing import typecheck


########################
# Generative datatypes #
########################


#####
# Selection
#####


@dataclasses.dataclass
class Selection(AddressTree):
    @abc.abstractmethod
    def complement(self) -> "Selection":
        """Return a `Selection` which filters addresses to the complement set
        of the provided `Selection`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.strip()
            selection = genjax.select("x")
            complement = selection.complement()
            filtered = chm.filter(complement)
            print(console.render(filtered))
            ```
        """

    def get_selection(self):
        return self

    def get_subtrees_shallow(self):
        raise Exception(
            f"Selection of type {type(self)} does not implement get_subtrees_shallow.",
        )

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        subtree = self.get_subtree(addr)
        return subtree

    ###################
    # Pretty printing #
    ###################

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


############################
# Concrete leaf selections #
############################


@dataclasses.dataclass
class NoneSelection(Selection, AddressLeaf):
    def flatten(self):
        return (), ()

    def complement(self):
        return AllSelection()

    def get_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf value."
        )

    def set_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        tree = tree.add("[bold](None)")
        return tree


@dataclasses.dataclass
class AllSelection(Selection, AddressLeaf):
    def flatten(self):
        return (), ()

    def complement(self):
        return NoneSelection()

    def get_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf value."
        )

    def set_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        tree = tree.add("[bold](All)")
        return tree


##################################
# Concrete structured selections #
##################################


@dataclasses.dataclass
class HierarchicalSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @classmethod
    @dispatch
    def new(cls, *addrs: Any):
        trie = Trie.new()
        for addr in addrs:
            trie[addr] = AllSelection()
        return HierarchicalSelection(trie)

    @classmethod
    @dispatch
    def new(cls, selections: Dict):
        assert isinstance(selections, Dict)
        trie = Trie.new()
        for k, v in selections.items():
            assert isinstance(v, Selection)
            trie[k] = v
        return HierarchicalSelection(trie)

    def complement(self):
        return ComplementHierarchicalSelection(self.trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return NoneSelection()
        else:
            subselect = value.get_selection()
            if isinstance(subselect, Trie):
                return HierarchicalSelection(subselect)
            else:
                return subselect

    def get_subtrees_shallow(self):
        def _inner(v):
            addr = v[0]
            subtree = v[1].get_selection()
            if isinstance(subtree, Trie):
                subtree = HierarchicalSelection(subtree)
            return (addr, subtree)

        return map(
            _inner,
            self.trie.get_subtrees_shallow(),
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        for k, v in self.get_subtrees_shallow():
            subk = tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        return tree


@dataclasses.dataclass
class ComplementHierarchicalSelection(HierarchicalSelection):
    trie: Trie

    def flatten(self):
        return (self.selection,), ()

    def complement(self):
        return HierarchicalSelection(self.trie)

    def has_subtree(self, addr):
        return jnp.logical_not(self.trie.has_subtree(addr))

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return AllSelection()
        else:
            subselect = value.get_selection()
            if isinstance(subselect, Trie):
                return HierarchicalSelection(subselect).complement()
            else:
                return subselect.complement()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Complement)")
        for k, v in self.get_subtrees_shallow():
            subk = sub_tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        tree.add(sub_tree)
        return tree


#####
# ChoiceMap
#####


@dataclasses.dataclass
class ChoiceMap(AddressTree):
    @abc.abstractmethod
    def is_empty(self) -> BoolArray:
        pass

    @abc.abstractmethod
    def merge(
        self,
        other: "ChoiceMap",
    ) -> Tuple["ChoiceMap", "ChoiceMap"]:
        pass

    @dispatch
    def filter(
        self,
        selection: AllSelection,
    ) -> "ChoiceMap":
        return self

    @dispatch
    def filter(
        self,
        selection: NoneSelection,
    ) -> "ChoiceMap":
        return EmptyChoiceMap()

    @dispatch
    def filter(
        self,
        selection: Selection,
    ) -> "ChoiceMap":
        """Filter the addresses in a choice map, returning a new choice map.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.strip()
            selection = genjax.select("x")
            filtered = chm.filter(selection)
            print(console.render(filtered))
            ```
        """
        raise NotImplementedError

    @dispatch
    def replace(
        self,
        selection: AllSelection,
        replacement: "ChoiceMap",
    ) -> "ChoiceMap":
        return replacement

    @dispatch
    def replace(
        self,
        selection: NoneSelection,
        replacement: "ChoiceMap",
    ) -> "ChoiceMap":
        return self

    @dispatch
    def replace(
        self,
        selection: Selection,
        replacement: "ChoiceMap",
    ) -> "ChoiceMap":
        """Replace the submaps selected by `selection` with `replacement`,
        returning a new choice map.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.strip()
            replacement = genjax.choice_map({"z": 5.0})
            selection = genjax.select("x")
            replaced = chm.replace(selection, replacement)
            print(console.render(replaced))
            ```
        """
        raise NotImplementedError

    @dispatch
    def insert(
        self,
        selection: AllSelection,
        extension: "ChoiceMap",
    ) -> "ChoiceMap":
        return extension

    @dispatch
    def insert(
        self,
        selection: NoneSelection,
        extension: "ChoiceMap",
    ) -> "ChoiceMap":
        return self

    @dispatch
    def insert(
        self,
        selection: Selection,
        extension: "ChoiceMap",
    ) -> "ChoiceMap":
        """Extend the submap selected by `selection` with `extension`,
        returning a new choice map.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.strip()
            extension = genjax.choice_map({"z": 5.0})
            selection = genjax.select("x")
            extended = chm.insert(selection, extension)
            print(console.render(extended))
            ```
        """
        raise NotImplementedError

    def get_selection(self) -> "Selection":
        """Convert a `ChoiceMap` to a `Selection`."""
        raise Exception(
            f"`get_selection` is not implemented for choice map of type {type(self)}",
        )

    def safe_merge(self, other: "ChoiceMap") -> "ChoiceMap":
        new, discard = self.merge(other)
        if not discard.is_empty():
            raise Exception(f"Discard is non-empty.\n{discard}")
        return new

    def unsafe_merge(self, other: "ChoiceMap") -> "ChoiceMap":
        new, _ = self.merge(other)
        return new

    def get_choices(self):
        return self

    def strip(self):
        def _check(v):
            return isinstance(v, Trace)

        def _inner(v):
            if isinstance(v, Trace):
                return v.strip()
            else:
                return v

        return jtu.tree_map(_inner, self, is_leaf=_check)

    ###########
    # Dunders #
    ###########

    def __eq__(self, other):
        return self.flatten() == other.flatten()

    # Optional: mutable setter.
    def __setitem__(self, key, value):
        raise Exception(
            f"ChoiceMap of type {type(self)} does not implement __setitem__.",
        )

    def __add__(self, other):
        return self.safe_merge(other)

    ###################
    # Pretty printing #
    ###################

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


#####
# Trace
#####


@dataclasses.dataclass
class Trace(ChoiceMap, AddressTree):
    """> Abstract base class for traces of generative functions.

    A `Trace` is a data structure used to represent sampled executions
    of generative functions.

    Traces track metadata associated with log probabilities of choices,
    as well as other data associated with the invocation of a generative
    function, including the arguments it was invoked with, its return
    value, and the identity of the generative function itself.
    """

    @abc.abstractmethod
    def get_retval(self) -> Any:
        """Returns the return value from the generative function invocation
        which created the `Trace`.

        Examples:

            Here's an example using `genjax.normal` (a distribution). For distributions, the return value is the same as the (only) value in the returned choice map.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            retval = tr.get_retval()
            chm = tr.get_choices()
            v = chm.get_leaf_value()
            print(console.render((retval, v)))
            ```
        """

    @abc.abstractmethod
    def get_score(self) -> FloatArray:
        """Return the score of the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            score = tr.get_score()
            x_score = bernoulli.logpdf(tr["x"], 0.3)
            y_score = bernoulli.logpdf(tr["y"], 0.3)
            print(console.render((score, x_score + y_score)))
            ```
        """

    @abc.abstractmethod
    def get_args(self) -> Tuple:
        pass

    @abc.abstractmethod
    def get_choices(self) -> ChoiceMap:
        """Return a `ChoiceMap` representation of the set of traced random
        choices sampled during the execution of the generative function to
        produce the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.get_choices()
            print(console.render(chm))
            ```
        """

    @abc.abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the generative function whose invocation created the
        `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            gen_fn = tr.get_gen_fn()
            print(console.render(gen_fn))
            ```
        """

    @dispatch
    def project(
        self,
        selection: NoneSelection,
    ) -> FloatArray:
        return 0.0

    @dispatch
    def project(
        self,
        selection: AllSelection,
    ) -> FloatArray:
        return self.get_score()

    @dispatch
    def project(self, selection: "Selection") -> FloatArray:
        """Given a `Selection`, return the total contribution to the score of
        the addresses contained within the `Selection`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            selection = genjax.select("x")
            x_score = tr.project(selection)
            x_score_t = genjax.bernoulli.logpdf(tr["x"], 0.3)
            print(console.render((x_score_t, x_score)))
            ```
        """
        raise NotImplementedError

    @dispatch
    def update(
        self,
        key: PRNGKey,
        choices: ChoiceMap,
        argdiffs: Tuple,
    ):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choices, argdiffs)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        choices: ChoiceMap,
    ):
        gen_fn = self.get_gen_fn()
        args = self.get_args()
        argdiffs = tree_diff_no_change(args)
        return gen_fn.update(key, self, choices, argdiffs)

    def get_aux(self) -> Tuple:
        raise NotImplementedError

    #################################
    # Default choice map interfaces #
    #################################

    def is_empty(self):
        return self.strip().is_empty()

    def filter(
        self,
        selection: Selection,
    ) -> Any:
        stripped = self.strip()
        filtered = stripped.filter(selection)
        return filtered

    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return self.strip().merge(other.strip())

    def has_subtree(self, addr) -> BoolArray:
        choices = self.get_choices()
        return choices.has_subtree(addr)

    def get_subtree(self, addr) -> ChoiceMap:
        choices = self.get_choices()
        return choices.get_subtree(addr)

    def get_subtrees_shallow(self):
        choices = self.get_choices()
        return choices.get_subtrees_shallow()

    def get_selection(self):
        return self.strip().get_selection()

    def strip(self):
        """Remove all `Trace` metadata, and return a choice map.

        `ChoiceMap` instances produced by `tr.get_choices()` will preserve `Trace` instances. `strip` recursively calls `get_choices` to remove `Trace` instances.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            chm = tr.strip()
            print(console.render(chm))
            ```
        """

        def _check(v):
            return isinstance(v, Trace)

        def _inner(v):
            if isinstance(v, Trace):
                return v.strip()
            else:
                return v

        return jtu.tree_map(_inner, self.get_choices(), is_leaf=_check)


#####
# Trace types
#####


@dataclasses.dataclass
class TraceType(AddressTree):
    def on_support(self, other):
        assert isinstance(other, TraceType)
        check = self.__check__(other)
        if check:
            return check, ()
        else:
            return check, (self, other)

    @abc.abstractmethod
    def __check__(self, other):
        pass

    @abc.abstractmethod
    def get_rettype(self):
        pass

    # TODO: think about this.
    # Overload now to play nicely with `Selection`.
    def get_choices(self):
        return self

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        subtree = self.get_subtree(addr)
        return subtree


BaseMeasure = Enum("BaseMeasure", ["Counting", "Lebesgue", "Bottom"])


@dataclasses.dataclass
class AddressLeafTraceType(TraceType, AddressLeaf):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @abc.abstractmethod
    def get_base_measure(self) -> BaseMeasure:
        pass

    @abc.abstractmethod
    def check_subset(self, other) -> Bool:
        pass

    def check_shape(self, other) -> Bool:
        return self.shape == other.shape

    def check_base_measure(self, other) -> Bool:
        m1 = self.get_base_measure()
        m2 = other.get_base_measure()
        return m1 == m2

    def __check__(self, other) -> Bool:
        shape_check = self.check_shape(other)
        measure_check = self.check_base_measure(other)
        subset_check = self.check_subset(other)
        check = (
            (shape_check and measure_check and subset_check)
            or isinstance(other, Bottom)
            or isinstance(self, Bottom)
        )
        return check

    def on_support(self, other):
        assert isinstance(other, TraceType)
        check = self.__check__(other)
        if check:
            return check, ()
        else:
            return check, (self, other)

    def get_leaf_value(self):
        raise Exception("AddressLeafTraceType doesn't keep a leaf value.")

    def set_leaf_value(self):
        raise Exception("AddressLeafTraceType doesn't allow setting a leaf value.")

    def get_rettype(self):
        return self


########################
# Concrete trace types #
########################


@dataclass
class Reals(AddressLeafTraceType, CustomPretty):
    def get_base_measure(self) -> BaseMeasure:
        return BaseMeasure.Lebesgue

    def check_subset(self, other):
        return isinstance(other, Reals) or isinstance(other, Bottom)

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℝ[/b] {self.shape}")
        return tree


@dataclass
class PositiveReals(AddressLeafTraceType, CustomPretty):
    def get_base_measure(self):
        return BaseMeasure.Lebesgue

    def check_subset(self, other):
        return (
            isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℝ⁺[/b] {self.shape}")
        return tree


@dataclass
class RealInterval(AddressLeafTraceType, CustomPretty):
    lower_bound: Any
    upper_bound: Any

    def flatten(self):
        return (), (self.shape, self.lower_bound, self.upper_bound)

    def get_base_measure(self):
        return BaseMeasure.Lebesgue

    # TODO: we need to check if `lower_bound` and `upper_bound`
    # are concrete.
    def check_subset(self, other):
        return (
            isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_free(self, **kwargs):
        tree = rich.tree.Tree(
            f"[b]ℝ[/b] [{self.lower_bound}, {self.upper_bound}]{self.shape}"
        )
        return tree


@dataclass
class Integers(AddressLeafTraceType, CustomPretty):
    def flatten(self):
        return (), (self.shape,)

    def get_base_measure(self):
        return BaseMeasure.Counting

    def check_subset(self, other):
        return (
            isinstance(other, Integers)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℤ[/b] {self.shape}")
        return tree


@dataclass
class Naturals(AddressLeafTraceType, CustomPretty):
    def get_base_measure(self):
        return BaseMeasure.Counting

    def subset_check(self, other):
        return (
            isinstance(other, Naturals)
            or isinstance(other, Integers)
            or isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℕ[/b] {self.shape}")
        return tree


@dataclass
class Finite(AddressLeafTraceType, CustomPretty):
    limit: IntArray

    def get_base_measure(self):
        return BaseMeasure.Counting

    def check_subset(self, other):
        return (
            isinstance(other, Naturals)
            or isinstance(other, Integers)
            or isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]𝔽[/b] [{self.limit}] {self.shape}")
        return tree


@dataclass
class Bottom(AddressLeafTraceType, CustomPretty):
    def __init__(self):
        super().__init__(())

    def get_base_measure(self):
        return BaseMeasure.Bottom

    def check_subset(self, other):
        return True

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree("[b]⊥[/b]")
        return tree


@dataclass
class Empty(AddressLeafTraceType, CustomPretty):
    def __init__(self):
        super().__init__(())

    def check_subset(self, other):
        return True

    # Pretty sure this is wrong - but `Empty` can't occur
    # in distributions return anyways.
    def get_base_measure(self):
        return BaseMeasure.Counting

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree("[b]ϕ[/b] (empty)")
        return tree


# Lift Python values to the trace type lattice.
def tt_lift(v, shape=()):
    if v is None:
        return Empty()
    elif v == jnp.int32:
        return Integers(shape)
    elif v == jnp.float32:
        return Reals(shape)
    elif v == bool:
        return Finite(shape, 2)
    elif static_check_is_array(v):
        return tt_lift(v.dtype, shape=v.shape)
    elif isinstance(v, jax.ShapeDtypeStruct):
        return tt_lift(v.dtype, shape=v.shape)
    elif isinstance(v, jc.ShapedArray):
        return tt_lift(v.dtype, shape=v.shape)


@dataclasses.dataclass
class HierarchicalTraceType(TraceType):
    trie: Trie
    retval_type: TraceType

    def flatten(self):
        return (), (self.trie, self.retval_type)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return Bottom()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()

    def get_rettype(self):
        return self.retval_type

    def on_support(self, other):
        if not isinstance(other, HierarchicalTraceType):
            return False, self
        else:
            check = True
            trie = Trie.new()
            for k, v in self.get_subtrees_shallow():
                if k in other.trie:
                    sub = other.trie[k]
                    subcheck, mismatch = v.on_support(sub)
                    if not subcheck:
                        trie[k] = mismatch
                else:
                    check = False
                    trie[k] = (v, None)

            for k, v in other.get_subtrees_shallow():
                if k not in self.trie:
                    check = False
                    trie[k] = (None, v)
            return check, trie

    def __check__(self, other):
        check, _ = self.on_support(other)
        return check


#####
# Generative function
#####


@dataclasses.dataclass
class GenerativeFunction(Pytree):
    """> Abstract base class for generative functions.

    Generative functions are computational objects which expose convenient interfaces for probabilistic modeling and inference. They consist (often, subsets) of a few ingredients:

    * $p(c, r; x)$: a probability kernel over choice maps ($c$) and untraced randomness ($r$) given arguments ($x$).
    * $q(r; x, c)$: a probability kernel over untraced randomness ($r$) given arguments ($x$) and choice map assignments ($c$).
    * $f(x, c, r)$: a deterministic return value function.
    * $q(u; x, u')$: internal proposal distributions for choice map assignments ($u$) given other assignments ($u'$) and arguments ($x$).

    The interface of methods and associated datatypes which these objects expose is called _the generative function interface_ (GFI). Inference algorithms are written against this interface, providing a layer of abstraction above the implementation.

    Generative functions are allowed to partially implement the interface, with the consequence that partially implemented generative functions may have restricted inference behavior.

    !!! info "Interaction with JAX"

        Concrete implementations of `GenerativeFunction` will likely interact with the JAX tracing machinery if used with the languages exposed by `genjax`. Hence, there are specific implementation requirements which are more stringent than the requirements
        enforced in other Gen implementations (e.g. Gen in Julia).

        * For broad compatibility, the implementation of the interfaces *should* be compatible with JAX tracing.
        * If a user wishes to implement a generative function which is not compatible with JAX tracing, that generative function may invoke other JAX compat generative functions, but likely cannot be invoked inside of JAX compat generative functions.

    Aside from JAX compatibility, an implementor *should* match the interface signatures documented below. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.
    """

    def get_trace_type(self, *args, **kwargs) -> TraceType:
        shape = kwargs.get("shape", ())
        return Bottom(shape)

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        """> Given a `key: PRNGKey` and arguments `x: Tuple`, the generative
        function sample a choice map $c \sim p(\cdot; x)$, as well as any
        untraced randomness $r \sim p(\cdot; x, c)$ to produce a trace $t = (x,
        c, r)$.

        While the types of traces `t` are formally defined by $(x, c, r)$, they will often store additional information - like the _score_ ($s$):

        $$
        s = \log \\frac{p(c, r; x)}{q(r; x, c)}
        $$

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            tr: A trace capturing the data and inference data associated with the generative function invocation.

        Examples:

            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            print(console.render(tr))
            ```

            Here's a slightly more complicated example using the `Builtin` generative function language. You can find more examples on the `Builtin` language page.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                y = genjax.normal(x, 1.0) @ "y"
                return y

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            print(console.render(tr))
            ```
        """
        raise NotImplementedError

    def propose(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple[Any, FloatArray, ChoiceMap]:
        """> Given a `key: PRNGKey` and arguments ($x$), execute the generative
        function, returning a tuple containing the return value from the
        generative function call, the score ($s$) of the choice map assignment,
        and the choice map ($c$).

        The default implementation just calls `simulate`, and then extracts the data from the `Trace` returned by `simulate`. Custom generative functions can overload the implementation for their own uses (e.g. if they don't have an associated `Trace` datatype, but can be used as a proposal).

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            retval: the return value from the generative function invocation
            s: the score ($s$) of the choice map assignment
            chm: the choice map assignment ($c$)

        Examples:

            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            (r, w, chm) = genjax.normal.propose(key, (0.0, 1.0))
            print(console.render(chm))
            ```

            Here's a slightly more complicated example using the `Builtin` generative function language. You can find more examples on the `Builtin` language page.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                y = genjax.normal(x, 1.0) @ "y"
                return y

            key = jax.random.PRNGKey(314159)
            (r, w, chm) = model.propose(key, ())
            print(console.render(chm))
            ```
        """
        tr = self.simulate(key, args)
        chm = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        return (retval, score, chm)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Trace]:
        """> Given a `key: PRNGKey`, a choice map indicating constraints ($u$),
        and arguments ($x$), execute the generative function, and return an
        importance weight estimate of the conditional density evaluated at the
        non-constrained choices, and a trace whose choice map ($c = u' ⧺ u$) is
        consistent with the constraints ($u$), with unconstrained choices
        ($u'$) proposed from an internal proposal.

        Arguments:
            key: A `PRNGKey`.
            chm: A choice map indicating constraints ($u$).
            args: Arguments to the generative function ($x$).

        Returns:
            w: An importance weight.
            tr: A trace capturing the data and inference data associated with the generative function invocation.

        The importance weight `w` is given by:

        $$
        w = \log \\frac{p(u' ⧺ u, r; x)}{q(u'; u, x)q(r; x, t)}
        $$
        """
        raise NotImplementedError

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        constraints: Mask,
        args: Tuple,
    ) -> Tuple[FloatArray, Trace]:
        def _inactive():
            w = 0.0
            tr = self.simulate(key, args)
            return w, tr

        def _active(chm):
            w, tr = self.importance(key, chm, args)
            return w, tr

        return constraints.match(_inactive, _active)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        new_constraints: ChoiceMap,
        diffs: Tuple,
    ) -> Tuple[Any, FloatArray, Trace, ChoiceMap]:
        raise NotImplementedError

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        new_constraints: Mask,
        argdiffs: Tuple,
    ) -> Tuple[Any, FloatArray, Trace, Mask]:
        # The semantics of the merge operation entail that the second returned value
        # is the discarded values after the merge.
        discard_option = prev.strip()
        possible_constraints = new_constraints.unsafe_unmask()
        _, possible_discards = discard_option.merge(possible_constraints)

        def _none():
            (retdiff, w, new_tr, _) = self.update(key, prev, EmptyChoiceMap(), argdiffs)
            if possible_discards.is_empty():
                discard = EmptyChoiceMap()
            else:
                # We return the possible_discards, but denote them as invalid via masking.
                discard = mask(False, possible_discards)
            primal = tree_diff_primal(retdiff)
            retdiff = tree_diff_unknown_change(primal)
            return (retdiff, w, new_tr, discard)

        def _some(chm):
            (retdiff, w, new_tr, _) = self.update(key, prev, chm, argdiffs)
            if possible_discards.is_empty():
                discard = EmptyChoiceMap()
            else:
                # The true_discards should match the Pytree type of possible_discards,
                # but these are valid.
                discard = mask(True, possible_discards)
            primal = tree_diff_primal(retdiff)
            retdiff = tree_diff_unknown_change(primal)
            return (retdiff, w, new_tr, discard)

        return new_constraints.match(_none, _some)

    def assess(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray]:
        """> Given a `key: PRNGKey`, a complete choice map indicating
        constraints ($u$) for all choices, and arguments ($x$), execute the
        generative function, and return the return value of the invocation, and
        the score of the choice map ($s$).

        Arguments:
            key: A `PRNGKey`.
            chm: A complete choice map indicating constraints ($u$) for all choices.
            args: Arguments to the generative function ($x$).

        Returns:
            retval: The return value from the generative function invocation.
            score: The score of the choice map.

        The score ($s$) is given by:

        $$
        s = \log \\frac{p(c, r; x)}{q(r; x, c)}
        $$
        """
        raise NotImplementedError

    def restore_with_aux(
        self,
        interface_data: Tuple,
        aux: Tuple,
    ) -> Trace:
        raise NotImplementedError


@dataclasses.dataclass
class JAXGenerativeFunction(GenerativeFunction, Pytree):
    """A `GenerativeFunction` subclass for JAX compatible generative
    functions."""

    # This is used to support tracing.
    # Below, a default implementation: GenerativeFunctions
    # may customize this to improve compilation time.
    def __abstract_call__(self, *args) -> Any:
        # This should occur only during abstract evaluation,
        # the fact that the value has type PRNGKey is all that matters.
        key = jax.random.PRNGKey(0)
        tr = self.simulate(key, args)
        retval = tr.get_retval()
        return retval

    def unzip(
        self,
        key: PRNGKey,
        fixed: ChoiceMap,
    ) -> Tuple[
        Callable[[ChoiceMap, Tuple], FloatArray],
        Callable[[ChoiceMap, Tuple], Any],
    ]:
        def score(differentiable: Tuple, nondifferentiable: Tuple) -> FloatArray:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (_, score) = self.assess(key, merged, args)
            return score

        def retval(differentiable: Tuple, nondifferentiable: Tuple) -> Any:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (retval, _) = self.assess(key, merged, args)
            return retval

        return score, retval

    # A higher-level gradient API - it relies upon `unzip`,
    # but provides convenient access to first-order gradients.
    @typecheck
    def choice_grad(self, key: PRNGKey, trace: Trace, selection: Selection):
        fixed = trace.strip().filter(selection.complement())
        chm = trace.strip().filter(selection)
        scorer, _ = self.unzip(key, fixed)
        grad, nograd = tree_grad_split(
            (chm, trace.get_args()),
        )
        choice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)
        return choice_gradient_tree


########################
# Concrete choice maps #
########################


@dataclasses.dataclass
class EmptyChoiceMap(ChoiceMap, AddressLeaf):
    def flatten(self):
        return (), ()

    def is_empty(self):
        return True

    def get_subtree(self, addr):
        return self

    def get_leaf_value(self):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def set_leaf_value(self, v):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def get_selection(self):
        return NoneSelection()

    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return other, self

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        tree = rich.tree.Tree("[bold](Empty)")
        return tree


@dataclasses.dataclass
class ValueChoiceMap(ChoiceMap, AddressLeaf):
    value: Any

    def flatten(self):
        return (self.value,), ()

    @classmethod
    def new(cls, v):
        if isinstance(v, ValueChoiceMap):
            return ValueChoiceMap.new(v.get_leaf_value())
        else:
            return ValueChoiceMap(v)

    def is_empty(self):
        return False

    def get_leaf_value(self):
        return self.value

    def set_leaf_value(self, v):
        return ValueChoiceMap(v)

    def get_selection(self):
        return AllSelection()

    @dispatch
    def merge(self, other: "ValueChoiceMap") -> Tuple[ChoiceMap, ChoiceMap]:
        return other, self

    ###########
    # Dunders #
    ###########

    def __hash__(self):
        if isinstance(self.value, np.ndarray):
            return hash(self.value.tobytes())
        else:
            return hash(self.value)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        if isinstance(self.value, Pytree):
            return self.value.__rich_tree__(tree)
        else:
            sub_tree = gpp.tree_pformat(self.value)
            tree.add(sub_tree)
            return tree


@dataclasses.dataclass
class HierarchicalChoiceMap(ChoiceMap):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, constraints: Dict):
        assert isinstance(constraints, Dict)
        trie = Trie.new()
        for k, v in constraints.items():
            v = (
                ValueChoiceMap(v)
                if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
                else v
            )
            trie[k] = v
        return HierarchicalChoiceMap(trie)

    def is_empty(self):
        return self.trie.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        def _inner(k, v):
            sub = selection.get_subtree(k)
            under = v.filter(sub)
            return k, under

        trie = Trie.new()
        iter = self.get_subtrees_shallow()
        for k, v in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        new = HierarchicalChoiceMap(trie)
        if new.is_empty():
            return EmptyChoiceMap()
        else:
            return new

    @dispatch
    def replace(
        self,
        selection: HierarchicalSelection,
        replacement: ChoiceMap,
    ) -> ChoiceMap:
        complement = self.filter(selection.complement())
        return complement.unsafe_merge(replacement)

    @dispatch
    def insert(
        self,
        selection: HierarchicalSelection,
        extension: ChoiceMap,
    ) -> ChoiceMap:
        raise NotImplementedError

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return EmptyChoiceMap()
        else:
            submap = value.get_choices()
            if isinstance(submap, Trie):
                return HierarchicalChoiceMap(submap)
            else:
                return submap

    def get_subtrees_shallow(self):
        def _inner(v):
            addr = v[0]
            subtree = v[1].get_choices()
            if isinstance(subtree, Trie):
                subtree = HierarchicalChoiceMap(subtree)
            return (addr, subtree)

        return map(
            _inner,
            self.trie.get_subtrees_shallow(),
        )

    def get_selection(self):
        trie = Trie.new()
        for k, v in self.get_subtrees_shallow():
            trie[k] = v.get_selection()
        return HierarchicalSelection(trie)

    @dispatch
    def merge(self, other: "HierarchicalChoiceMap"):
        new_inner, discard = self.trie.merge(other.trie)
        return HierarchicalChoiceMap(new_inner), HierarchicalChoiceMap(discard)

    @dispatch
    def merge(self, other: EmptyChoiceMap):
        return self, other

    @dispatch
    def merge(self, other: ValueChoiceMap):
        return other, self

    @dispatch
    def merge(self, other: ChoiceMap):
        raise Exception(
            f"Merging with choice map type {type(other)} not supported.",
        )

    ###########
    # Dunders #
    ###########

    def __setitem__(self, k, v):
        v = (
            ValueChoiceMap(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.trie[k] = v

    def __hash__(self):
        return hash(self.trie)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        for k, v in self.get_subtrees_shallow():
            subk = tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        return tree


##############
# Shorthands #
##############

empty_choice_map = EmptyChoiceMap.new
emp_chm = empty_choice_map
value_choice_map = ValueChoiceMap.new
val_chm = value_choice_map
all_select = AllSelection.new
all_sel = all_select
none_select = NoneSelection.new
none_sel = none_select
choice_map = HierarchicalChoiceMap.new
select = HierarchicalSelection.new
