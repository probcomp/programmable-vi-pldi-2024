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

import copy
from dataclasses import dataclass

import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.address_tree import AddressTree
from genjax._src.core.datatypes.hashable_dict import HashableDict
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.pretty_printing import CustomPretty


#####
# Trie
#####


@dataclass
class Trie(AddressTree, CustomPretty):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls):
        return Trie(hashable_dict())

    def is_empty(self):
        return not bool(self.inner)

    def get_selection(self):
        raise Exception("Trie doesn't provide conversion to Selection.")

    # Returns a new `Trie` with shallow copied inner dictionary.
    def trie_insert(self, addr, value):
        copied_inner = copy.copy(self.inner)
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in copied_inner:
                subtree = Trie(hashable_dict())
            else:
                subtree = copied_inner[first]
            new_subtree = subtree.trie_insert(rest, value)
            copied_inner[first] = new_subtree
            return Trie(copied_inner)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            copied_inner[addr] = value
            return Trie(copied_inner)

    def has_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.has_subtree(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.inner

    def get_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.get_subtree(rest)
            else:
                return None
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            if addr not in self.inner:
                return None
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    # This is provided for compatibility with the ChoiceMap interface.
    def get_choices(self):
        return self

    # This is provided for compatibility with the Selection interface.
    def get_selection(self):
        return self

    def merge(self, other: "Trie"):
        new = hashable_dict()
        discard = hashable_dict()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other.get_subtree(k)
                new[k], discard[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                new[k] = v
        return Trie(new), Trie(discard)

    ###########
    # Dunders #
    ###########

    def __setitem__(self, k, v):
        new_trie = self.trie_insert(k, v)
        self.inner = new_trie.inner

    def __getitem__(self, k):
        return self.get_subtree(k)

    def __contains__(self, k):
        return self.has_subtree(k)

    def __hash__(self):
        return hash(self.inner)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        for (k, v) in self.get_subtrees_shallow():
            subk = tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        return tree

    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        for (k, v) in self.inner.items():
            subk = tree.add(f"[bold]:{k}")
            subtree = gpp._pformat(v, **kwargs)
            subk.add(subtree)
        return tree


##############
# Shorthands #
##############

trie = Trie.new
