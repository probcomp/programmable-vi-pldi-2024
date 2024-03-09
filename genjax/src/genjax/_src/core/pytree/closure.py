# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from jax import api_util

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Tuple


@dataclass
class DynamicClosure(Pytree):
    fn: Callable
    dyn_args: Tuple

    def flatten(self):
        return (self.dyn_args,), (self.fn,)

    @classmethod
    def new(cls, callable, *dyn_args):
        if isinstance(callable, DynamicClosure):
            return DynamicClosure(callable.fn, (*callable.dyn_args, *dyn_args))
        else:
            return DynamicClosure(callable, dyn_args)

    def __call__(self, *args):
        return self.fn(*self.dyn_args, *args)

    def __hash__(self):
        avals = tuple(api_util.shaped_abstractify(i) for i in self.dyn_args)
        return hash((self.fn, *avals))


def dynamic_closure(*args):
    return lambda fn: DynamicClosure.new(fn, *args)
