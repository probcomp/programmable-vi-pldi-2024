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
"""This module contains a transformation infrastructure based on custom
primitives, interpreters with stateful contexts and custom primitive handling
lookups."""

import copy
import dataclasses
import functools
import itertools as it
from contextlib import contextmanager

import jax.core as jc
import jax.tree_util as jtu
from jax import api_util
from jax import tree_util
from jax import util as jax_util
from jax._src import core as jax_core
from jax.extend import linear_util as lu
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

from adevjax.hashable_dict import HashableDict
from adevjax.hashable_dict import hashable_dict
from adevjax.pytree import Pytree
from adevjax.staging import stage
from adevjax.typing import List
from adevjax.typing import Union
from adevjax.typing import Value


#########################
# Custom JAX primitives #
#########################

safe_map = jax_core.safe_map


def batch_fun(fun: lu.WrappedFun, in_dims):
    fun, out_dims = batching.batch_subtrace(fun)
    return _batch_fun(fun, in_dims), out_dims


@lu.transformation
def _batch_fun(in_dims, *in_vals, **params):
    with jax_core.new_main(
        batching.BatchTrace, axis_name=jax_core.no_axis_name
    ) as main:
        out_vals = (
            yield (
                main,
                in_dims,
            )
            + in_vals,
            params,
        )
        del main
    yield out_vals


class FlatPrimitive(jax_core.Primitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super(FlatPrimitive, self).__init__(name)
        self.multiple_results = True

        def _abstract(*flat_avals, **params):
            return pe.abstract_eval_fun(self.impl, *flat_avals, **params)

        self.def_abstract_eval(_abstract)

        def _jvp(primals, tangents, **params):
            primals_out, tangents_out = ad.jvp(
                lu.wrap_init(self.impl, params)
            ).call_wrapped(primals, tangents)
            tangents_out = jax_util.safe_map(
                ad.recast_to_float0, primals_out, tangents_out
            )
            return primals_out, tangents_out

        ad.primitive_jvps[self] = _jvp

        def _batch(args, dims, **params):
            batched, out_dims = batch_fun(lu.wrap_init(self.impl, params), dims)
            return batched.call_wrapped(*args), out_dims()

        batching.primitive_batchers[self] = _batch

        def _mlir(c, *mlir_args, **params):
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(c, *mlir_args, **params)

        mlir.register_lowering(self, _mlir)


class InitialStylePrimitive(FlatPrimitive):
    """Contains default implementations of transformations."""

    def __init__(self, name, batch_semantics=None):
        super().__init__(name)

        def fun_impl(*args, **params):
            impl = params["impl"]
            return impl(*args, **params)

        self.def_impl(fun_impl)

        if batch_semantics is None:

            def _batch(args, dims, **params):
                batched, out_dims = batch_fun(lu.wrap_init(self.impl, params), dims)
                return batched.call_wrapped(*args), out_dims()

            batching.primitive_batchers[self] = _batch

        else:
            batching.primitive_batchers[self] = functools.partial(batch_semantics, self)

    def subcall(self, name):
        return InitialStylePrimitive(f"{self.name}/{name}")


def initial_style_bind(prim, **params):
    """Binds a primitive to a function call."""

    def bind(f):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to a call primitive."""
            jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)

            def _impl(*args, **params):
                consts, args = jax_util.split_list(args, [params["num_consts"]])
                return jax_core.eval_jaxpr(jaxpr.jaxpr, consts, *args)

            outs = prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                impl=_impl,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(jaxpr.literals),
                **params,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind


def initial_style_flat_bind_with_params(prim, **params):
    """Binds a primitive to a function call."""

    def bind(f):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to a call primitive."""
            jaxpr, (flat_args, _, _) = stage(f)(*args, **kwargs)
            outs = prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                jaxpr=jaxpr.jaxpr,
                **params,
            )
            return outs

        return wrapped

    return bind


###################
# CPS interpreter #
###################

VarOrLiteral = Union[jc.Var, jc.Literal]


@dataclasses.dataclass
class Environment(Pytree):
    """Keeps track of variables and their values during propagation."""

    env: HashableDict[jc.Var, Value]

    def flatten(self):
        return (self.env,), ()

    @classmethod
    def new(cls):
        return Environment(hashable_dict())

    def read(self, var: VarOrLiteral) -> Value:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Value) -> Value:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.read(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Value:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var in self.env

    def copy(self):
        return copy.copy(self)


@dataclasses.dataclass
class CPSInterpreter(Pytree):
    def flatten(self):
        return (), ()

    # This produces an instance of `Interpreter`
    # as a context manager - to allow us to control error stack traces,
    # if required.
    @classmethod
    @contextmanager
    def new(cls):
        try:
            yield CPSInterpreter()
        except Exception as e:
            raise e

    def _eval_jaxpr_cps(
        self,
        jaxpr: jc.Jaxpr,
        consts: List[Value],
        args: List[Value],
        allowlist: List[jc.Primitive],
    ):
        env = Environment.new()
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)

        def eval_jaxpr_iterate(eqns, env, invars, args):
            jax_util.safe_map(env.write, invars, args)

            for (eqn_idx, eqn) in list(enumerate(eqns)):
                in_vals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals

                if eqn.primitive in allowlist:

                    # Create continuation.
                    def kont(eqns, eqn_idx, outvars, env, *args):
                        return eval_jaxpr_iterate(
                            eqns[eqn_idx + 1 :], env, outvars, [*args]
                        )

                    def _binder(env, tree_args):
                        flat_args = jtu.tree_leaves(tree_args)
                        return eqn.primitive.impl(*flat_args, **params)

                    in_tree = params["in_tree"]
                    tree_args = jtu.tree_unflatten(in_tree, args)

                    # Bind the continuation as a static parameter into an invocation
                    # of a primitive that "wants" a continuation.
                    outvals = initial_style_bind(
                        eqn.primitive,
                        kont=functools.partial(kont, eqns, eqn_idx, eqn.outvars),
                    )(_binder)(env.copy(), tree_args)

                # Otherwise, fall through -- we just use the default bind.
                else:
                    outvals = eqn.primitive.bind(*args, **params)

                jax_util.safe_map(
                    env.write,
                    eqn.outvars,
                    jtu.tree_leaves(outvals),
                )

            return jax_util.safe_map(env.read, jaxpr.outvars)

        return eval_jaxpr_iterate(jaxpr.eqns, env, jaxpr.invars, args)

    def run_interpreter(self, kont, allowlist, fn, *args, **kwargs):
        def _inner(*args, **kwargs):
            return kont(fn(*args, **kwargs))

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_cps(jaxpr, consts, flat_args, allowlist)
        if flat_out:
            return jtu.tree_unflatten(out_tree(), flat_out)


Cont = CPSInterpreter


def cps(f, kont, allowlist):
    # Runs the interpreter.
    def _run_interpreter(*args):
        with Cont.new() as interpreter:
            return interpreter.run_interpreter(kont, allowlist, f, *args)

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(*args):
        fun = lu.wrap_init(_run_interpreter)
        flat_args, args_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, args_tree)
        retvals = flat_fun.call_wrapped(*flat_args)
        out_tree_def = out_tree()
        return jtu.tree_unflatten(out_tree_def, retvals)

    return wrapped
