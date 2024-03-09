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

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from plum import dispatch

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import typecheck


#####
# Importance sampling
#####


@dataclasses.dataclass
class BootstrapImportanceSampling(Pytree):
    """Bootstrap importance sampling for generative functions."""

    num_particles: IntArray
    model: GenerativeFunction

    def flatten(self):
        return (), (self.num_particles, self.model, self.proposal)

    @typecheck
    @classmethod
    def new(
        cls,
        num_particles: IntArray,
        model: GenerativeFunction,
    ):
        return BootstrapImportanceSampling(num_particles, model)

    def apply(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
    ):
        sub_keys = jax.random.split(self.num_particles)
        (lws, trs) = jax.vmap(self.model.importance, in_axes=(0, None, None))(
            sub_keys,
            observations,
            model_args,
        )
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        return (trs, log_normalized_weights, log_ml_estimate)

    @typecheck
    def __call__(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        return self.apply(key, choice_map, *args)


@dataclasses.dataclass
class CustomProposalImportanceSampling(Pytree):
    """Custom proposal importance sampling for generative functions."""

    num_particles: IntArray
    model: GenerativeFunction
    proposal: GenerativeFunction

    def flatten(self):
        return (), (self.num_particles, self.model, self.proposal)

    @typecheck
    @classmethod
    def new(
        cls,
        num_particles: IntArray,
        model: GenerativeFunction,
        proposal: GenerativeFunction,
    ):
        return CustomProposalImportanceSampling(num_particles, model, proposal)

    def apply(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
        proposal_args: Tuple,
    ):
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        p_trs = jax.vmap(self.proposal.simulate, in_axes=(0, None))(
            sub_keys,
            (observations, *proposal_args),
        )

        def _inner(key, proposal_chm, model_args):
            chm = proposal_chm.safe_merge(observations)
            (w, m_tr) = self.model.importance(
                key,
                chm,
                model_args,
            )
            return (w, m_tr)

        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        (lws, _) = jax.vmap(_inner, in_axes=(0, 0, None))(
            sub_keys, p_trs.strip(), model_args
        )
        lws = lws - p_trs.get_score()
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        return (p_trs, log_normalized_weights, log_ml_estimate)

    @typecheck
    def __call__(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        return self.apply(key, choice_map, *args)


@dataclasses.dataclass
class BootstrapSamplingImportanceResampling(Pytree):
    num_particles: IntArray
    model: GenerativeFunction

    def flatten(self):
        return (), (self.num_particles, self.model)

    @classmethod
    def new(
        cls,
        num_particles: IntArray,
        model: GenerativeFunction,
    ):
        return BootstrapSamplingImportanceResampling(
            num_particles,
            model,
        )

    def apply(
        self,
        key: PRNGKey,
        obs: ChoiceMap,
        model_args: Tuple,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (lws, trs) = jax.vmap(self.model.importance, in_axes=(0, None, None))(
            sub_keys, obs, model_args
        )
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        ind = jax.random.categorical(key, log_normalized_weights)
        tr = jtu.tree_map(lambda v: v[ind], trs)
        lnw = log_normalized_weights[ind]
        return (tr, lnw, log_ml_estimate)

    @typecheck
    def __call__(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        return self.apply(key, choice_map, *args)


@dataclasses.dataclass
class CustomProposalSamplingImportanceResampling(Pytree):
    num_particles: IntArray
    model: GenerativeFunction
    proposal: GenerativeFunction

    def flatten(self):
        return (), (self.num_particles, self.model, self.proposal)

    @classmethod
    def new(
        cls,
        num_particles: IntArray,
        model: GenerativeFunction,
        proposal: GenerativeFunction,
    ):
        return CustomProposalSamplingImportanceResampling(
            num_particles,
            model,
            proposal,
        )

    def apply(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
        proposal_args: Tuple,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        p_trs = jax.vmap(self.proposal.simulate, in_axes=(0, None, None))(
            sub_keys,
            observations,
            proposal_args,
        )

        def _inner(key, proposal):
            constraints = observations.safe_merge(proposal)
            (lws, _) = self.model.importance(key, constraints, model_args)
            return lws

        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(key, self.num_particles)
        lws = jax.vmap(_inner)(sub_keys, p_trs.strip())
        lws = lws - p_trs.get_score()
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        ind = jax.random.categorical(key, log_normalized_weights)
        tr = jtu.tree_map(lambda v: v[ind], p_trs)
        lnw = log_normalized_weights[ind]
        return (tr, lnw, log_ml_estimate)

    @typecheck
    def __call__(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        return self.apply(key, choice_map, *args)


##############
# Shorthands #
##############


@dispatch
def importance_sampling(
    model: GenerativeFunction,
    n_particles: IntArray,
):
    return BootstrapImportanceSampling.new(n_particles, model)


@dispatch
def importance_sampling(
    model: GenerativeFunction,
    proposal: GenerativeFunction,
    n_particles: IntArray,
):
    return CustomProposalImportanceSampling.new(n_particles, model, proposal)


@dispatch
def sampling_importance_resampling(
    model: GenerativeFunction,
    n_particles: IntArray,
):
    return BootstrapSamplingImportanceResampling.new(n_particles, model)


@dispatch
def sampling_importance_resampling(
    model: GenerativeFunction,
    proposal: GenerativeFunction,
    n_particles: IntArray,
):
    return CustomProposalSamplingImportanceResampling.new(n_particles, model, proposal)
