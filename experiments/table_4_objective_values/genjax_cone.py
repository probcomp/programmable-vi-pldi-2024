#!/usr/bin/env python
# coding: utf-8


import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

import genjax
from genjax import vi

console = genjax.pretty()
key = jax.random.PRNGKey(314159)
sns.set_theme(style="white")
label_fontsize = 70  # Set the desired font size here


# ## Model


@genjax.gen
def model():
    x = vi.normal_reparam(0.0, 10.0) @ "x"
    y = vi.normal_reparam(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    z = vi.normal_reparam(rs, 0.1 + (rs / 100.0)) @ "z"


# ## Naive variational guide


# Now, we define our variational proposal.
@genjax.gen
def variational_family(data, ϕ):
    μ1, μ2, log_σ1, log_σ2 = ϕ
    x = vi.normal_reparam(μ1, jnp.exp(log_σ1)) @ "x"
    y = vi.normal_reparam(μ2, jnp.exp(log_σ2)) @ "y"


data = genjax.choice_map({"z": 5.0})
objective = vi.elbo(model, variational_family, data)


# Training.
key = jax.random.PRNGKey(314159)
ϕ = (0.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 20000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, (_, ϕ_grads)) = jitted(sub_keys, ((), (data, ϕ)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    losses.append(jnp.mean(loss))


# ### Mean loss after convergence


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 5000)
loss, (_, (_, ϕ_grads)) = jitted(sub_keys, ((), (data, ϕ)))
print("ELBO:")
print((jnp.mean(loss), jnp.var(loss)))

# ## Training with IWAE

# $$\text{IWAE ELBO} = E_{\{z_k\}_{k = 0}^N \sim Q}[\log \frac{1}{N}\sum_i w_i(z_i)]$$
#
# where $w_i(z_i) = \frac{P(z_i, x)}{Q(z_i)}$

# ## 5 particle IWAE


iwae_objective = vi.iwae_elbo(model, variational_family, data, 5)


# Training with IWAE.
key = jax.random.PRNGKey(314159)
ϕ = (3.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(iwae_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 20000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 1)
    (
        loss,
        (
            _,
            (
                _,
                ϕ_grads,
            ),
        ),
    ) = jitted(sub_keys, ((), (data, ϕ)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    losses.append(jnp.mean(loss))

key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 5000)
loss, (_, (_, _)) = jitted(sub_keys, ((), (data, ϕ)))
print("IWAE(K = 5):")
print((jnp.mean(loss), jnp.var(loss)))
