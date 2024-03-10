#!/usr/bin/env python
# coding: utf-8


import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import genjax
from genjax import vi

console = genjax.pretty()
key = jax.random.PRNGKey(314159)
label_fontsize = 70  # Set the desired font size here


@genjax.gen
def model():
    x = vi.normal_reparam(0.0, 10.0) @ "x"
    y = vi.normal_reparam(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    z = vi.normal_reparam(rs, 0.1 + (rs / 100.0)) @ "z"


# A more expressive variational family.
@genjax.gen
def expressive_variational_family(data, ϕ):
    z = data["z"]
    u = vi.uniform() @ "u"
    θ = 2 * jnp.pi * u
    log_σ1, log_σ2 = ϕ
    x = vi.normal_reparam(jnp.sqrt(z) * jnp.cos(θ), jnp.exp(log_σ1)) @ "x"
    y = vi.normal_reparam(jnp.sqrt(z) * jnp.sin(θ), jnp.exp(log_σ2)) @ "y"


marginal_q = vi.marginal(
    genjax.select("x", "y"), expressive_variational_family, lambda: vi.sir(1)
)

data = genjax.choice_map({"z": 5.0})
hvi_objective = vi.elbo(model, marginal_q, data)


# Training.
key = jax.random.PRNGKey(314159)
ϕ = (0.0, 0.0)
jitted = jax.jit(jax.vmap(hvi_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, ((ϕ_grads,), ())) = jitted(sub_keys, ((), ((ϕ,), ())))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    if i % 1000 == 0:
        print(jnp.mean(loss))
    losses.append(jnp.mean(loss))
print(ϕ)


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 5000)
loss, (_, ((ϕ_grads,), ())) = jitted(sub_keys, ((), ((ϕ,), ())))
print("HVI-ELBO(N = 1):")
print((jnp.mean(loss), jnp.var(loss)))


# ### HVI with SIR


marginal_q = vi.marginal(
    genjax.select("x", "y"), expressive_variational_family, lambda: vi.sir(5)
)

data = genjax.choice_map({"z": 5.0})
hvi_objective = vi.iwae_elbo(model, marginal_q, data, 1)

# Training with IWAE.
key = jax.random.PRNGKey(314159)
ϕ = (0.0, 0.0)
jitted = jax.jit(jax.vmap(hvi_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, ((ϕ_grads,), ())) = jitted(sub_keys, ((), ((ϕ,), ())))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    losses.append(jnp.mean(loss))

key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 5000)
loss, (_, ((ϕ_grads,), ())) = jitted(sub_keys, ((), ((ϕ,), ())))
print("HVIWAE(N = 5, K = 1):")
print((jnp.mean(loss), jnp.var(loss)))


marginal_q = vi.marginal(
    genjax.select("x", "y"), expressive_variational_family, lambda: vi.sir(5)
)

data = genjax.choice_map({"z": 5.0})
hvi_objective = vi.iwae_elbo(model, marginal_q, data, 5)

# Training with IWAE.
key = jax.random.PRNGKey(314159)
ϕ = (0.0, 0.0)
jitted = jax.jit(jax.vmap(hvi_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, ((ϕ_grads,), ())) = jitted(sub_keys, ((), ((ϕ,), ())))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    losses.append(jnp.mean(loss))

key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 5000)
loss, (_, ((ϕ_grads,), ())) = jitted(sub_keys, ((), ((ϕ,), ())))
print("HVIWAE(N = 5, K = 5):")
print((jnp.mean(loss), jnp.var(loss)))
