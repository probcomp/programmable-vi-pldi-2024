#!/usr/bin/env python
# coding: utf-8


import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# the only pyro dependency
from numpyro.infer import SVI, RenyiELBO, TraceGraph_ELBO
from optax import sgd

import genjax

key = jax.random.PRNGKey(314159)
console = genjax.pretty()
label_fontsize = 70  # Set the desired font size here


# ## Model


def model(data):
    x = numpyro.sample("x", dist.Normal(0.0, 10.0))
    y = numpyro.sample("y", dist.Normal(0.0, 10.0))
    rs = x**2 + y**2
    z = numpyro.sample("z", dist.Normal(rs, 0.1 + (rs / 100.0)), obs=data)
    return (x, y, z)


# ## Naive variational guide


# Now, we define our variational proposal.
def guide(data):
    μ1 = numpyro.param("μ1", 0.0)
    μ2 = numpyro.param("μ2", 0.0)
    log_σ1 = numpyro.param("log_sigma1", 0.0)
    log_σ2 = numpyro.param("log_sigma2", 0.0)
    x = numpyro.sample("x", dist.Normal(μ1, jnp.exp(log_σ1)))
    y = numpyro.sample("y", dist.Normal(μ2, jnp.exp(log_σ2)))


# ## Training


svi = SVI(model, guide, sgd(1e-3), loss=TraceGraph_ELBO(num_particles=64))
key, sub_key = jax.random.split(key)
svi_result = svi.run(sub_key, 20000, 5.0)

print("TraceGraph ELBO:")
print((svi_result.losses[1000:].mean(), svi_result.losses[1000:].std()))


# ## Training (IWAE)


svi = SVI(model, guide, sgd(1e-3), loss=RenyiELBO(num_particles=5))
key, sub_key = jax.random.split(key)
svi_result = svi.run(sub_key, 20000, 5.0)

print("RenyiELBO(k = 5):")
print((svi_result.losses[1000:].mean(), svi_result.losses[1000:].std()))
