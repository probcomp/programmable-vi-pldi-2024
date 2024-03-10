#!/usr/bin/env python
# coding: utf-8


import logging
import os

import jax
import matplotlib.pyplot as plt
import pyro

# the only pyro dependency
import pyro.distributions as dist
import torch
from matplotlib.gridspec import GridSpec
from optax import adam

import genjax

key = jax.random.PRNGKey(314159)
console = genjax.pretty()
smoke_test = "CI" in os.environ
assert pyro.__version__.startswith("1.8.6")

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Set matplotlib settings
plt.style.use("default")
label_fontsize = 70  # Set the desired font size here


ϕ = (0.0, 0.0, 1.0, 1.0)


# ## Model


def model():
    x = pyro.sample("x", dist.Normal(0.0, 10.0))
    y = pyro.sample("y", dist.Normal(0.0, 10.0))
    rs = x**2 + y**2
    z = pyro.sample("z", dist.Normal(rs, 0.1 + (rs / 100.0)))
    return (x, y, z)


with pyro.plate("samples", 2000, dim=-1):
    samples = model()
x, y, z = samples


# ## Naive variational guide


softplus = torch.nn.Softplus()


# Now, we define our variational proposal.
def guide():
    μ1 = pyro.param("mu1", torch.tensor(0.0))
    μ2 = pyro.param("mu2", torch.tensor(0.0))
    log_σ1 = pyro.param("log_sigma1", torch.tensor(1.0))
    log_σ2 = pyro.param("log_sigma2", torch.tensor(1.0))
    x = pyro.sample("x", dist.Normal(μ1, torch.exp(log_σ1)))
    y = pyro.sample("y", dist.Normal(μ2, torch.exp(log_σ2)))


# ## Training


pyro.clear_param_store()
data = torch.tensor(5.0)

# These should be reset each training loop.
adam = pyro.optim.SGD({"lr": 1.0e-3})  # Consider decreasing learning rate.
elbo = pyro.infer.TraceGraph_ELBO(num_particles=64, vectorize_particles=True)
conditioned_model = pyro.condition(model, data={"z": data})
svi = pyro.infer.SVI(conditioned_model, guide, adam, elbo)

losses = []
for step in range(6000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step()
    losses.append(loss)

print("Pyro ELBO:")
print((torch.tensor(losses)[5000:].mean(), torch.tensor(losses)[5000:].std()))


# ## Training (IWAE)


pyro.clear_param_store()
data = torch.tensor(5.0)

# These should be reset each training loop.
adam = pyro.optim.SGD({"lr": 1.0e-3})  # Consider decreasing learning rate.
elbo = pyro.infer.RenyiELBO(num_particles=5, vectorize_particles=True)
conditioned_model = pyro.condition(model, data={"z": data})
svi = pyro.infer.SVI(conditioned_model, guide, adam, elbo)

losses = []
for step in range(6000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step()
    losses.append(loss)

print("Pyro IWAE(K = 5):")
print((torch.tensor(losses)[5000:].mean(), torch.tensor(losses)[5000:].std()))
