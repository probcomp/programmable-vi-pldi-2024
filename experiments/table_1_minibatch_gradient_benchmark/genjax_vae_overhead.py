#!/usr/bin/env python
# coding: utf-8


import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.examples.datasets import MNIST, load_dataset
from tensorflow_probability.substrates import jax as tfp

import genjax
from genjax import Pytree, choice_map, vi
from genjax.typing import Any

console = genjax.pretty()
key = jax.random.PRNGKey(314159)

train_init, train_fetch = load_dataset(MNIST, batch_size=4096, split="train")
num_train, train_idx = train_init()
data_batch = train_fetch(0)[0]
batch_sizes = [64, 128, 256, 512, 1024]


# Utilities for defining the model and the guide.
@dataclass
class Decoder(Pytree):
    dense_1: Any
    dense_2: Any

    def flatten(self):
        return (self.dense_1, self.dense_2), ()

    @classmethod
    def new(cls, key1, key2):
        dense_1 = eqx.nn.Linear(10, 200, key=key1)
        dense_2 = eqx.nn.Linear(200, 28 * 28, key=key2)
        return Decoder(dense_1, dense_2)

    def __call__(self, z_what):
        v = self.dense_1(z_what)
        v = jax.nn.leaky_relu(v)
        v = self.dense_2(v)
        return jax.nn.sigmoid(v)


# Create our decoder.
key, sub_key1, sub_key2 = jax.random.split(key, 3)
decoder = Decoder.new(sub_key1, sub_key2)


@dataclass
class Encoder(Pytree):
    dense_1: Any
    dense_2: Any

    def flatten(self):
        return (self.dense_1, self.dense_2), ()

    @classmethod
    def new(cls, key1, key2):
        dense_1 = eqx.nn.Linear(28 * 28, 200, key=key1)
        dense_2 = eqx.nn.Linear(200, 20, key=key2)
        return Encoder(dense_1, dense_2)

    def __call__(self, data):
        v = self.dense_1(data)
        v = jax.nn.leaky_relu(v)
        v = self.dense_2(v)
        return v[0:10], jax.nn.softplus(v[10:])


key, sub_key1, sub_key2 = jax.random.split(key, 3)
encoder = Encoder.new(sub_key1, sub_key2)


@genjax.gen
def model(decoder):
    latent = genjax.tfp_mv_normal_diag(jnp.zeros(10), jnp.ones(10)) @ "latent"
    logits = decoder(latent)
    _ = genjax.tfp_bernoulli(logits) @ "image"


@genjax.gen
def guide(chm, encoder):
    image = chm["image"]
    μ, Σ_scale = encoder(image)
    _ = vi.mv_normal_diag_reparam(μ, Σ_scale) @ "latent"


def batch_elbo_grad_estimate(key, encoder, decoder, data_batch):
    def _inner(key, encoder, decoder, data):
        chm = choice_map({"image": data.flatten()})
        objective = vi.elbo(model, guide, chm)
        return objective.grad_estimate(key, ((decoder,), (chm, encoder,)))

    sub_keys = jax.random.split(key, len(data_batch))
    return jax.vmap(_inner, in_axes=(0, None, None, 0))(
        sub_keys, encoder, decoder, data_batch
    )


jit1 = jax.jit(batch_elbo_grad_estimate)


vi_runtime_over_batches = []
std_ds = []
for batch_size in batch_sizes:
    train_init, train_fetch = load_dataset(MNIST, batch_size=batch_size, split="train")
    num_train, train_idx = train_init()
    data_batch = train_fetch(0)[0]
    durations = []
    jit1(key, encoder, decoder, data_batch)
    for i in range(0, 300):
        t0 = time.perf_counter()
        jit1(key, encoder, decoder, data_batch)
        duration = time.perf_counter() - t0
        durations.append(duration)
    vi_runtime_over_batches.append(jnp.mean(jnp.array(durations)))
    std_ds.append(jnp.std(jnp.array(durations)))

print("GenJAX VI timings:")
print(f"Batch sizes: {batch_sizes}")
print((1000 * np.array(vi_runtime_over_batches), 1000 * np.array(std_ds)))


tfd = tfp.distributions

MvNormalDiag = tfd.MultivariateNormalDiag
Bernoulli = tfd.Bernoulli


# Handcoded.
def batch_elbo_grad_estimate(key, encoder, decoder, data_batch):
    def single_estimate(key, encoder, decoder, data):
        image = data.flatten()

        def loss_estimate(params):
            (encoder, decoder) = params
            μ, Σ_scale = encoder(image)
            v = MvNormalDiag(jnp.zeros(10), jnp.ones(10)).sample(seed=key)
            s = μ + v * Σ_scale
            guide_normal_logp = MvNormalDiag(μ, Σ_scale).log_prob(s)
            model_normal_logp = MvNormalDiag(jnp.zeros(10), jnp.ones(10)).log_prob(s)
            logits = decoder(s)
            model_bernoulli_logp = Bernoulli(logits=logits).log_prob(image).sum()
            return (model_bernoulli_logp + model_normal_logp) - guide_normal_logp

        return jax.grad(loss_estimate)((encoder, decoder))

    sub_keys = jax.random.split(key, len(data_batch))
    return jax.vmap(single_estimate, in_axes=(0, None, None, 0))(
        sub_keys, encoder, decoder, data_batch
    )


jit2 = jax.jit(batch_elbo_grad_estimate)


hand_runtime_over_batches = []
hand_std_ds = []
for batch_size in batch_sizes:
    print(batch_size)
    train_init, train_fetch = load_dataset(MNIST, batch_size=batch_size, split="train")
    num_train, train_idx = train_init()
    data_batch = train_fetch(0)[0]
    durations = []
    jit2(key, encoder, decoder, data_batch)
    for i in range(0, 300):
        t0 = time.perf_counter()
        jit2(key, encoder, decoder, data_batch)
        duration = time.perf_counter() - t0
        durations.append(duration)
    hand_runtime_over_batches.append(jnp.mean(jnp.array(durations)))
    hand_std_ds.append(jnp.std(jnp.array(durations)))

print("Handcoded timings:")
print(f"Batch sizes: {batch_sizes}")
print((1000.0 * np.array(hand_runtime_over_batches), 1000.0 * np.array(hand_std_ds)))
