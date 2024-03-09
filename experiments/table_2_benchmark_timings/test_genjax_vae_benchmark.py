import inspect
import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import jit, lax, random
from jax.example_libraries import stax
from jax.random import PRNGKey
from numpyro import optim
from numpyro.examples.datasets import MNIST, load_dataset

import genjax
from genjax import vi


def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )


def decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()),
        stax.Sigmoid,
    )


# Define our gradient estimator using our loss language.
def svi_update(
    model,
    guide,
    optimizer,
):
    def _inner(key, encoder_params, decoder_params, data):
        img_data = genjax.choice_map({"image": data.reshape((28 * 28,))})

        objective = vi.elbo(model, guide, img_data)

        (
            loss,
            (
                (decoder_params_grad,),
                (encoder_params_grad, _),
            ),
        ) = objective.value_and_grad_estimate(
            key, ((decoder_params,), (encoder_params, img_data))
        )
        return (encoder_params_grad, decoder_params_grad), loss

    def batch_update(key, svi_state, batch):
        optimizer_state = svi_state
        encoder_params, decoder_params = optimizer.get_params(optimizer_state)
        sub_keys = jax.random.split(key, len(batch))
        (encoder_grads, decoder_grads), loss = jax.vmap(
            _inner, in_axes=(0, None, None, 0)
        )(sub_keys, encoder_params, decoder_params, batch)
        encoder_grads, decoder_grads = jtu.tree_map(
            jnp.mean, (encoder_grads, decoder_grads)
        )
        optimizer_state = optimizer.update(
            (encoder_grads, decoder_grads), optimizer_state
        )
        return optimizer_state, loss

    return batch_update


@jit
def binarize(rng_key, batch):
    return random.bernoulli(rng_key, batch).astype(batch.dtype)


# Benchmark via `pytest`.
def test_benchmark(benchmark):
    hidden_dim = 10
    z_dim = 100
    learning_rate = 1.0e-3
    batch_size = 64

    encoder_nn_init, encoder_nn_apply = encoder(hidden_dim, z_dim)
    decoder_nn_init, decoder_nn_apply = decoder(hidden_dim, 28 * 28)

    # Model + guide close over the neural net apply functions.
    @genjax.gen
    def model(decoder_params):
        latent = vi.mv_normal_diag_reparam(jnp.zeros(z_dim), jnp.ones(z_dim)) @ "latent"
        probs = decoder_nn_apply(decoder_params, latent)
        _ = vi.flip_enum(probs) @ "image"

    @genjax.gen
    def guide(encoder_params, chm):
        image = chm["image"]
        μ, Σ_scale = encoder_nn_apply(encoder_params, image)
        _ = vi.mv_normal_diag_reparam(μ, Σ_scale) @ "latent"

    adam = optim.Adam(learning_rate)
    svi_updater = svi_update(model, guide, adam)
    train_init, train_fetch = load_dataset(MNIST, batch_size=batch_size, split="train")
    num_train, train_idx = train_init()
    rng_key = PRNGKey(0)
    encoder_init_key, decoder_init_key = random.split(rng_key)
    _, encoder_params = encoder_nn_init(encoder_init_key, (784,))
    _, decoder_params = decoder_nn_init(decoder_init_key, (z_dim,))
    num_train, train_idx = train_init()

    @jit
    def epoch_train(svi_state, key1, key2, train_idx):
        def body_fn(i, val):
            svi_state = val
            rng_key_binarize = random.fold_in(key1, i)
            batch = binarize(rng_key_binarize, train_fetch(i, train_idx)[0])
            updater_key = random.fold_in(key2, i)
            svi_state, loss = svi_updater(updater_key, svi_state, batch)
            return svi_state

        return lax.fori_loop(0, num_train, body_fn, svi_state)

    key = random.PRNGKey(314159)
    key, sub_key = jax.random.split(key)
    optimizer_state = adam.init((encoder_params, decoder_params))
    svi_state = optimizer_state
    # Warm up.
    epoch_train(svi_state, key, sub_key, train_idx)
    svi_state = benchmark(epoch_train, svi_state, key, sub_key, train_idx)
