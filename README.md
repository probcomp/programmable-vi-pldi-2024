# Probabilistic programming with programmable variational inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10806202.svg)](https://doi.org/10.5281/zenodo.10806202)

This repository contains the JAX implementation that accompanies the paper [_Probabilistic programming with programmable variational inference_](./pldi24_programmable_vi_original_submit.pdf), as well as the experiments used to generate figures and numbers in the empirical evaluation section.

## Overview

The artifact accompanying our paper is a Python library for probabilistic programming with variational inference.

Using our library, users can write *probabilistic programs* encoding probabilistic models and variational families. They can then define *variational objectives* to optimize, such as the ELBO. Our system can automate the estimation of gradients of these objectives, enabling gradient-based training.

### Example

For example, here is the model from Figure 2 of our paper, encoding a probability distribution on triples (x, y, z) located near a cone:

```python
import genjax
from genjax import gensp, vi

@genjax.gen
def model():
    x = vi.normal_reparam(0.0, 10.0) @ "x"
    y = vi.normal_reparam(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    z = vi.normal_reparam(rs, 0.1 + (rs / 100.0)) @ "z"
```

Our goal will be to infer values of $x$ and $y$ consistent with an observation that $z = 5$:
```python
data = genjax.choice_map({"z": 5.0})
```

To do so, we define a *variational family* -- a parametric family of distributions over the latent variables (x, y).
```python
@genjax.gen
def variational_family(_, ϕ):
    μ1, μ2, log_σ1, log_σ2 = ϕ
    x = vi.normal_reparam(μ1, jnp.exp(log_σ1)) @ "x"
    y = vi.normal_reparam(μ2, jnp.exp(log_σ2)) @ "y"
```

We want to find parameters that minimize the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound), a loss function that encourages the variational family to be close to the Bayesian posterior over the latent variables:

```python
objective = vi.elbo(model, variational_family, data)
```

For convenience, the ELBO is defined as a library function, but users can also define their own objectives.

Our library can automatically estimate gradients of the ELBO and other probabilistic loss functions, and these gradients can be used to optimize the variational family's parameters.

```python
import jax
import jax.tree_util as jtu

# Training.
key = jax.random.PRNGKey(314159)
ϕ = (0.0, 0.0, 1.0, 1.0)
L = jax.jit(jax.vmap(objective.value_and_grad_estimate, in_axes=(0, None)))
for i in range(5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    (loss, (_, (_, ϕ_grads))) = L(sub_keys, ((), (data, ϕ)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
```

The full code to run this example and plot results can be found in the `experiments/fig_2_noisy_cone` directory.

### Documentation

To ensure reproducibility, we have packaged a frozen version of our `genjax` library in this repository. Documentation is publicly available for the relevant modules, but please note that some APIs have changed since we submitted our paper, and the documentation may be out-of-sync with the code in this repository:
* [`adevjax`](https://probcomp.github.io/genjax/library/adev.html)
* [`genjax`](https://probcomp.github.io/genjax/library/index.html)
* [`genjax.vi`](https://probcomp.github.io/genjax/library/inference/vi.html)

For guidance on usage of the exact library version included in this artifact, please instead see our included tutorial notebook on [using and extending `genjax.vi`](./extending_our_work.ipynb).

## Reproducing the paper's claims

We have provided code to reproduce all of the experiments in the paper, namely the results in Figure 2, Figure 7, Table 1, Table 2, and Table 4. At a high level, these experiments validate our claims that (1) our implementation introduces minimal overhead compared to hand-coded gradient estimators, (2) our implementation of variational inference is faster than Pyro's, and on par with NumPyro's, for algorithms that all systems can express, and (3) our system supports gradient estimators that Pyro and NumPyro do not automate, some of which empirically converge faster than the algorithms they do support.

We've organized the experiments code under the `experiments` directory. The `experiments` directory contains the following subdirectories, which map onto the figures and tables in the submitted version of the paper:

* `fig_2_noisy_cone`
* `fig_7_air_estimator_evaluation`
* `table_1_minibatch_gradient_benchmark`
* `table_2_benchmark_timings`
* `table_4_objective_values`

Each directory contains code used to run the experiment, and also to reproduce the graphs or table results that appear in the submission.

**Computational cost of the experiments:** Some of the experiments can be run on CPU in a reasonable amount of time, whereas others require GPU. (See further discussion below.) For reference:

* (**CPU likely okay**) For `fig_2_noisy_cone` and `table_4_objective_values`, a CPU should be sufficient to run the experiments. These experiments illustrate usage of our system to automate gradient estimators for [hierarchical variational inference](https://arxiv.org/abs/1511.02386) (HVI), and nested importance weighted HVI.

* (**GPU likely required**) For `fig_7_air_estimator_evaluation`, `table_1_minibatch_gradient_benchmark`, and `table_2_benchmark_timings`, we recommend running on a GPU. These experiments illustrate various performance comparisons between our system, handcoded gradient estimators, and [Pyro](https://pyro.ai/) for several variational objectives and estimators.


### Setting up your environment

There are several possible ways to make an environment which can run the experiments.

#### Setup using `docker`

Possibly the easiest way is to use [`docker`](https://docs.docker.com/). We've provided a `Dockerfile` with the repository, with a public image which we've curated. To setup an environment, run:

```
docker build .
```

This will proceed to build a container which you can use. If the build succeeds, you can then run a virtual machine using the container:

* **With GPU support (requires underlying Nvidia driver)**
```
docker run --runtime nvidia -it <YOUR_IMAGE_ID>
```

* **Without GPU support**
```
docker run -it <YOUR_IMAGE_ID>
```

where `<YOUR_IMAGE_ID>` is the hash of the image you built.

With this method, you can ignore the setup for `poetry` and `just` below, and jump to [GPU acceleration](https://github.com/probcomp/programmable-vi-pldi-2024/tree/main?tab=readme-ov-file#gpu-acceleration) (if you have access to a GPU) and then [Running the experiments](https://github.com/probcomp/programmable-vi-pldi-2024/tree/main?tab=readme-ov-file#running-the-experiments).

#### Setup using `poetry` and `just`

There's an alternative path if you forego `docker`: we utilize [`poetry`](https://python-poetry.org/docs/#installation) to manage Python dependencies, and utilize [`just`](https://github.com/casey/just) as a command runner. We recommend installing both of these tools, using the documentation at the links provided (at a bare minimum, you'll need to install `poetry`, but we also recommend installing `just` to utilize some of our convenience commands (to run experiments, and get compatible versions of `torch` and `jaxlib`)).

With `poetry` installed, you can use `poetry install` to create and install our dependencies into a virtual environment. Run:
```
poetry install
```
to create and install dependencies into a virtual environment. Then, run:
```
poetry shell
```
to instantiate the virtual environment.

#### GPU acceleration

Several of our experiments are computationally intensive, and we recommend GPU acceleration.

For GPU acceleration, we assume access to a CUDA 12 enabled environment. There is a convenience command to install `torch` and `jaxlib` with support for CUDA 12:
```
just gpu
```
This will fetch versions of `torch` and `jaxlib` _which are compatible with each other_ (because we're benchmarking both `torch` and `jax`-enabled code). 

The versions we've selected we've guaranteed for compatibility, so we recommend attempting to setup your system so that you can run this command successfully. If you have a CUDA 12 enabled system, and you ran `poetry install` as above (or you came from the `docker` setup), you should be okay.

### Running the experiments

> [!IMPORTANT] 
> Several of the experiments are computationally intensive, and may take a long time to run. We recommend running them on a machine with a GPU, and using `jax` and `torch` backend that supports GPU computation. 
> 
> In particular, `fig_7_air_estimator_evaluation` (`just fig_7`), `table_2_benchmark_timings` (`just table_2`), and `table_1_minibatch_gradient_benchmark` (`just table_1`) will take quite a long time on a CPU.

**Using `experiments.ipynb` to run experiments**

[There's a Jupyter notebook](./experiments.ipynb) which provides access to all the experiments via a single interface. Jupyter also allows you to view the PDFs generated by the experiments.

**Using `just` to run experiments**

To run _all of the experiments_, it suffices to run (**this will take a long time and not recommended on CPU**):

```
just run_all
```

The experiments will be run in order, and the figure results will be saved to the `./fig` directory at the top level as PDF files.

You can also run each experiment individually, the set of recipes available via `just` are:
```
❯ just -l
Available recipes:
    fig_2   # These are components for the overview figure for the paper.
    fig_7   # These are components for the AIR figure for the paper.
    gpu     # get GPU jax
    table_1 # Not a plot, just timings printed out.
    table_2 # Not a plot, just timings printed out using `pytest-benchmark`.
    table_4 # Not a plot, just timings printed out.
```

meaning that one can run any of these experiments using `just`, for example:

```
just table_1
```

## Interpreting the results

### Abbreviations

There are also several abbreviations which are not collected in a single place in the artifact:

* AIR -- Attend, Infer, Repeat, a generative model of multi-object images
* VAE -- variational autoencoder
* ELBO -- evidence lower bound
* MVD -- the measure valued derivative estimator
* REINFORCE -- the score function derivative estimator
* IWELBO -- importance weighted ELBO
* HVI -- hierarchical variational inference
* IWHVI -- importance weighted HVI
* DIWHVI -- doubly importance weighted HVI

### Correspondence between print out results and tables

Several of our experiments (the experiments which produce results for the tables) print out results to stdout. Below, we give a guide to interpreting the results:

* (**Table 1**): For Table 1, "Ours" refers to the GenJAX VI timings. The rows of the table go by batch size, and the first array returned by the print out is the mean over several runs, the second is the std deviation.

* (**Table 2**): To generate Table 2 in the paper, we took the mean and std dev numbers from the `pytest-benchmark` print out. The labels for the columns in the table are mapped from the names e.g. 
  * `genjax_reinforce` and `pyro_reinforce[TraceGraph_ELBO]` -> REINFORCE
  * `genjax_mvd` -> MVD
  * `genjax_iwae_mvd` -> IWAE + MVD
  * `pyro_reinforce[RenyiELBO]` -> IWAE + REINFORCE

  and so on. Each of these names correspond to particular _gradient estimators strategies_ used in variational inference.

* (**Table 4**): For Table 4, the "IWAE" label is equivalent to "IWELBO", as is the RenyiELBO name (from Pyro and NumPyro). All system comparison experiments (Pyro and NumPyro) are labeled with their names in this table. We did not report standard deviation in this table, but for each experiment, the first array is the mean over several trials, and the second is standard deviation.

## Notes on artifact evaluation

For our submission results, our hardware was a Linux box with a Nvidia RTX 4090, and an AMD Ryzen 7 7800x3D CPU, with CUDA support (checked via `nvidia-smi`) up to 12.4. 

We also ran our experiments on a Linux box with an Nvidia Tesla V100 SMX2 16 GB, and an Intel Xeon (8) @ 2.2 GHz, with support for CUDA 12. In both experiment environments, we observed the same general trends, including the same order-of-magnitude speed-ups of our JAX-based gradient estimators compared to Pyro's.

Note that even on GPU, Pyro's implementation of the reweighted wake-sleep (RWS) algorithm may be prohibitively slow (due to the `batch_size = 1` restriction).


## Building on our work
There are several ways to build on our work. We've provided [a tutorial notebook](./extending_our_work.ipynb) which provides an introduction to our system in the context of using it for variational inference on new model and guide programs.
