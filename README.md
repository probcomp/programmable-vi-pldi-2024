# Probabilistic programming with programmable variational inference
This repository contains the JAX implementation that accompanies the paper [_Probabilistic programming with programmable variational inference_](./pldi24_programmable_vi_original_submit.pdf), as well as the experiments used to generate figures and numbers in the empirical evaluation section.

## Overview

![architecture](architecture.png)

The architecture of our implementation is shown above. It consists of two main components:

* `adevjax`: a JAX-based prototype of [ADEV](https://dl.acm.org/doi/abs/10.1145/3571198), which also supports a reverse move variant.
* `genjax`: a JAX-based implementation of [the Gen probabilistic programming language](https://dl.acm.org/doi/10.1145/3314221.3314642)

The `genjax.vi` module combines these two components to automate the derivation of unbiased gradient estimators for variational inference objective functions.

## Reproducing our results

We've organized the experiments code under the `experiments` directory. The `experiments` directory contains the following subdirectories (which map directly onto the figures and tables in the submitted version of the paper):

* `fig_2_noisy_cone`
* `fig_7_air_estimator_evaluation`
* `table_1_minibatch_gradient_benchmark`
* `table_2_benchmark_timings`
* `table_4_objective_values`

Each directory contains the code used to create an artifact in the submission.

### Setting up your environment

We utilize [`poetry`](https://python-poetry.org/docs/#installation) to manage Python dependencies, and utilize [`just`](https://github.com/casey/just) as a command runner. At a bare minimum, you'll need to install `poetry`, but we also recommend installing `just` to utilize some of our convenience commands (to run experiments, and get compatible versions of `torch` and `jaxlib`).

With `poetry` installed, you can use `poetry shell` to create a virtual environment. Run:
```
poetry shell
poetry install
```
to instantiate a virtual environment and install the Python dependencies.

### GPU acceleration

Several of our experiments are computationally intensive, and we recommend GPU acceleration.

For GPU acceleration, we assume access to a CUDA 11 enabled environment, we've setup a convenient command to install `torch` and `jaxlib` with support for CUDA 11:
```
just gpu
```
This will fetch versions of `torch` and `jaxlib` _which are compatible with each other_ (because we're benchmarking both `torch` and `jax`-enabled code). This is not a trivial thing: the versions we've selected we've guaranteed for compatibility, so we recommend attempting to setup your system so that you can run this command successfully. If you have a CUDA 11 enabled system, and you ran `poetry install` as above, you should be okay.

### Running the experiments

> [!IMPORTANT] 
> Several of the experiments are computationally intensive, and may take a long time to run. We recommend running them on a machine with a GPU, and using `jax` and `torch` backend that supports GPU computation. In particular, `fig_7_air_estimator_evaluation` (`just fig_7`), `table_2_benchmark_timings` (`just table_2`), and `table_1_minibatch_gradient_benchmark` (`just table_1`) will take quite a long time on a CPU.


To run _all of the experiments_, it suffices to run (**this will take a long time**):

```
just run_all
```

The experiments will be run in order, and the figure results will be saved to the `./fig` directory at the top level as PDF files.

## Extending our work
There are several ways to extend our work. We've provided [a tutorial notebook](./notebooks/extending_our_work.ipynb) which covers a few of these ways:
* (**Extending ADEV, the automatic differentiation algorithm, with new samplers equipped with gradient strategies.**) After implementing the ADEV interfaces for these objects, they can be freely lifted into the `Distribution` type of our language, and can be used in modeling and guide code. We illustrate this process by implementing `beta_implicit`, and using it in a model and guide program from the Pyro tutorials.
* (**Implementing new loss functions, by utilizing the modeling interfaces in our language.**) We illustrate this process by implementing [SDOS](https://arxiv.org/abs/2103.01030), an estimator for a symmetric KL divergence, using our language and automated the derivation of gradients for a guide program.
* (**Using a standard loss function (like `genjax.vi.elbo`) with new models and guides.**) By virtue of the programmability of our system, this is a standard means of extending our work. This extension is covered in the tutorial for the first case, above.
