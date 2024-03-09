# Probabilistic programming with programmable variational inference
This repository contains the JAX prototype that accompanies the paper [_Probabilistic programming with programmable variational inference_](./pldi24_programmable_vi_original_submit.pdf), as well as the experiments used to generate (some of) the figures in the empirical evaluation section.

## Overview of the artifact

![architecture](./architecture.png)

The architecture of our artifact is shown above. It consists of two main components:

* `adevjax`: a JAX-based prototype of [ADEV](https://dl.acm.org/doi/abs/10.1145/3571198), which also supports a reverse move variant.
* `genjax`: a JAX-based implementation of [the Gen probabilistic programming language](https://dl.acm.org/doi/10.1145/3314221.3314642)

Our `genjax.vi` artifact combines these two components to automate the derivation of unbiased gradient estimators for variational inference objective functions.

## Reproducing our results

We've organized the experiments code under the `experiments` directory. The `experiments` directory contains the following subdirectories (which map directly onto the figures in the submitted version of the paper):

* `fig_2_noisy_cone`
* `fig_7_air_estimator_evaluation`
* `table_1_minibatch_gradient_benchmark`
* `table_2_benchmark_timings`
* `table_4_objective_values`

Each directory contains the code used to create an artifact in the submission.

### Running the experiments
