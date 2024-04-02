# The default `just` lists all available commands.
default:
  @just --list

# Get GPU torch & jax
gpu:
  poetry run pip install --upgrade torch==2.1.0 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
  poetry run pip install jax[cuda12_pip]==0.4.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# make fig_2_noisy_cone
# These are components for the overview figure for the paper.
fig_2:
  mkdir -p figs
  poetry run python experiments/fig_2_noisy_cone/genjax_cone.py
  poetry run python experiments/fig_2_noisy_cone/genjax_cone_marginal.py

# make fig_7_air_estimator_evaluation
# These are components for the AIR comparison between our system and Pyro.
# Note: we omitted Pyro RWS in this script (because it's slow).
fig_7:
  mkdir -p figs
  mkdir -p training_runs
  poetry run python experiments/fig_7_air_estimator_evaluation/genjax_enum_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/genjax_mvd_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/genjax_reinforce_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/genjax_hybrid_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/genjax_rws_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/pyro_reinforce_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/pyro_baselines_air.py
  poetry run python experiments/fig_7_air_estimator_evaluation/air_analysis.py

# generate the numbers for table_1_minibatch_gradient_benchmark
# Not a plot, just timings printed out.
table_1:
  poetry run python experiments/table_1_minibatch_gradient_benchmark/genjax_vae_overhead.py

# generate the numbers for table_2_benchmark_timings
# Not a plot, just timings printed out using `pytest-benchmark`.
table_2:
  poetry run pytest experiments/table_2_benchmark_timings --benchmark-disable-gc

# generate the numbers for table_4_objective_values
# Not a plot, just timings printed out.
table_4:
  poetry run python experiments/table_4_objective_values/genjax_cone.py
  poetry run python experiments/table_4_objective_values/genjax_cone_marginal.py
  poetry run python experiments/table_4_objective_values/numpyro_cone.py
  poetry run python experiments/table_4_objective_values/pyro_cone.py
  
run_all:
  @just fig_2
  @just table_4
  @just table_1
  @just table_2
  @just fig_7
