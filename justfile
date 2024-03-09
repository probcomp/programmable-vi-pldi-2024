# get GPU jax
gpu:
  pip install --upgrade torch==2.1.0 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
  pip install jax[cuda11_local]==0.4.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# make fig_7_air_estimator_evaluation
fig_7:
  poetry run python experiments/fig_7_air_estimator_evaluation/genjax_enum_air.py

# generate the numbers for table_1_minibatch_gradient_benchmark
table_1:
  poetry run python experiments/table_1_minibatch_gradient_benchmark/genjax_vae_overhead.py

# generate the numbers for table_2_benchmark_timings
table_2:
  poetry run pytest experiments/table_2_benchmark_timings
