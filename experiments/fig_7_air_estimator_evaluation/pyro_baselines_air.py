#!/usr/bin/env python
# coding: utf-8


import time

import numpy as np
import pandas as pd
import pyro
import pyro.contrib.examples.multi_mnist as multi_mnist
import torch
from pyro.infer import (
    SVI,
    TraceEnum_ELBO,
)
from pyro.optim import Adam
from pyro_air import AIR, count_accuracy, get_per_param_lr


#####################
# Benchmark Configs
#####################
seed = 123456
use_cuda = torch.cuda.is_available()
batch_size = 64
num_epoches = 41

z_pres_prior = 0.01
learning_rate = 1e-4
baseline_lr = 1e-1
elbo = TraceEnum_ELBO()
# explicitly list out all configurable options
air_model_args = dict(
    num_steps=3,
    x_size=50,
    window_size=28,
    z_what_size=50,
    rnn_hidden_size=256,
    encoder_net=[200],
    decoder_net=[200],
    predict_net=[200],
    embed_net=None,
    bl_predict_net=[200],
    non_linearity="ReLU",
    decoder_output_bias=-2,
    decoder_output_use_sigmoid=True,
    use_masking=True,
    use_baselines=False,
    baseline_scalar=None,
    scale_prior_mean=3.0,
    scale_prior_sd=0.2,
    pos_prior_mean=0.0,
    pos_prior_sd=1.0,
    likelihood_sd=0.3,
    use_cuda=use_cuda,
)


#####################
# Initial Setup
#####################
device = torch.device("cuda" if use_cuda else "cpu")


def z_pres_prior_fn(t):
    return [0.05, 0.05**2.3, 0.05**5][t]


X, Y = multi_mnist.load("data/air/.data")
X = (torch.from_numpy(X).float() / 255.0).to(device)
visualize_examples = X[5:10]
# Using float to allow comparison with values sampled from
# Bernoulli.
true_counts = torch.tensor([len(objs) for objs in Y], dtype=torch.float)


ac_losses, ac_accuracy, ac_wall_clock_times = None, None, None
pyro.distributions.enable_validation(False)
pyro.clear_param_store()  # just in case

air = AIR(**air_model_args)
adam = Adam(get_per_param_lr(learning_rate, baseline_lr))
svi = SVI(air.model, air.guide, adam, loss=elbo)

all_loss = []
all_accuracy = []
time_per_epoch = []

for i in range(num_epoches):
    start_time = time.perf_counter()
    # technically this might step over slightly more than 1 epoch...
    losses = []
    for j in range(int(np.ceil(X.size(0) / batch_size))):
        losses.append(
            svi.step(X, batch_size=batch_size, z_pres_prior_p=z_pres_prior_fn)
        )
    end_time = time.perf_counter()

    accuracy, counts, error_z, error_ix = count_accuracy(X, true_counts, air, 1000)

    all_loss.append(np.mean(losses) / X.size(0))
    all_accuracy.append(accuracy)
    time_per_epoch.append(end_time - start_time)

    print(
        f"Epoch={i}, current_epoch_step_time={time_per_epoch[-1]:.2f}, loss={all_loss[-1]:.2f}"
    )
    print(f"accuracy={accuracy}, counts={counts}")

# Save run.
wall_clock_times = np.cumsum(time_per_epoch)
arr = np.array([all_loss, all_accuracy, wall_clock_times])
df = pd.DataFrame(arr.T, columns=["ELBO loss", "Accuracy", "Epoch wall clock times"])
df.to_csv(
    f"./training_runs/pyro_air_epochs_{num_epoches}.csv",
    index=False,
)
