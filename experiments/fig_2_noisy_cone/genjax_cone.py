#!/usr/bin/env python
# coding: utf-8


import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

import genjax
from genjax import gensp
from genjax import vi

console = genjax.pretty()
key = jax.random.PRNGKey(314159)
sns.set_theme(style="white")
label_fontsize = 70  # Set the desired font size here


# ## Model


@genjax.gen
def model():
    x = vi.normal_reparam(0.0, 10.0) @ "x"
    y = vi.normal_reparam(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    z = vi.normal_reparam(rs, 0.1 + (rs / 100.0)) @ "z"


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 5000)
tr = jax.jit(jax.vmap(model.simulate, in_axes=(0, None)))(sub_keys, ())
chm = tr.strip()


x, y, z = chm["x"], chm["y"], chm["z"]
scores = tr.get_score()
ch = z < 30.0
x = x[ch]
y = y[ch]
z = z[ch]
scores = scores[ch]

fig = plt.figure(figsize=(16, 16))  # Adjust the size for better visualization
gs = GridSpec(1, 2)
xp = np.linspace(-4, 4, 2)
yp = np.linspace(-4, 4, 2)
xp, yp = np.meshgrid(xp, yp)
zp = np.full_like(xp, 5)
ax1 = fig.add_subplot(gs[0, 0], projection="3d")  # This spans both columns
ax1.scatter(x, y, z, c=scores, cmap="viridis")
ax1.set_zlim(0, 25)
ax1.set_ylim(-4, 4)
ax1.set_xlim(-4, 4)
ax1.set_xlabel("x", fontsize=label_fontsize)
ax1.set_ylabel("y", fontsize=label_fontsize)
ax1.text(
    x=-8.5,
    y=0,
    z=2,
    s="z",
    rotation=90,
    verticalalignment="center",
    fontsize=label_fontsize,
)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
xp = np.linspace(-6, 6, 2)
yp = np.linspace(-6, 6, 2)
xp, yp = np.meshgrid(xp, yp)
zp = np.full_like(xp, 5)
ax2 = fig.add_subplot(gs[0, 1], projection="3d")
ax2.set_zlim(0, 25)
ax2.set_ylim(-4, 4)
ax2.set_xlim(-4, 4)
ax2.scatter(x, y, z, c=scores, cmap="viridis")
ax2.plot_surface(xp, yp, zp, alpha=0.3, zorder=1)
ax2.text(
    3.0, -3.4, 6.8, "z = 5.0", color="black", zorder=3, fontsize=label_fontsize / 2
)
ax2.set_xlabel("x", fontsize=label_fontsize)
ax2.set_zlabel("z", fontsize=label_fontsize)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.view_init(elev=0, azim=-90)
plt.tight_layout()
fig.savefig("./fig/fig_2_model_prior.pdf", format="pdf")


# ## Naive variational guide


# Now, we define our variational proposal.
@genjax.gen
def variational_family(data, ϕ):
    μ1, μ2, log_σ1, log_σ2 = ϕ
    x = vi.normal_reparam(μ1, jnp.exp(log_σ1)) @ "x"
    y = vi.normal_reparam(μ2, jnp.exp(log_σ2)) @ "y"


data = genjax.choice_map({"z": 5.0})
objective = vi.elbo(model, variational_family, data)


# Training.
key = jax.random.PRNGKey(314159)
ϕ = (0.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, (ϕ_grads,)) = jitted(sub_keys, ((), (ϕ,)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    losses.append(jnp.mean(loss))
    if i % 1000 == 0:
        print(jnp.mean(loss))
print(ϕ)


# ### Sampling from the trained variational family


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 50000)
data = genjax.choice_map({"z": 5.0})
tr = jax.jit(jax.vmap(variational_family.simulate, in_axes=(0, None)))(
    sub_keys, (data, ϕ)
)
chm = tr.strip()
x, y = chm["x"], chm["y"]
scores = jnp.exp(tr.get_score())
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.scatter(x, y, c=scores, cmap="viridis", marker=".", s=20)
circle = patches.Circle((0.0, 0.0), radius=jnp.sqrt(5.0), fc="none", ec="black", lw=4)
ax.add_patch(circle)
ax.text(2.0, 2.3, "z = 5.0", ha="center", va="center", fontsize=label_fontsize)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x", fontsize=label_fontsize)
ax.set_ylabel("y", fontsize=label_fontsize)
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.labelpad = 18  # adjust the value as needed
ax.yaxis.label.set_rotation(0)  # 90 degrees for vertical
plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
fig.savefig("./fig/fig_2_naive_variational_elbo_samples.pdf", format="pdf")


# Training.
key = jax.random.PRNGKey(2)
ϕ = (3.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, (ϕ_grads,)) = jitted(sub_keys, ((), (ϕ,)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    if i % 1000 == 0:
        print(jnp.mean(loss))
    losses.append(jnp.mean(loss))
print(ϕ)


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 50000)
data = genjax.choice_map({"z": 5.0})
tr = jax.jit(jax.vmap(variational_family.simulate, in_axes=(0, None)))(
    sub_keys, (data, ϕ)
)
chm = tr.strip()


x, y = chm["x"], chm["y"]
scores = jnp.exp(tr.get_score())
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.scatter(x, y, marker=".", s=20)
circle = patches.Circle((0.0, 0.0), radius=jnp.sqrt(5.0), fc="none", ec="black", lw=4)
ax.add_patch(circle)
ax.text(2.0, 2.3, "z = 5.0", ha="center", va="center", fontsize=label_fontsize)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x", fontsize=label_fontsize)
ax.set_ylabel("y", fontsize=label_fontsize)
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.labelpad = 18  # adjust the value as needed
ax.yaxis.label.set_rotation(0)  # 90 degrees for vertical
plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area

fig.savefig("./fig/fig_2_naive_variational_elbo_samples_2.pdf", format="pdf")

# ## Training with IWAE

# $$\text{IWAE ELBO} = E_{\{z_k\}_{k = 0}^N \sim Q}[\log \frac{1}{N}\sum_i w_i(z_i)]$$
#
# where $w_i(z_i) = \frac{P(z_i, x)}{Q(z_i)}$

# ## 5 particle IWAE


data = genjax.choice_map({"z": 5.0})
iwae_objective = vi.iwae_elbo(model, variational_family, data, 5)


# Training with IWAE.
key = jax.random.PRNGKey(314159)
ϕ = (3.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(iwae_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 1)
    loss, (_, (ϕ_grads,)) = jitted(sub_keys, ((), (ϕ,)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    if i % 1000 == 0:
        print(jnp.mean(loss))
    losses.append(jnp.mean(loss))

print(ϕ)


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 50000)
data = genjax.choice_map({"z": 5.0})
tr = jax.jit(jax.vmap(variational_family.simulate, in_axes=(0, None)))(
    sub_keys, (data, ϕ)
)
chm = tr.strip()


x, y = chm["x"], chm["y"]
scores = jnp.exp(tr.get_score())
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.scatter(x, y, c=scores, cmap="viridis", marker=".", s=20)
circle = patches.Circle((0.0, 0.0), radius=jnp.sqrt(5.0), fc="none", ec="black", lw=4)
ax.add_patch(circle)
ax.text(-2.0, 2.3, "z = 5.0", ha="center", va="center", fontsize=label_fontsize)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x", fontsize=label_fontsize)
ax.set_ylabel("y", fontsize=label_fontsize)
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.labelpad = 18  # adjust the value as needed
ax.yaxis.label.set_rotation(0)  # 90 degrees for vertical
plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area

fig.savefig("./fig/fig_2_naive_variational_iwae_elbo_5_samples.pdf", format="pdf")


# ## 20 particle IWAE


iwae_objective = vi.iwae_elbo(
    model, variational_family, genjax.choice_map({"z": 5.0}), 20
)


# Training with IWAE.
key = jax.random.PRNGKey(314159)
ϕ = (3.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(iwae_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, (ϕ_grads,)) = jitted(sub_keys, ((), (ϕ,)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    if i % 1000 == 0:
        print(jnp.mean(loss))
    losses.append(jnp.mean(loss))
print(ϕ)


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 50000)
data = genjax.choice_map({"z": 5.0})
tr = jax.jit(jax.vmap(variational_family.simulate, in_axes=(0, None)))(
    sub_keys, (data, ϕ)
)
chm = tr.strip()


x, y = chm["x"], chm["y"]
scores = jnp.exp(tr.get_score())
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.scatter(x, y, c=scores, cmap="viridis", marker=".", s=20)
circle = patches.Circle((0.0, 0.0), radius=jnp.sqrt(5.0), fc="none", ec="black", lw=4)
ax.add_patch(circle)
ax.text(-2.0, 2.3, "z = 5.0", ha="center", va="center", fontsize=label_fontsize)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x", fontsize=label_fontsize)
ax.set_ylabel("y", fontsize=label_fontsize)
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.labelpad = 18  # adjust the value as needed
ax.yaxis.label.set_rotation(0)  # 90 degrees for vertical
plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
fig.savefig("./fig/fig_2_naive_variational_iwae_elbo_20_samples.pdf", format="pdf")


# ## 50 particle IWAE
# NOTE: slow to run, because vmap + ADEV is not
# yet fully supported.

iwae_objective = vi.iwae_elbo(
    model, variational_family, genjax.choice_map({"z": 5.0}), 50
)


# Training with IWAE.
key = jax.random.PRNGKey(314159)
ϕ = (3.0, 0.0, 1.0, 1.0)
jitted = jax.jit(jax.vmap(iwae_objective.value_and_grad_estimate, in_axes=(0, None)))
losses = []
for i in range(0, 5000):
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, 64)
    loss, (_, (ϕ_grads,)) = jitted(sub_keys, ((), (ϕ,)))
    ϕ = jtu.tree_map(lambda v, g: v + 1e-3 * jnp.mean(g), ϕ, ϕ_grads)
    if i % 1000 == 0:
        print(jnp.mean(loss))
    losses.append(jnp.mean(loss))
print(ϕ)


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 50000)
data = genjax.choice_map({"z": 5.0})
tr = jax.jit(jax.vmap(variational_family.simulate, in_axes=(0, None)))(
    sub_keys, (data, ϕ)
)
chm = tr.strip()
x, y = chm["x"], chm["y"]
scores = jnp.exp(tr.get_score())
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.scatter(x, y, c=scores, cmap="viridis", marker=".", s=20)
circle = patches.Circle((0.0, 0.0), radius=jnp.sqrt(5.0), fc="none", ec="black", lw=4)
ax.add_patch(circle)
ax.text(-2.0, 2.3, "z = 5.0", ha="center", va="center", fontsize=label_fontsize)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x", fontsize=label_fontsize)
ax.set_ylabel("y", fontsize=label_fontsize)
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.labelpad = 18  # adjust the value as needed
ax.yaxis.label.set_rotation(0)  # 90 degrees for vertical
plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
fig.savefig("./fig/fig_2_naive_variational_iwae_elbo_50_samples.pdf", format="pdf")


# ### Plotting inference results using SIR.


@genjax.gen
def hacky_model(ϕ):
    x = vi.normal_reparam(0.0, 10.0) @ "x"
    y = vi.normal_reparam(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    z = vi.normal_reparam(rs, 0.1 + (rs / 100.0)) @ "z"


@genjax.gen
def hacky_variational_family(tgt):
    (ϕ,) = tgt.args
    μ1, μ2, log_σ1, log_σ2 = ϕ
    x = vi.normal_reparam(μ1, jnp.exp(log_σ1)) @ "x"
    y = vi.normal_reparam(μ2, jnp.exp(log_σ2)) @ "y"


key, sub_key = jax.random.split(key)
sub_keys = jax.random.split(sub_key, 50000)
data = genjax.choice_map({"z": 5.0})
chm_variational = gensp.choice_map_distribution(hacky_variational_family)
sir = gensp.CustomImportance(50, chm_variational)
scores, v_chm = jax.vmap(sir.random_weighted, in_axes=(0, None))(
    sub_keys, gensp.target(hacky_model, (ϕ,), data)
)
chm = v_chm.get_leaf_value()


x, y = chm["x"], chm["y"]
scores = jnp.exp(tr.get_score())
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.scatter(x, y, marker=".", s=20)
circle = patches.Circle((0.0, 0.0), radius=jnp.sqrt(5.0), fc="none", ec="black", lw=4)
ax.add_patch(circle)
ax.text(-2.0, 2.3, "z = 5.0", ha="center", va="center", fontsize=label_fontsize)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x", fontsize=label_fontsize)
ax.set_ylabel("y", fontsize=label_fontsize)
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.labelpad = 18  # adjust the value as needed
ax.yaxis.label.set_rotation(0)  # 90 degrees for vertical
plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
fig.savefig("./fig/fig_2_naive_variational_iwae_elbo_50_sir_samples.pdf", format="pdf")
