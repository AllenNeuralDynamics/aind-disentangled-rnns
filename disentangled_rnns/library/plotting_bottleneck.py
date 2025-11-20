"""Plotting functions for inspecting Disentangled RNNs."""

import copy
from typing import Optional

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import multisubject_disrnn
from disentangled_rnns.library import rnn_utils
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


# Fontsizes and formatting for plots.
small = 15
medium = 18
large = 20
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'


def plot_bottlenecks(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    sort_latents: bool = True,
) -> plt.Figure:
  """Plot the bottleneck sigmas from an hk.DisentangledRNN."""

  if isinstance(disrnn_config, multisubject_disrnn.MultisubjectDisRnnConfig):
    params_disrnn = params['multisubject_dis_rnn']
    subject_embedding_size = disrnn_config.subject_embedding_size
    update_input_names = [
        f'SubjEmb {i+1}' for i in range(subject_embedding_size)
    ] + disrnn_config.x_names[1:]
    # For update_sigmas: concatenate transposed reparameterized sigmas
    # Order of inputs to update nets: subject_embedding, observations, latents
    update_subj_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_subj_sigma_params']
        )
    )
    update_obs_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_obs_sigma_params']
        )
    )
    update_latent_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_latent_sigma_params']
        )
    )
    update_sigmas = np.concatenate(
        (update_subj_sigmas_t, update_obs_sigmas_t, update_latent_sigmas_t),
        axis=1,
    )

    # For choice_sigmas: concatenate reparameterized sigmas
    # Order of inputs to choice net: subject_embedding, latents
    choice_subj_sigmas = disrnn.reparameterize_sigma(
        params_disrnn['choice_net_subj_sigma_params']
    )
    choice_latent_sigmas = disrnn.reparameterize_sigma(
        params_disrnn['choice_net_latent_sigma_params']
    )
    choice_sigmas = np.concatenate((choice_subj_sigmas, choice_latent_sigmas))
  elif isinstance(disrnn_config, disrnn.DisRnnConfig):
    params_disrnn = params['hk_disentangled_rnn']
    subject_embedding_size = 0
    update_input_names = disrnn_config.x_names
    # For update_sigmas: concatenate transposed reparameterized sigmas
    # Order of inputs to update nets: observations, latents
    update_obs_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_obs_sigma_params']
        )
    )
    update_latent_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_latent_sigma_params']
        )
    )
    update_sigmas = np.concatenate(
        (update_obs_sigmas_t, update_latent_sigmas_t), axis=1)
    choice_sigmas = np.array(
        disrnn.reparameterize_sigma(
            np.transpose(params_disrnn['choice_net_sigma_params'])
        )
    )
  else:
    raise ValueError(
        'plot_bottlenecks only supports DisRnnConfig and'
        ' MultisubjectDisRnnConfig.'
    )

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(params_disrnn['latent_sigma_params'])
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)
    latent_sigmas = latent_sigmas[latent_sigma_order]

    # Sort choice sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist.
    choice_sigma_order = np.concatenate(
        (
            np.arange(0, subject_embedding_size),
            subject_embedding_size + latent_sigma_order,
        ),
        axis=0,
    )
    choice_sigmas = choice_sigmas[choice_sigma_order]

    # Sort update sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist, then observations, then latents.
    non_latent_input_size = subject_embedding_size + disrnn_config.obs_size
    update_sigma_order = np.concatenate(
        (
            np.arange(0, non_latent_input_size, 1),
            non_latent_input_size + latent_sigma_order,
        ),
        axis=0,
    )
    update_sigmas = update_sigmas[latent_sigma_order, :]
    update_sigmas = update_sigmas[:, update_sigma_order]

  latent_names = np.arange(1, disrnn_config.latent_size + 1)
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  # Plot Latent Bottlenecks on axes[0]
  im1 = axes[0].imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
  im1.set_clim(vmin=0, vmax=1)
  axes[0].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  axes[0].set_xticks(ticks=[])
  axes[0].set_ylabel('Latent #', fontsize=medium)
  axes[0].set_title('Latent Bottlenecks', fontsize=large)

  # Plot Choice Bottlenecks on axes[1]
  # These bottlenecks apply to the inputs of the choice network:
  # [subject embeddings, latents]
  choice_input_dim = subject_embedding_size + disrnn_config.latent_size
  choice_input_names = np.concatenate((
      [f'SubjEmb {i+1}' for i in range(subject_embedding_size)],
      [f'Latent {i}' for i in latent_names]
  ))
  im2 = axes[1].imshow(np.swapaxes([1 - choice_sigmas], 0, 1), cmap='Oranges')
  im2.set_clim(vmin=0, vmax=1)
  axes[1].set_yticks(
      ticks=range(choice_input_dim), labels=choice_input_names, fontsize=small
  )
  axes[1].set_xticks(ticks=[])
  axes[1].set_ylabel('Choice Network Input', fontsize=medium)
  axes[1].set_title('Choice Network Bottlenecks', fontsize=large)

  # Plot Update Bottlenecks on axes[2]
  im3 = axes[2].imshow(1 - update_sigmas, cmap='Oranges')
  im3.set_clim(vmin=0, vmax=1)
  cbar = fig.colorbar(im3, ax=axes[2])
  # Y-axis corresponds to the target latent (sorted if sort_latents=True)
  cbar.ax.tick_params(labelsize=small)
  axes[2].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  # X-axis corresponds to the inputs to the update network:
  # [subject embeddings, observations, latents]
  xlabels = update_input_names + [f'Latent {i}' for i in latent_names]
  axes[2].set_xticks(
      ticks=range(len(xlabels)),
      labels=xlabels,
      rotation='vertical',
      fontsize=small,
  )
  axes[2].set_ylabel('Latent #', fontsize=medium)
  axes[2].set_xlabel('Update Network Inputs', fontsize=medium)
  axes[2].set_title('Update Network Bottlenecks', fontsize=large)
  fig.tight_layout()  # Adjust layout to prevent overlap
  return fig
