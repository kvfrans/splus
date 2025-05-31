import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import tqdm

from utils.fid import get_fid_network, fid_from_stats
get_fid_activations = get_fid_network() 

@jax.jit
def call_model(train_state, x, t, l):
    return train_state.call_model(x, t, l)

def plot_generated_images(FLAGS, train_state, x_shape, shard_data, data_sharding, vae_decode, step):
    with jax.spmd_mode('allow_all'):
        num_imgs = 64
        key = jax.random.PRNGKey(0)
        eps_key, label_key = jax.random.split(jax.random.fold_in(key, jax.process_index()), 2)
        x = jax.random.normal(eps_key, (num_imgs, *x_shape))
        labels = jax.random.randint(label_key, (num_imgs,), minval=0, maxval=FLAGS.model['num_classes'])
        x, labels = shard_data(x, labels)
        for i in range(FLAGS.model['denoise_timesteps']):
            t = shard_data(jnp.ones(num_imgs) * i / FLAGS.model['denoise_timesteps'])
            v = call_model(train_state, x, t, labels)
            x = x + v * (1 / FLAGS.model['denoise_timesteps'])
        img = vae_decode(x)
        img = img * 0.5 + 0.5
        img = jnp.clip(img, 0, 1)
        img = jax.experimental.multihost_utils.process_allgather(img)
        img = np.array(img)

        if jax.process_index() == 0:
            fig, axs = plt.subplots(8, 8, figsize=(30, 30))
            axs_flat = axs.flatten()
            for i in range(num_imgs):
                axs_flat[i].imshow(img[i])
                axs_flat[i].axis('off')
            wandb.log({'generated_images': wandb.Image(fig)}, step=step)
            plt.close(fig)


def eval_fid(FLAGS, train_state, x_shape, shard_data, vae_decode, truth_fid_stats, step):
    with jax.spmd_mode('allow_all'):
        x_shape = (FLAGS.batch_size, *x_shape)
        num_generations = 4096
        activations = []
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            x = jax.random.normal(eps_key, x_shape)
            labels = jax.random.randint(label_key, (x_shape[0],), 0, FLAGS.model.num_classes)
            x, labels = shard_data(x, labels)
            delta_t = 1.0 / FLAGS.model.denoise_timesteps
            for ti in range(FLAGS.model.denoise_timesteps):
                t = ti / FLAGS.model.denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((x_shape[0], ), t)
                t_vector = shard_data(t_vector)
                v = call_model(train_state, x, t_vector, labels)
                x = x + v * delta_t # Euler sampling.
            if FLAGS.model.use_stable_vae:
                x = vae_decode(x) # Image is in [-1, 1] space.
            x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
            x = jnp.clip(x, -1, 1)
            acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
            acts = jax.experimental.multihost_utils.process_allgather(acts)
            acts = np.array(acts)
            activations.append(acts)

        if jax.process_index() == 0:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            wandb.log({'fid': fid}, step=step)