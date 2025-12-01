import random

import einops
import matplotlib.pyplot as plt
import moviepy
import moviepy.editor as mpy
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian
from torch.distributions.uniform import Uniform
import torch.nn.functional as F


DimY = 2
y_min = torch.tensor([0.0, 0.0])
y_max = torch.tensor([1.0, 1.0])
step_size_init = 1e-3
step_size_final = 1e-3
step_size_power = 2
n_iters = 100
num_samples = 128


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def get_step_size(iteration):
    blend = iteration / (n_iters - 1)
    blend = blend ** step_size_power
    step_size = step_size_init + blend * (step_size_final - step_size_init)
    return step_size


def uniform_sample(num_samples, batch_size):
    lb = y_min.expand(batch_size, num_samples, DimY)
    ub = y_max.expand(batch_size, num_samples, DimY)
    return Uniform(lb, ub).sample()


def langevin_step(y_samples, dE_dys, step_size, use_ibc_style=True):
    # Independent draw for covariance.
    y_noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=y_samples.shape,
        device=y_samples.device,
    )

    # Perturb samples according to gradient and desired noise level.
    if use_ibc_style:
        # From google-research/ibc.
        delta_y = -step_size * (0.5 * dE_dys + y_noise)
    else:
        # Better formulation. See:
        # https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm  # noqa
        delta_y = -step_size * dE_dys + np.sqrt(2 * step_size) * y_noise

    # Shift current actions
    y_samples = y_samples + delta_y

    return y_samples


def gradient_wrt_act(ebm_net, x, ys):
    """Same as in google-research/ibc."""
    assert not torch.is_grad_enabled()

    def Ex_sum(ys):
        # Adapt trick from:
        # https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/5  # noqa
        energies = ebm_net(x, ys)
        return energies.sum()

    # WARNING: This may be rather slow.
    with torch.set_grad_enabled(True):
        dE_dys = jacobian(Ex_sum, ys)
    assert dE_dys.shape == ys.shape
    return dE_dys


def langevin_sample(ebm_net, x, num_samples, callback=None, *, use_ibc_style=True):
    assert not torch.is_grad_enabled()
    N = x.shape[0]

    # Draw initial samples.
    y_samples = uniform_sample(num_samples, batch_size=N)

    for i in range(n_iters + 1):
        is_last = i == n_iters

        if callback is not None:
            callback(i, y_samples)

        if not is_last:
            # Compute gradient.
            dE_dys = gradient_wrt_act(ebm_net, x, y_samples)
            # Compute step size given current iteration.
            step_size = get_step_size(i)
            # Produce next set of samples (driving towards typical set).
            y_samples = langevin_step(y_samples, dE_dys, step_size, use_ibc_style=use_ibc_style)

    return y_samples
def torch_log_normal_pdf(x, mu, std):
    """Log of multivariate normal distribution."""
    N, k = x.shape
    assert mu.shape == (k,)
    assert mu.shape == std.shape
    cov = torch.diag(std ** 2)
    inv_cov = torch.diag(1 / std ** 2)
    center = (x - mu).unsqueeze(-1)
    weighted = -0.5 * center.transpose(1, 2) @ inv_cov @ center
    log_denom = (2 * np.pi) * (k / 2) + std.sum()
    out = weighted - log_denom
    out = out.reshape(N)
    return out


class NormalEbm(nn.Module):
    """
    Provides an energy function that represents a normal distribution.
    """

    def __init__(self, y_mu, y_std):
        super().__init__()
        assert y_std.shape == y_mu.shape
        self._y_std = y_std
        self._y_mu = y_mu

    def forward(self, x, ys):
        N, K, _ = ys.shape
        ys = einops.rearrange(ys, "N K DimY -> (N K) DimY")
        # N.B. Incoprorate log into pdf computation so we avoid numeric
        # problems in autograd.
        probs = torch_log_normal_pdf(ys, self._y_mu, self._y_std)
        probs = einops.rearrange(probs, "(N K) -> N K 1", N=N)
        energies = -probs
        return energies


@torch.no_grad()
def plot_2d_ebm(ebm_net, x, grid_size=50, alpha=None, temperature=1.0):
    action_x = torch.linspace(y_min[0], y_max[0], steps=grid_size)
    action_y = torch.linspace(y_min[1], y_max[1], steps=grid_size)
    action_y_grid, action_x_grid = torch.meshgrid(action_x, action_y)

    action_xs = einops.rearrange(action_x_grid, "H W -> (H W)")
    action_ys = einops.rearrange(action_y_grid, "H W -> (H W)")
    ys = einops.rearrange([action_xs, action_ys], "C HW -> () HW C").to(x)
    Zs = ebm_net(x, ys)

    Z_grid = einops.rearrange(Zs, "1 N 1 -> N")
    # Show probabilities used for sampling.
    num_grid_samples = grid_size ** 2
    Z_grid = F.softmax(-Z_grid / temperature, dim=0)
    Z_grid = F.normalize(Z_grid, dim=0)
    Z_grid = einops.rearrange(Z_grid, "(H W) -> H W", H=grid_size)
    mesh = plt.pcolormesh(
        action_x_grid.numpy(),
        action_y_grid.numpy(),
        Z_grid.numpy(),
        cmap="magma",
        alpha=alpha,
        shading="auto",
    )
    return mesh


def mpl_figure_to_image(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    W, H = fig.canvas.get_width_height()
    C = 3
    data = data.reshape((H, W, C))
    return data


def plot_2d_ebm_callback(ebm_net, iteration, x, y_samples):
    N, _ = x.shape
    assert N == 1
    fig, ax = plt.subplots()
    mesh = plot_2d_ebm(ebm_net, x)
    plt.colorbar(mesh)
    plt.grid(False)
    plt.axis("scaled")
    y_samples = y_samples.squeeze(0)
    plt.scatter(
        y_samples[:, 0], y_samples[:, 1], marker="x", s=10, color="green"
    )
    plt.title(f"Iter {iteration} / {n_iters}")

    image = mpl_figure_to_image(fig)
    plt.close(fig)
    return image


def _repeat_last(images, *, count=5):
    # To ensure last frame is saved.
    assert len(images) > 0
    return images + [images[-1]] * count


def display_movie(clip):
    return moviepy.video.io.html_tools.ipython_display(
        clip, fps=10, loop=False, autoplay=False, rd_kwargs={"logger": None}
    )

@torch.no_grad()
def check_langevin_distribution(use_ibc_style):
    seed(0)

    mean = torch.tensor([0.3, 0.4])
    std = torch.as_tensor([0.1, 0.2])

    images = []
    iteration_interval = max(1, n_iters // 10)

    def callback(iteration, y_samples):
        if iteration % iteration_interval != 0:
            return
        image = plot_2d_ebm_callback(ebm_net, iteration, x, y_samples)
        images.append(image)

    ebm_net = NormalEbm(mean, std)
    ebm_net.eval()
    # Observation `x` isn't really used here.
    x = torch.zeros(size=(1, 0))
    ys = langevin_sample(
        ebm_net,
        x,
        num_samples,
        callback=callback,
        use_ibc_style=use_ibc_style,
    )
    ys = ys.squeeze(0)

    # Check statistics.
    std_actual, mean_actual = torch.std_mean(ys, dim=0, unbiased=False)
    mean_rel_error = ((mean - mean_actual) / mean).abs().max()
    std_rel_error = ((std - std_actual) / std).abs().max()
    print(f"mean_rel_error: {mean_rel_error}")
    print(f"std_rel_error: {std_rel_error}")
    return mpy.ImageSequenceClip(_repeat_last(images), fps=10)
clip = check_langevin_distribution(use_ibc_style=True)
clip.write_videofile("/tmp/langevin_ibc_true.mp4", logger=None)
display_movie(clip)
clip = check_langevin_distribution(use_ibc_style=False)
clip.write_videofile("/tmp/langevin_ibc_false.mp4", logger=None)
display_movie(clip)