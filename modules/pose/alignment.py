"""Contains functions to align two trajectories"""
from typing import Tuple

import torch
import numpy as np

# Code taken from evo https://github.com/MichaelGrupp/evo/blob/9ecf31f3ffbba15d6521d767dd654cbf0a86732f/evo/core/geometry.py
def umeyama_alignment(x: torch.Tensor, y: torch.Tensor,
                      with_scale: bool = False) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor

    License:
    author: Michael Grupp

    This file is part of evo (github.com/MichaelGrupp/evo).

    evo is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    evo is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with evo.  If not, see <http://www.gnu.org/licenses/>.
    """
    # Adapted to accept and return torch tensors, the logic is the same
    device = x.device
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    if x.shape != y.shape:
        raise ValueError("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise RuntimeError("Degenerate covariance rank, "
                           "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    # Return torch tensors on the same device
    r = torch.from_numpy(r).double().to(device)
    t = torch.from_numpy(t).double().view(3, 1).to(device) # reshape for broadcasting
    c = float(c) # python floats are float64's, pytorch floats are float32's

    return r, t, c
