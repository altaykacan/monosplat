from typing import List, Union
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.core.constants import TRAJECTORY_COLORS
from modules.eval.utils import get_dummy_stamps
from modules.eval.tum_rgbd_tools.evaluate_ate import plot_traj


def save_traj(trajectories: List[torch.Tensor], labels: List[str], filename: Union[str, Path], output_dir: Union[str, Path] = ".", show_diff: bool = False) -> None:
    """Plots and saves trajectories. Trajectories should be given as `[3, num_frames]` tensors."""
    if show_diff and len(trajectories) != 2:
        raise RuntimeError(f"If you want to plot the differences between two trajectories, you need to provide only two trajectories, you gave {len(trajectories)}!")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        stamps = get_dummy_stamps(traj) # [N]
        traj_np = traj.detach().cpu().numpy()
        stamps_np = stamps.detach().cpu().numpy()
        color = TRAJECTORY_COLORS[i]

        # Function expects trajectories as [N, 3] so we need to transpose
        plot_traj(ax, stamps_np, traj_np.T, "-", color, label)

    if show_diff:
        label = "difference"
        for (x1, y1, z1), (x2, y2, z2) in zip(trajectories[0].T, trajectories[1].T):
            ax.plot([x1,x2],[y1,y2],'-',color="red",label=label, linewidth=0.5)
            label="" # resets the label for next lines

    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(Path(output_dir) / Path(filename), dpi=90)
    plt.close()




