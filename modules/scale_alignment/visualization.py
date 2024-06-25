from pathlib import Path
from typing import List, Union, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.core.interfaces import BaseDataset
from modules.io.datasets import CombinedColmapDataset
from modules.scale_alignment.utils import centered_moving_average


def save_histogram(
        tensor_inputs: List[torch.Tensor],
        labels: List[str] = None,
        title: str = "Histogram Plots",
        filename: Union[Path, str] = "debug_hist",
        output_dir: Union[Path, str] = ".",
        bin_count: int=100,
        ) -> None:
    """
    Prepares `center`, `hist`, and `width` for using
    `plt.bar(center, hist, align='center', width=width)` and saves the histogram
    Expects the `tensor_input` to be a one dimensional torch tensor which is used
    to construct the histogram.
    """
    if not isinstance(tensor_inputs, list):
        tensor_inputs = [tensor_inputs]

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(filename, str):
        filename = Path(filename)

    if labels == None:
        labels = ["Histogram " + str(num) for num in range(len(tensor_inputs))]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig = plt.figure()
    plt.title(title)
    plt.grid()
    for tensor_input, label in zip(tensor_inputs, labels):
        tensor_input = tensor_input.detach().cpu().numpy()
        hist, bins = np.histogram(tensor_input, bins=bin_count)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        bars = plt.bar(center, hist, align="center", alpha=0.7, width=width, label=label)
        bar_color = bars[0].get_facecolor()

        # Find the bin with the highest count (mode) and plot a vertical line
        mode_index = np.argmax(hist)
        mode_value = center[mode_index]
        plt.axvline(x=mode_value, color=bar_color, linestyle='dashed', linewidth=1.5)

    plt.xlabel("Scale Value Bins")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(str(output_path) + ".png")
    plt.close()

    return None


def save_scale_plot(
        scales: List[torch.Tensor],
        frame_ids: List[List[int]],
        labels: List[str] = None,
        title: str = "Scale plot",
        filename: Union[Path, str]="scale_plot",
        output_dir: Union[Path, str]=".",
        window_size: int = 5,
        ) -> None:
        if not isinstance(scales, list):
            scales = [scales]
        if not isinstance(labels, list):
            labels = [labels]

        # If we have a single list and not a nested list we duplicate it
        if not isinstance(frame_ids[0], list):
            frame_ids = [frame_ids for num in range(len(scales))]

        if labels == None:
            pass
        if isinstance(labels, list) and len(labels) != len(scales):
            raise ValueError(f"The number of scale tensors ({len(scales)}) don't match the number of labels you provided ({len(labels)})!")

        if isinstance(filename, str):
            filename = Path(filename)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.grid()
        plt.title(title)
        for curr_scale, curr_frame_id, curr_label in zip(scales, frame_ids, labels):
            # curr_scale and curr_frame_id are both 1D lists
            scale_array = curr_scale.detach().cpu().numpy()
            moving_avg = centered_moving_average(scale_array, window_size)
            mean_scale = np.mean(scale_array)
            line = plt.plot(curr_frame_id, scale_array, label=curr_label, alpha=0.4)
            plt.plot(curr_frame_id, moving_avg, color=line[0].get_color(), alpha=1.0)
            plt.hlines(mean_scale, curr_frame_id[0], curr_frame_id[-1], linewidth=0.5, colors=line[0].get_color(), alpha=1.0, linestyles='dashed')

        plt.xlabel("Frames")
        plt.ylabel("Scale values")
        plt.legend()
        plt.savefig(str(output_dir / filename) + ".png")
        plt.close()


def plot_dense_alignment_results(
        scales: Dict,
        frame_ids_for_hist: List[int],
        frame_scales_for_hist: Dict,
        scale_tensors_to_plot: Dict,
        log_dir: Path,
        dataset: BaseDataset,
        ) -> None:

    # Plot the mean of scales for the different flow steps
    labels = []
    frame_ids_for_plot = [] # lists of lists of integers
    frame_indices_for_plot = [] # needed for combined datasets
    scales_for_plot = [] # lists of 1D tensors
    for flow_step, scale_per_flow in scales.items():
        sorted_pairs = sorted(scale_per_flow.items()) # the keys are the target frame ids
        sorted_frame_ids = [pair[0] for pair in sorted_pairs]
        sorted_frame_indices = [dataset.frame_ids.index(frame_id) for frame_id in sorted_frame_ids]
        sorted_scales = torch.tensor([pair[1] for pair in sorted_pairs])

        labels.append(f"Dense Scales with Flow Step {flow_step}")
        frame_ids_for_plot.append(sorted_frame_ids)
        frame_indices_for_plot.append(sorted_frame_indices)
        scales_for_plot.append(sorted_scales)

        # Save current flow step
        if isinstance(dataset, CombinedColmapDataset):
            # Combined datasets have jumps in the frame ids due to naming convention
            # need to look at the index of the said frame instead of the id when plotting
            save_scale_plot(
                scales_for_plot[-1],
                frame_indices_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_scale_plot_flow_step_{flow_step}",
                output_dir=log_dir
                )
        else:
            save_scale_plot(
                scales_for_plot[-1],
                frame_ids_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_scale_plot_flow_step_{flow_step}",
                output_dir=log_dir
                )

    # Save all flow steps
    if isinstance(dataset, CombinedColmapDataset):
        # Combined datasets have jumps in the frame ids due to naming convention
        # need to look at the index of the said frame instead of the id when plotting
        save_scale_plot(
            scales_for_plot,
            frame_indices_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_scale_plot_all",
            output_dir=log_dir
            )
    else:
        save_scale_plot(
            scales_for_plot,
            frame_ids_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_scale_plot_all",
            output_dir=log_dir
            )

    # Plot histograms of per-frame scales for different flow steps
    for curr_frame_id in frame_ids_for_hist:
        labels = []
        scales_for_histogram = [] # list of 1D tensors
        filtered_scales_for_histogram = [] # lists of 1D tensors to get nicer visualizations
        for flow_step, scale_per_flow in frame_scales_for_hist.items():
                curr_scales = scale_per_flow[curr_frame_id]["forward"]
                labels.append(f"Flow {flow_step} - Forward")
                scales_for_histogram.append(curr_scales)

                # Remove values deviating more than 3 times the std from the mean
                std = curr_scales.std()
                mean = curr_scales.mean()
                filtered_scales = curr_scales[torch.abs(curr_scales - mean) < 3 * std]
                filtered_scales_for_histogram.append(filtered_scales)

        save_histogram(
            scales_for_histogram,
            labels,
            title=f"Dense Scale Align. Frame {curr_frame_id} for Different Flow Steps",
            filename=f"dense_scale_hist_different_flow_steps_frame_{curr_frame_id}",
            output_dir=log_dir
            )

        save_histogram(
            filtered_scales_for_histogram,
            labels,
            title=f"Dense Scale Align. - Frame {curr_frame_id} for Different Flow Steps 3" + r'$\sigma$' + " thresholding",
            filename=f"dense_scale_filtered_hist_different_flow_steps_frame_{curr_frame_id}",
            output_dir=log_dir
            )

    # Plot histograms of different frames for a given flow step
    for flow_step, scale_per_flow in frame_scales_for_hist.items():
        labels = []
        scales_for_histogram = [] # list of 1D tensors
        for curr_frame_id, curr_scales in scale_per_flow.items():
            if len(curr_scales["forward"]) == 0:
                continue
            labels.append(f"Frame {curr_frame_id} - Forward")
            scales_for_histogram.append(curr_scales["forward"])

        save_histogram(
            scales_for_histogram,
            labels,
            title=f"Dense Scale Alignment Histograms - Different Frames for Flow Step {flow_step}",
            filename=f"dense_scale_hist_flow_step_{flow_step}",
            output_dir=log_dir
            )

    # Plot histograms for same frames backward and forward for different flow steps
    for curr_frame_id in frame_ids_for_hist:
        labels = []
        scales_for_histogram = [] # list of 1D tensors
        for flow_step, scale_per_flow in frame_scales_for_hist.items():
            curr_scales = scale_per_flow[curr_frame_id]
            if len(curr_scales["forward"]) == 0:
                continue

            labels.append(f"Flow Step {flow_step} - Forward")
            labels.append(f"Flow Step {flow_step} - Backward")
            scales_for_histogram.append(curr_scales["forward"])
            scales_for_histogram.append(curr_scales["backward"])

            save_histogram(
                scales_for_histogram,
                labels,
                title=f"Dense Scale Alignment Histograms - Frame {curr_frame_id}",
                filename=f"dense_scale_hist_frame_forward_backward_frame_{curr_frame_id}_flow_step_{flow_step}",
                output_dir=log_dir
                )

    # Plot the scale tensors
    output_dir = log_dir / Path("tensors")
    output_dir.mkdir(exist_ok=True, parents=True)
    for flow_step, scales_per_frame in scale_tensors_to_plot.items():
        for frame_id, scale_tensors in scales_per_frame.items():
            f = scale_tensors["forward"]
            b = scale_tensors["backward"]

            filename = Path(f"cleaned_forward_scales_frame_{frame_id}_flow_step_{flow_step}.png")
            f = f.detach().cpu().squeeze().numpy()
            max_value = np.max(f[f > 0])
            min_value = np.min(f[f > 0])
            mean_value = np.mean(f[f > 0])
            plt.figure()
            plt.imshow(f, cmap="viridis")
            plt.axis('off')
            plt.title(f"Frame {frame_id} - Flow Step {flow_step} Mean: {mean_value:.2f}")
            plt.colorbar(shrink=0.7)
            plt.savefig(str(output_dir / filename))
            plt.close()
