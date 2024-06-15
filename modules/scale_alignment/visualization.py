from pathlib import Path
from typing import List, Union, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt


def save_histogram(
        tensor_inputs: List[torch.Tensor],
        labels: List[str] = None,
        title: str = "Histogram Plots",
        filename: Union[Path, str] = "debug_hist",
        output_dir: Union[Path, str] = ".",
        bin_count: int=100
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
        output_dir: Union[Path, str]="."
        ) -> None:
        if not isinstance(scales, list):
            scales = [scales]

        # If we have a single list and not a nested list we duplicate it
        if not isinstance(frame_ids[0], list):
            frame_ids = [frame_ids for num in range(len(scales))]

        if labels == None:
            labels = ["Scales " + str(num) for num in range(len(scales))]
        if len(labels) != len(scales):
            raise ValueError(f"The number of scale tensors ({len(scales)}) don't match the number of labels you provided ({len(labels)})!")

        if isinstance(filename, str):
            filename = Path(filename)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.grid()
        plt.title(title)
        max_frames = 0
        for curr_scale, curr_frame_ids, curr_label in zip(scales, frame_ids, labels):
            # TODO plot the means of each curve with the same color as the curve for  the same flow step
            scale_array = curr_scale.detach().cpu().numpy()
            mean_scale = np.mean(scale_array)
            line = plt.plot(curr_frame_ids, scale_array, label=curr_label, alpha=0.7)
            plt.hlines(mean_scale,min(curr_frame_ids), max(curr_frame_ids), linewidth=0.5, colors=line[0].get_color(), linestyles='dashed')

            # Keep track of the most amount of frame ids for nice plotting
            if len(curr_frame_ids) > max_frames:
                max_frames = len(curr_frame_ids)

        plt.xticks(range(0,max_frames))
        plt.xlabel("Frame IDs")
        plt.ylabel("Scale values")
        plt.legend()
        plt.savefig(str(output_dir / filename) + ".png")
        plt.close()


def plot_dense_alignment_results(scales: Dict, frame_id_for_hist: int, frame_scales_for_hist: Dict, log_dir: Path) -> None:
    # Plot the mean of scales for the different flow steps
    labels = []
    frame_ids_for_plot = [] # lists of lists of integers
    scales_for_plot = [] # lists of 1D tensors
    for flow_step, scale_per_flow in scales.items():
        sorted_pairs = sorted(scale_per_flow.items()) # the keys are the target frame ids
        sorted_frame_ids = [pair[0] for pair in sorted_pairs]
        sorted_scales = torch.tensor([pair[1] for pair in sorted_pairs])

        labels.append(f"Dense Scales with Flow Step {flow_step}")
        frame_ids_for_plot.append(sorted_frame_ids)
        scales_for_plot.append(sorted_scales)

    save_scale_plot(
        scales_for_plot,
        frame_ids_for_plot,
        labels,
        title="Dense Scale Alignment Results",
        filename="dense_scale_plot",
        output_dir=log_dir / Path("plots")
        )

    # Plot histograms of per-frame scales for different flow steps
    labels = []
    scales_for_histogram = [] # list of 1D tensors
    for flow_step, scale_per_flow in frame_scales_for_hist.items():
        curr_scales = scale_per_flow[frame_id_for_hist]
        if len(curr_scales["forward"]) == 0:
            continue
        labels.append(f"Frame {frame_id_for_hist}, flow {flow_step} - Forward")
        scales_for_histogram.append(curr_scales["forward"])

    save_histogram(
        scales_for_histogram,
        labels,
        title=f"Dense Scale Alignment - Frame {frame_id_for_hist} for Different Flow Steps",
        filename="dense_scale_hist_different_flow_steps",
        output_dir=log_dir / Path("plots")
        )

    # Plot histograms of different frames for a given flow step
    for flow_step, scale_per_flow in frame_scales_for_hist.items():
        labels = []
        scales_for_histogram = [] # list of 1D tensors
        for curr_frame_id, curr_scales in scale_per_flow.items():
            if len(curr_scales["forward"]) == 0:
                continue
            labels.append(f"Frame {curr_frame_id}, flow {flow_step} - Forward")
            scales_for_histogram.append(curr_scales["forward"])

        save_histogram(
            scales_for_histogram,
            labels,
            title=f"Dense Scale Alignment Histograms - Different Frames for Flow Step {flow_step}",
            filename=f"dense_scale_hist_flow_step_{flow_step}",
            output_dir=log_dir / Path("plots")
            )

    # Plot histograms for same frames backward and forward for different flow steps
    for flow_step, scale_per_flow in frame_scales_for_hist.items():
        labels = []
        scales_for_histogram = [] # list of 1D tensors
        curr_scales = scale_per_flow[frame_id_for_hist]
        if len(curr_scales["forward"]) == 0:
            continue

        labels.append(f"Flow Step {flow_step} - Forward")
        labels.append(f"Flow Step {flow_step} - Backward")
        scales_for_histogram.append(curr_scales["forward"])
        scales_for_histogram.append(curr_scales["backward"])

        save_histogram(
            scales_for_histogram,
            labels,
            title=f"Dense Scale Alignment Histograms - Frame {frame_id_for_hist}, Flow Step {flow_step}",
            filename=f"dense_scale_hist_frame_{frame_id_for_hist}_flow_step_{flow_step}_forward_backward",
            output_dir=log_dir / Path("plots")
            )