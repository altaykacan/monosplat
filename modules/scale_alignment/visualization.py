import pickle
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
        color: str = None,
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
    fig = plt.figure(dpi=300)
    # plt.title(title)
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
    plt.legend(bbox_to_anchor=(1.1, 1.05))
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
        ylabel: str = "Scale values",
        color: str = None,
        linewidth: float = None,
        show_raw_data: bool = True,
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
        plt.figure(dpi=300)
        plt.grid()
        # plt.title(title)
        for curr_scale, curr_frame_id, curr_label in zip(scales, frame_ids, labels):
            # curr_scale and curr_frame_id are both 1D lists
            scale_array = curr_scale.detach().cpu().numpy()
            moving_avg = centered_moving_average(scale_array, window_size)
            mean_scale = np.mean(scale_array)
            line = plt.plot(curr_frame_id, moving_avg, color=color, label=curr_label, alpha=1.0, linewidth=linewidth)
            if show_raw_data:
                plt.plot(curr_frame_id, scale_array, alpha=0.4, color=line[0].get_color(), linewidth=linewidth)

            plt.hlines(mean_scale, curr_frame_id[0], curr_frame_id[-1], linewidth=1.5, colors=line[0].get_color(), alpha=1.0, linestyles='dashed')

        plt.xlabel("Frames")
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right")
        plt.savefig(str(output_dir / filename) + ".png")
        plt.close()


def plot_dense_alignment_results(
        scales: Dict,
        frame_ids_for_hist: List[int],
        frame_scales_for_hist: Dict,
        scale_tensors_to_plot: Dict,
        tz_to_plot: Dict,
        std_to_plot: Dict,
        log_dir: Path,
        dataset: BaseDataset,
        ) -> None:
    ###
    # Code commented out that was used to save time when creating plots
    ###
    # # Pickle the input so we can use them later to adjust the plots
    # with open(log_dir / Path("scales.pickle"), "wb") as handle:
    #     pickle.dump(scales, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(log_dir / Path("frame_ids_for_hist.pickle"), "wb") as handle:
    #     pickle.dump(frame_ids_for_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(log_dir / Path("frame_scales_for_hist.pickle"), "wb") as handle:
    #     pickle.dump(frame_scales_for_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(log_dir / Path("scale_tensors_to_plot.pickle"), "wb") as handle:
    #     pickle.dump(scale_tensors_to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(log_dir / Path("tz_to_plot.pickle"), "wb") as handle:
    #     pickle.dump(tz_to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(log_dir / Path("log_dir.pickle"), "wb") as handle:
    #     pickle.dump(log_dir, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(log_dir / Path("dataset.pickle"), "wb") as handle:
    #     pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot the mean of scales for the different flow steps, scale plot
    labels = []
    frame_ids_for_plot = [] # lists of lists of integers
    frame_indices_for_plot = [] # needed for combined datasets
    scales_for_plot = [] # lists of 1D tensors

    # To ensure plot colors stay the same even when we plot them individually
    plot_idx = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for flow_step, scale_per_flow in scales.items():
        sorted_pairs = sorted(scale_per_flow.items()) # the keys are the target frame ids
        sorted_frame_ids = [pair[0] for pair in sorted_pairs]
        sorted_scales = torch.tensor([pair[1] for pair in sorted_pairs])

        # Useful for combined colmap datasets
        sorted_frame_indices = [dataset.frame_ids.index(frame_id) for frame_id in sorted_frame_ids]

        labels.append(f"Flow Step {flow_step}")
        frame_ids_for_plot.append(sorted_frame_ids)
        frame_indices_for_plot.append(sorted_frame_indices)
        scales_for_plot.append(sorted_scales)

        # Save current flow step, scale plot
        if isinstance(dataset, CombinedColmapDataset):
            # Combined datasets have jumps in the frame ids due to naming convention
            # need to look at the index of the said frame instead of the id when plotting
            save_scale_plot(
                scales_for_plot[-1],
                frame_indices_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_scale_plot_flow_step_{flow_step}",
                output_dir=log_dir,
                color=colors[plot_idx]
                )
        else:
            save_scale_plot(
                scales_for_plot[-1],
                frame_ids_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_scale_plot_flow_step_{flow_step}",
                output_dir=log_dir,
                color=colors[plot_idx]
                )
        plot_idx += 1

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
            output_dir=log_dir,
            linewidth=0.8,
            show_raw_data=False,
            )
    else:
        save_scale_plot(
            scales_for_plot,
            frame_ids_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_scale_plot_all",
            output_dir=log_dir,
            linewidth=0.8,
            show_raw_data=False,
            )

    # Plot the t_z values for the different flow steps, tz plot
    labels = []
    frame_ids_for_plot = [] # lists of lists of integers
    frame_indices_for_plot = [] # needed for combined datasets
    tz_for_plot = [] # lists of 1D tensors

    # To ensure plot colors stay the same even when we plot them individually
    plot_idx = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for flow_step, tz_per_flow in tz_to_plot.items():
        sorted_pairs = sorted(tz_per_flow.items()) # the keys are the target frame ids
        sorted_frame_ids = [pair[0] for pair in sorted_pairs]
        sorted_tz = torch.tensor([pair[1] for pair in sorted_pairs])

        # Frame indices are the index in the lists not the frame ids, useful for CombinedColmapDatasets
        sorted_frame_indices = [dataset.frame_ids.index(frame_id) for frame_id in sorted_frame_ids]

        labels.append(f"Flow step {flow_step}")
        frame_ids_for_plot.append(sorted_frame_ids)
        frame_indices_for_plot.append(sorted_frame_indices)
        tz_for_plot.append(sorted_tz)

        # Save current flow step
        if isinstance(dataset, CombinedColmapDataset):
            # Combined datasets have jumps in the frame ids due to naming convention
            # need to look at the index of the said frame instead of the id when plotting
            save_scale_plot(
                tz_for_plot[-1],
                frame_indices_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_tz_plot_flow_step_{flow_step}",
                output_dir=log_dir,
                ylabel="t_z values",
                color=colors[plot_idx],
                )
        else:
            save_scale_plot(
                tz_for_plot[-1],
                frame_ids_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_tz_plot_flow_step_{flow_step}",
                output_dir=log_dir,
                ylabel="t_z values",
                color=colors[plot_idx],
                )
        plot_idx += 1

    # Save all flow steps for tz plot
    if isinstance(dataset, CombinedColmapDataset):
        # Combined datasets have jumps in the frame ids due to naming convention
        # need to look at the index of the said frame instead of the id when plotting
        save_scale_plot(
            tz_for_plot,
            frame_indices_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_tz_plot_all",
            output_dir=log_dir,
            ylabel="t_z values"
            )
    else:
        save_scale_plot(
            tz_for_plot,
            frame_ids_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_tz_plot_all",
            output_dir=log_dir,
            ylabel="t_z values"
            )

    # Plot the standard deviations just like the t_z plots
    labels = []
    frame_ids_for_plot = [] # lists of lists of integers
    frame_indices_for_plot = [] # needed for combined datasets
    std_for_plot = [] # lists of 1D tensors

    # To ensure plot colors stay the same even when we plot them individually
    plot_idx = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for flow_step, std_per_flow in std_to_plot.items():
        sorted_pairs = sorted(std_per_flow.items()) # the keys are the target frame ids
        sorted_frame_ids = [pair[0] for pair in sorted_pairs]
        sorted_std = torch.tensor([pair[1] for pair in sorted_pairs])

        # Frame indices are the index in the lists not the frame ids, useful for CombinedColmapDatasets
        sorted_frame_indices = [dataset.frame_ids.index(frame_id) for frame_id in sorted_frame_ids]

        labels.append(f"Flow step {flow_step}")
        frame_ids_for_plot.append(sorted_frame_ids)
        frame_indices_for_plot.append(sorted_frame_indices)
        std_for_plot.append(sorted_std)

        # Save current flow step
        if isinstance(dataset, CombinedColmapDataset):
            # Combined datasets have jumps in the frame ids due to naming convention
            # need to look at the index of the said frame instead of the id when plotting
            save_scale_plot(
                std_for_plot[-1],
                frame_indices_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_std_plot_flow_step_{flow_step}",
                output_dir=log_dir,
                ylabel="Standard Deviation Values",
                color=colors[plot_idx],
                )
        else:
            save_scale_plot(
                std_for_plot[-1],
                frame_ids_for_plot[-1],
                labels[-1],
                title=f"Dense Scale Alignment Results - Flow Step {flow_step}",
                filename=f"dense_std_plot_flow_step_{flow_step}",
                output_dir=log_dir,
                ylabel="Standard Deviation Values",
                color=colors[plot_idx],
                )
        plot_idx += 1

    # Save all flow steps for tz plot
    if isinstance(dataset, CombinedColmapDataset):
        # Combined datasets have jumps in the frame ids due to naming convention
        # need to look at the index of the said frame instead of the id when plotting
        save_scale_plot(
            std_for_plot,
            frame_indices_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_std_plot_all",
            output_dir=log_dir,
            ylabel="Standard Deviation Values"
            )
    else:
        save_scale_plot(
            std_for_plot,
            frame_ids_for_plot,
            labels,
            title="Dense Scale Alignment Results",
            filename="dense_std_plot_all",
            output_dir=log_dir,
            ylabel="Standard Deviation Values"
            )

    # Plot histograms of scales for different flow steps
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
        for flow_step, scale_per_flow in frame_scales_for_hist.items():
            labels = []
            scales_for_histogram = [] # list of 1D tensors
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

            f = f.detach().cpu().squeeze().numpy()
            max_value = np.max(f[f > 0])
            min_value = np.min(f[f > 0])
            mean_value = np.mean(f[f > 0])
            filename = Path(f"cleaned_forward_scales_frame_{frame_id}_flow_step_{flow_step}_mean_{mean_value}.png")
            fig = plt.figure(dpi=300)
            ax = plt.axes()
            im = ax.imshow(f, cmap="viridis")
            plt.axis('off')

            # Colorbar code from: https://stackoverflow.com/a/56900830/17588877
            # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
            # You can change 0.01 to adjust the distance between the main image and the colorbar.
            # You can change 0.02 to adjust the width of the colorbar.
            # This practice is universal for both subplots and GeoAxes.
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)

            # plt.title(f"Frame {frame_id} - Flow Step {flow_step} Mean: {mean_value:.2f}")
            plt.savefig(str(output_dir / filename), bbox_inches="tight")
            plt.close()
