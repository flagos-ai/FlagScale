"""
Usage:
    python pca_3d_visualizer.py latent1.npy latent2.npy ...

Example:
    python pca_3d_visualizer.py latents_run1.npy latents_run2.npy
"""

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    from matplotlib.colormaps import get_cmap
except ImportError:
    from matplotlib.pyplot import get_cmap


def save_3d_animation(
    fig,
    ax,
    output,
    filename="3d_pca_visualization.html",
    elev=20,
    azim_range=(0, 360),
    frame_interval=4,
    delay=50,
    embed_limit_mb=50,
):
    """
    Saves the 3D figure as a self-contained HTML animation file.

    Args:
        fig (matplotlib.figure.Figure): The figure object containing the 3D axis.
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The 3D axis object.
        filename (str): Output HTML file name. Defaults to "3d_pca_visualization.html".
        elev (float): Elevation angle for the view. Defaults to 20.
        azim_range (tuple): Azimuth angle range (start, end) for the animation. Defaults to (0, 360).
        frame_interval (float): Increment between frames for azimuth. Defaults to 4.
        delay (int): Delay between frames in milliseconds. Defaults to 50.
        embed_limit_mb (float): Max size (MB) for embedded HTML animation. Defaults to 50.
                                Set to None to not modify the default limit.
    """
    original_limit = plt.rcParams.get("animation.embed_limit", 20.0)
    if embed_limit_mb is not None:
        print(
            f"Temporarily increasing animation embed limit from {original_limit} MB to {embed_limit_mb} MB."
        )
        plt.rcParams["animation.embed_limit"] = embed_limit_mb

    def _animate(angle):
        ax.view_init(elev=20, azim=angle)

    frames = np.arange(azim_range[0], azim_range[1] + frame_interval / 2, frame_interval)
    rot_animation = animation.FuncAnimation(fig, _animate, frames=frames, interval=50)

    print(f"Saving animation to {filename}...")
    # filename=f"{output}/{filename}"
    output_path = os.path.join(output, filename)
    try:
        with open(output_path, "w") as f:
            f.write(rot_animation.to_jshtml())
        print(f"Animation saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        # --- Restore original animation embed limit ---
        if embed_limit_mb is not None:
            print(f"Restoring original animation embed limit to {original_limit} MB.")
            plt.rcParams["animation.embed_limit"] = original_limit


def visualize_3d(latent_datasets, output, html_embed_limit=50):
    """
    Visualizes multiple 3D PCA latent vector trajectories in a single plot.

    Args:
        latent_datasets (list of tuples): Each tuple is (latents_3d, n_steps, label, series_color)
            - latents_3d: PCA-reduced numpy array (n_samples, 3)
            - n_steps: Number of samples in the dataset
            - label: Dataset label for legend and text
            - series_color: Fixed color for the data series (e.g., 'blue', 'red')
        save_html (bool): Whether to save an interactive HTML animation. Defaults to False.
        html_embed_limit (float): Embed limit for HTML animation in MB. Defaults to 50.
    """
    if not latent_datasets:
        print("No datasets provided for visualization.")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    fig.suptitle(
        "3D PCA Trajectory Visualization of Latent Spaces", fontsize=16, fontweight="bold", y=0.98
    )

    explanation_lines = []
    start_end_colors = get_cmap("tab10", max(1, len(latent_datasets) * 2))
    start_end_color_indices = iter(range(max(1, len(latent_datasets) * 2)))

    for i, (_, _, label, _) in enumerate(latent_datasets):
        try:
            start_idx = next(start_end_color_indices)
            end_idx = next(start_end_color_indices)
        except StopIteration:
            start_end_color_indices = iter(range(max(1, len(latent_datasets) * 2)))
            start_idx = next(start_end_color_indices)
            end_idx = next(start_end_color_indices)

        start_color_val = start_end_colors(start_idx)
        end_color_val = start_end_colors(end_idx)
        start_hex = f"#{int(start_color_val[0] * 255):02x}{int(start_color_val[1] * 255):02x}{int(start_color_val[2] * 255):02x}"
        end_hex = f"#{int(end_color_val[0] * 255):02x}{int(end_color_val[1] * 255):02x}{int(end_color_val[2] * 255):02x}"

        explanation_lines.append(f"Start ({label}): {start_hex} Circle (○)")
        explanation_lines.append(f"End ({label}): {end_hex} Square (■)")

    explanation_text = "\n".join(explanation_lines)
    fig.text(
        0.5,
        0.92,
        explanation_text,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # --- Reset color index for actual plotting ---
    start_end_color_indices_plot = iter(range(max(1, len(latent_datasets) * 2)))

    # Get fixed colors for each series
    series_colors = get_cmap("tab10", len(latent_datasets))
    if len(latent_datasets) == 1:
        series_colors_list = [series_colors(0)]
    else:
        series_colors_list = [series_colors(i) for i in range(len(latent_datasets))]

    for i, (latents_3d, n_steps, label, _) in enumerate(latent_datasets):
        series_color = series_colors_list[i % len(series_colors_list)]

        # --- Plot main scatter points (fixed color) ---
        # Removed c= and cmap=, used color= parameter
        # scatter = ax.scatter(
        #     latents_3d[:, 0],
        #     latents_3d[:, 1],
        #     latents_3d[:, 2],
        #     color=series_color,
        #     alpha=0.7,
        #     s=50,
        #     label=label,
        #     picker=True,
        # )

        # Plot connecting lines (same fixed color)
        for j in range(n_steps - 1):
            ax.plot(
                [latents_3d[j, 0], latents_3d[j + 1, 0]],
                [latents_3d[j, 1], latents_3d[j + 1, 1]],
                [latents_3d[j, 2], latents_3d[j + 1, 2]],
                color=series_color,
                alpha=0.4,
                linewidth=1,  # <-- Use fixed color
            )

        # --- Plot start and end points ---
        try:
            start_idx = next(start_end_color_indices_plot)
            end_idx = next(start_end_color_indices_plot)
        except StopIteration:
            start_end_color_indices_plot = iter(range(max(1, len(latent_datasets) * 2)))
            start_idx = next(start_end_color_indices_plot)
            end_idx = next(start_end_color_indices_plot)

        start_color = start_end_colors(start_idx)
        end_color = start_end_colors(end_idx)

        ax.scatter(
            latents_3d[0, 0],
            latents_3d[0, 1],
            latents_3d[0, 2],
            c=[start_color],
            s=120,
            marker="o",
            edgecolors="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            latents_3d[-1, 0],
            latents_3d[-1, 1],
            latents_3d[-1, 2],
            c=[end_color],
            s=120,
            marker="s",
            edgecolors="black",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12)
    ax.set_zlabel("Principal Component 3 (PC3)", fontsize=12)

    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10)

    # --- Adjust layout ---
    plt.tight_layout(rect=[0, 0, 0.85, 0.90])

    # --- Set view to ensure trajectory runs left-to-right on x-axis ---
    # This makes the first PC (x-axis) the primary horizontal direction.
    # Adjust elev and azim as needed for best visual.
    ax.view_init(elev=20, azim=-90)

    if output != "" and not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    save_3d_animation(
        fig,
        ax,
        output,
        filename="3d_pca_visualization.html",
        elev=20,
        azim_range=(-90, 270),
        frame_interval=4,
        delay=50,
        embed_limit_mb=html_embed_limit,
    )

    plt.savefig(f"{output}/3d_pca_visualization_multiple_latents.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output}/3d_pca_visualization_multiple_latents.pdf", bbox_inches="tight")

    plt.show()


def main():
    """Main function to parse arguments and run visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize latent vectors from multiple .npy files using 3D PCA.",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve newlines in help
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Paths to the .npy files containing latent vectors.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pca_output",
        help="Base name (without extension) for the output plot files. Default: %(default)s",
    )
    parser.add_argument(
        "--html-embed-limit",
        type=float,
        default=50,
        help="Maximum size (in MB) for the embedded HTML animation. Default is 50.",
    )

    args = parser.parse_args()

    latent_datasets = []
    predefined_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        try:
            latents = np.load(file_path)
            print(f"Loaded {file_path} with shape {latents.shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping.")
            continue

        n_steps = latents.shape[0]
        # Flatten latent vectors for PCA
        latents_flat = latents.reshape(n_steps, -1)
        print(f"  Flattened to shape {latents_flat.shape}")

        # Perform 3D PCA
        pca = PCA(n_components=3)
        latents_3d = pca.fit_transform(latents_flat)
        print(
            f"  After PCA to 3D, shape is {latents_3d.shape}. "
            f"Explained variance ratio: {pca.explained_variance_ratio_}"
        )

        label = os.path.splitext(os.path.basename(file_path))[0]

        series_color = predefined_colors[i % len(predefined_colors)]

        latent_datasets.append((latents_3d, n_steps, label, series_color))

    if latent_datasets:
        visualize_3d(latent_datasets, output=args.output, html_embed_limit=args.html_embed_limit)
    else:
        print("No valid datasets were loaded. Exiting.")


if __name__ == "__main__":
    main()
