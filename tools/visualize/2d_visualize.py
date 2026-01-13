import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def visualize_2d(latents_list, identifiers, output):
    """
    Visualizes latent vectors from multiple files in 2D using PCA,
    applying PCA separately to each dataset (matching original logic).

    Args:
        latents_list (list): A list of numpy arrays, each containing latent vectors.
        identifiers (list): A list of strings identifying each dataset (e.g., filenames).
    """
    if not latents_list or len(latents_list) != len(identifiers):
        print("Error: Mismatch between data and identifiers or empty input.")
        return

    # --- Prepare for plotting ---
    plt.figure(figsize=(12, 10))

    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x"]
    cmap = plt.get_cmap("tab10")

    for i, (latents, identifier) in enumerate(zip(latents_list, identifiers)):
        n_samples = latents.shape[0]
        if n_samples == 0:
            print(f"Warning: Skipping {identifier} as it contains no samples.")
            continue

        latents_flat = latents.reshape(n_samples, -1)

        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents_flat)
        print(
            f"Dataset '{identifier}': PCA completed. Explained variance ratio: {pca.explained_variance_ratio_}"
        )

        color = cmap(i % 10)
        marker = marker_list[i % len(marker_list)]

        plt.scatter(
            latents_2d[:, 0],
            latents_2d[:, 1],
            alpha=0.7,
            color=color,
            marker=marker,
            label=identifier,
            s=50,
        )

        step = 2
        for j, (x, y) in enumerate(latents_2d):
            if j % step == 0:
                plt.annotate(
                    str(j),
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    weight="bold",
                )

    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.title("2D PCA Visualization of Latent Vectors (Per-Dataset PCA)")
    legend = plt.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file_base = "2d_pca_visualization_multi_separate"

    if output != "" and not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    output_path = os.path.join(output, output_file_base)

    plt.savefig(f"{output_path}.png", dpi=300, bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
    print(f"Plots saved as {output_path}.png and {output_path}.pdf")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs="+", help="Paths to the .npy files containing latent vectors."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pca_output",
        help="Base name (without extension) for the output plot files. Default: %(default)s",
    )
    args = parser.parse_args()

    latents_list = []
    identifiers = []

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}. Skipping.")
            continue
        if not file_path.endswith(".npy"):
            print(f"Warning: File {file_path} is not a .npy file. Skipping.")
            continue

        try:
            latents = np.load(file_path)
            if latents.ndim < 2:
                print(f"Warning: Data in {file_path} must be at least 2D. Skipping.")
                continue

            identifier = os.path.splitext(os.path.basename(file_path))[0]

            latents_list.append(latents)
            identifiers.append(identifier)
            print(f"Loaded data from {file_path} with shape {latents.shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping.")

    if not latents_list:
        print("No valid files loaded. Exiting.")
        return

    visualize_2d(latents_list, identifiers, args.output)


if __name__ == "__main__":
    main()
