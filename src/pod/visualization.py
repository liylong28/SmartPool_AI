from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_energy(
    energy_ratio: np.ndarray,
    cumulative_energy: np.ndarray,
    n_modes: int,
    save_path=None,
):

    energy_ratio = np.asarray(
        energy_ratio
    )

    cumulative_energy = np.asarray(
        cumulative_energy
    )

    modes = np.arange(
        1,
        len(energy_ratio) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        modes,
        energy_ratio * 100.0,
        alpha=0.7,
        label="Individual energy",
    )

    ax.plot(
        modes,
        cumulative_energy * 100.0,
        marker="o",
        markersize=3,
        label="Cumulative energy",
    )

    ax.axhline(
        99.0,
        linestyle="--",
        label="99% threshold",
    )

    ax.axvline(
        n_modes,
        linestyle="--",
        label=f"Selected modes = {n_modes}",
    )

    ax.set_xlabel(
        "POD mode"
    )

    ax.set_ylabel(
        "Energy (%)"
    )

    ax.set_title(
        "POD Energy Distribution"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def _prepare_grid(
    field: np.ndarray,
    ny: int,
    nx: int,
):

    field = np.asarray(
        field
    )

    if field.size != ny * nx:
        raise ValueError(
            f"Cannot reshape {field.size} values "
            f"into ({ny}, {nx})."
        )

    return field.reshape(
        ny,
        nx,
    )


def plot_mode(
    mode: np.ndarray,
    mode_index: int,
    ny: int = 180,
    nx: int = 300,
    coordinates=None,
    save_path=None,
):

    grid = _prepare_grid(
        mode,
        ny,
        nx,
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    if coordinates is not None:

        coordinates = np.asarray(
            coordinates
        )

        X = coordinates[:, 0].reshape(
            ny,
            nx,
        )

        Y = coordinates[:, 1].reshape(
            ny,
            nx,
        )

        im = ax.pcolormesh(
            X,
            Y,
            grid,
            shading="auto",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    else:

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
        )

    fig.colorbar(
        im,
        ax=ax,
        label="POD mode amplitude",
    )

    ax.set_title(
        f"POD Mode {mode_index + 1}"
    )

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_title: str = "",
    ny: int = 180,
    nx: int = 300,
    coordinates=None,
    save_path=None,
):

    original_grid = _prepare_grid(
        original,
        ny,
        nx,
    )

    reconstructed_grid = _prepare_grid(
        reconstructed,
        ny,
        nx,
    )

    error_grid = (
        reconstructed_grid
        - original_grid
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    if coordinates is not None:

        coordinates = np.asarray(
            coordinates
        )

        X = coordinates[:, 0].reshape(
            ny,
            nx,
        )

        Y = coordinates[:, 1].reshape(
            ny,
            nx,
        )

        plots = [
            original_grid,
            reconstructed_grid,
            error_grid,
        ]

        titles = [
            "Original",
            "POD reconstruction",
            "Reconstruction error",
        ]

        for ax, field, title in zip(
            axes,
            plots,
            titles,
        ):

            im = ax.pcolormesh(
                X,
                Y,
                field,
                shading="auto",
            )

            fig.colorbar(
                im,
                ax=ax,
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(title)

    else:

        plots = [
            original_grid,
            reconstructed_grid,
            error_grid,
        ]

        titles = [
            "Original",
            "POD reconstruction",
            "Reconstruction error",
        ]

        for ax, field, title in zip(
            axes,
            plots,
            titles,
        ):

            im = ax.imshow(
                field,
                origin="lower",
                aspect="auto",
            )

            fig.colorbar(
                im,
                ax=ax,
            )

            ax.set_title(title)

    fig.suptitle(
        sample_title
    )

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)
