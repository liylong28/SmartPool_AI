from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pod import POD
from .reconstruction import (
    summarize_reconstruction,
    sample_relative_l2,
)
from .visualization import (
    plot_energy,
    plot_mode,
    plot_reconstruction,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="POD reduction for SmartPool AI"
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pod",
    )

    parser.add_argument(
        "--energy",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--n-modes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract mean wave field before POD.",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="svd",
        choices=[
            "svd",
            "snapshot",
        ],
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--max-mode-plots",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--n-reconstruction-plots",
        type=int,
        default=5,
    )

    return parser.parse_args()


def load_processed_data(
    processed_dir: Path,
):

    wave_path = (
        processed_dir
        / "wave_matrix.npy"
    )

    vh_path = (
        processed_dir
        / "VH.npy"
    )

    coordinate_path = (
        processed_dir
        / "coordinates.npy"
    )

    if not wave_path.exists():
        raise FileNotFoundError(
            wave_path
        )

    if not vh_path.exists():
        raise FileNotFoundError(
            vh_path
        )

    wave_matrix = np.load(
        wave_path
    ).astype(
        np.float64,
        copy=False,
    )

    VH = np.load(
        vh_path
    ).astype(
        np.float64,
        copy=False,
    )

    coordinates = None

    if coordinate_path.exists():

        coordinates = np.load(
            coordinate_path
        ).astype(
            np.float64,
            copy=False,
        )

    return (
        wave_matrix,
        VH,
        coordinates,
    )


def validate_dataset(
    wave_matrix,
    VH,
    coordinates,
    nx,
    ny,
):

    if wave_matrix.ndim != 2:
        raise ValueError(
            "wave_matrix must be 2D."
        )

    if VH.ndim != 2:
        raise ValueError(
            "VH must be 2D."
        )

    if wave_matrix.shape[0] != VH.shape[0]:
        raise ValueError(
            "wave_matrix and VH sample count mismatch."
        )

    expected_features = (
        nx * ny
    )

    if wave_matrix.shape[1] != expected_features:

        raise ValueError(
            f"Expected {expected_features} spatial nodes "
            f"({nx}x{ny}), but wave_matrix has "
            f"{wave_matrix.shape[1]}."
        )

    if coordinates is not None:

        if coordinates.shape != (
            expected_features,
            2,
        ):

            raise ValueError(
                "coordinates must have shape "
                f"({expected_features}, 2), "
                f"got {coordinates.shape}."
            )

    if not np.all(
        np.isfinite(wave_matrix)
    ):

        raise ValueError(
            "wave_matrix contains NaN/Inf."
        )

    if not np.all(
        np.isfinite(VH)
    ):

        raise ValueError(
            "VH contains NaN/Inf."
        )


def main():

    args = parse_args()

    processed_dir = Path(
        args.processed_dir
    )

    output_dir = Path(
        args.output_dir
    )

    figure_dir = (
        output_dir
        / "figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =====================================================
    # 1. 读取数据
    # =====================================================

    (
        wave_matrix,
        VH,
        coordinates,
    ) = load_processed_data(
        processed_dir
    )


    # =====================================================
    # 2. 检查
    # =====================================================

    validate_dataset(
        wave_matrix=wave_matrix,
        VH=VH,
        coordinates=coordinates,
        nx=args.nx,
        ny=args.ny,
    )

    print(
        "=" * 60
    )

    print(
        "POD DATA INFORMATION"
    )

    print(
        "=" * 60
    )

    print(
        f"wave_matrix shape : {wave_matrix.shape}"
    )

    print(
        f"VH shape          : {VH.shape}"
    )

    if coordinates is not None:

        print(
            f"coordinates shape : {coordinates.shape}"
        )

    print(
        f"wave min          : {wave_matrix.min():.8e}"
    )

    print(
        f"wave max          : {wave_matrix.max():.8e}"
    )

    print(
        f"wave mean         : {wave_matrix.mean():.8e}"
    )

    print()


    # =====================================================
    # 3. POD
    # =====================================================

    pod = POD(
        energy_threshold=args.energy,
        n_modes=args.n_modes,
        center=args.center,
        solver=args.solver,
    )

    coeff = pod.fit_transform(
        wave_matrix
    )

    reconstructed = (
        pod.inverse_transform(
            coeff
        )
    )


    # =====================================================
    # 4. 重构指标
    # =====================================================

    metrics = summarize_reconstruction(
        wave_matrix,
        reconstructed,
    )

    sample_errors = (
        sample_relative_l2(
            wave_matrix,
            reconstructed,
        )
    )


    # =====================================================
    # 5. 输出信息
    # =====================================================

    print(
        "=" * 60
    )

    print(
        "POD RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"solver              : {args.solver}"
    )

    print(
        f"center              : {args.center}"
    )

    print(
        f"energy threshold    : {args.energy:.6f}"
    )

    print(
        f"selected modes      : {pod.n_modes_}"
    )

    print(
        f"coeff shape         : {coeff.shape}"
    )

    print(
        "retained energy     : "
        f"{pod.cumulative_energy_[pod.n_modes_ - 1] * 100:.6f}%"
    )

    print(
        "global relative L2  : "
        f"{metrics['global_relative_l2_percent']:.6f}%"
    )

    print(
        "mean sample L2      : "
        f"{metrics['mean_sample_relative_l2_percent']:.6f}%"
    )

    print(
        "max sample L2       : "
        f"{metrics['max_sample_relative_l2_percent']:.6f}%"
    )

    print()


    # =====================================================
    # 6. 保存模型和数组
    # =====================================================

    pod.save(
        output_dir
        / "pod_model.npz"
    )

    np.save(
        output_dir
        / "pod_modes.npy",
        pod.modes_,
    )

    np.save(
        output_dir
        / "pod_coeff.npy",
        coeff,
    )

    np.save(
        output_dir
        / "pod_mean.npy",
        pod.mean_,
    )

    np.save(
        output_dir
        / "singular_values.npy",
        pod.singular_values_,
    )

    np.save(
        output_dir
        / "energy_ratio.npy",
        pod.energy_ratio_,
    )

    np.save(
        output_dir
        / "cumulative_energy.npy",
        pod.cumulative_energy_,
    )

    np.save(
        output_dir
        / "sample_reconstruction_error.npy",
        sample_errors,
    )


    # =====================================================
    # 7. 保存metadata
    # =====================================================

    metadata = {

        "n_samples":
            int(
                wave_matrix.shape[0]
            ),

        "n_features":
            int(
                wave_matrix.shape[1]
            ),

        "nx":
            int(args.nx),

        "ny":
            int(args.ny),

        "solver":
            args.solver,

        "center":
            bool(args.center),

        "energy_threshold":
            float(args.energy),

        "n_modes":
            int(
                pod.n_modes_
            ),

        "retained_energy":
            float(
                pod.cumulative_energy_[
                    pod.n_modes_ - 1
                ]
            ),

        **metrics,
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )


    # =====================================================
    # 8. 能量图
    # =====================================================

    plot_energy(
        energy_ratio=pod.energy_ratio_,
        cumulative_energy=pod.cumulative_energy_,
        n_modes=pod.n_modes_,
        save_path=(
            figure_dir
            / "pod_energy.png"
        ),
    )


    # =====================================================
    # 9. POD模态可视化
    # =====================================================

    n_mode_plots = min(
        pod.n_modes_,
        args.max_mode_plots,
    )

    for i in range(
        n_mode_plots
    ):

        plot_mode(
            mode=pod.modes_[i],
            mode_index=i,
            ny=args.ny,
            nx=args.nx,
            coordinates=coordinates,
            save_path=(
                figure_dir
                / f"pod_mode_{i + 1:02d}.png"
            ),
        )


    # =====================================================
    # 10. 找代表性重构案例
    # =====================================================

    n_reconstruction_plots = min(
        args.n_reconstruction_plots,
        wave_matrix.shape[0],
    )

    # 按误差从小到大排序
    order = np.argsort(
        sample_errors
    )

    if n_reconstruction_plots == 1:

        selected_samples = [
            order[len(order) // 2]
        ]

    else:

        # 从最好、中间、最差等位置均匀选择
        positions = np.linspace(
            0,
            len(order) - 1,
            n_reconstruction_plots,
        ).astype(int)

        selected_samples = (
            order[positions]
        )


    for sample_id in selected_samples:

        V = VH[
            sample_id,
            0,
        ]

        H = VH[
            sample_id,
            1,
        ]

        error = sample_errors[
            sample_id
        ]

        title = (
            f"sample={sample_id}, "
            f"V={V:.3f} m/s, "
            f"H={H:.2f} m, "
            f"L2={error:.4f}%"
        )

        plot_reconstruction(
            original=wave_matrix[
                sample_id
            ],
            reconstructed=reconstructed[
                sample_id
            ],
            sample_title=title,
            ny=args.ny,
            nx=args.nx,
            coordinates=coordinates,
            save_path=(
                figure_dir
                / (
                    f"reconstruction_"
                    f"{sample_id:03d}.png"
                )
            ),
        )


    print(
        "=" * 60
    )

    print(
        f"POD results saved to: {output_dir}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()
