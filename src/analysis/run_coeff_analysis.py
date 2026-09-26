from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .coeff_diagnostics import validate_inputs, build_beta_grid, coefficient_statistics, surface_diagnostics, feature_correlations
from .trend_probe import ridge_trend_diagnostics
from .visualization import plot_beta_vs_v, plot_beta_vs_h, plot_beta_map, plot_correlation_matrix, plot_coefficient_std

# ============================================================
# 参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="POD coefficient analysis before Trend-GP modeling"
    )
    parser.add_argument(
        "--vh",
        type=str,
        default="data/processed/VH.npy",
    )
    parser.add_argument(
        "--pod-dir",
        type=str,
        default="outputs/pod",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
    )
    return parser.parse_args()

# ============================================================
# 主程序
# ============================================================
def main():
    args = parse_args()
    vh_path = Path(args.vh)
    pod_dir = Path(args.pod_dir)
    output_dir = Path(args.output_dir)

    table_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    beta_v_dir = figure_dir / "beta_vs_v"
    beta_h_dir = figure_dir / "beta_vs_h"
    beta_map_dir = figure_dir / "beta_maps"
    residual_map_dir = figure_dir / "residual_maps"
    corr_dir = figure_dir / "feature_correlations"

    for folder in [table_dir, data_dir, beta_v_dir, beta_h_dir, beta_map_dir, residual_map_dir, corr_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. 加载
    # ========================================================
    VH = np.load(vh_path)
    beta = np.load(pod_dir / "pod_coeff.npy")
    energy_path = pod_dir / "energy_ratio.npy"
    if energy_path.exists():
        energy_ratio = np.load(energy_path)
    else:
        energy_ratio = None

    # ========================================================
    # 2. 基础检查
    # ========================================================
    VH, beta = validate_inputs(VH, beta)
    print("=" * 70)
    print("POD COEFFICIENT ANALYSIS")
    print("=" * 70)
    print(f"VH shape       : {VH.shape}")
    print(f"beta shape     : {beta.shape}")
    print(f"speed range    : {VH[:,0].min():.3f} ~ {VH[:,0].max():.3f}")
    print(f"depth range    : {VH[:,1].min():.3f} ~ {VH[:,1].max():.3f}")
    print()

    # ========================================================
    # 3. 构造二维参数网格
    # ========================================================
    speeds, depths, beta_grid = build_beta_grid(VH, beta, require_full_grid=True)
    print(f"n_speeds       : {len(speeds)}")
    print(f"n_depths       : {len(depths)}")
    print(f"beta_grid      : {beta_grid.shape}")
    print()

    # ========================================================
    # 4. 基本统计量
    # ========================================================
    stats_df = coefficient_statistics(beta=beta, energy_ratio=energy_ratio)
    stats_df.to_csv(table_dir / "mode_statistics.csv", index=False, encoding="utf-8-sig")

    # ========================================================
    # 5. 曲面诊断
    # ========================================================
    surface_df = surface_diagnostics(speeds=speeds, depths=depths, beta_grid=beta_grid)
    surface_df.to_csv(table_dir / "surface_diagnostics.csv", index=False, encoding="utf-8-sig")

    # ========================================================
    # 6. 物理特征相关性
    # ========================================================
    pearson_df, spearman_df = feature_correlations(VH, beta)
    pearson_df.to_csv(table_dir / "feature_corr_pearson.csv", index=False, encoding="utf-8-sig")
    spearman_df.to_csv(table_dir / "feature_corr_spearman.csv", index=False, encoding="utf-8-sig")

    # ========================================================
    # 7. Ridge趋势探针
    # ========================================================
    trend_df, residuals = ridge_trend_diagnostics(VH, beta)
    trend_df.to_csv(table_dir / "ridge_trend_probe.csv", index=False, encoding="utf-8-sig")
    np.save(data_dir / "ridge_residuals.npy", residuals)

    # ========================================================
    # 8. 合并所有核心诊断
    # ========================================================
    summary_df = (
        stats_df
        .merge(surface_df, on="mode", how="left")
        .merge(trend_df, on="mode", how="left")
    )
    summary_df.to_csv(table_dir / "diagnostic_summary.csv", index=False, encoding="utf-8-sig")

    # ========================================================
    # 9. beta-V/H可视化
    # ========================================================
    n_modes = beta.shape[1]
    for mode_idx in range(n_modes):
        mode_id = mode_idx + 1
        field = beta_grid[mode_idx]
        plot_beta_vs_v(
            speeds=speeds,
            depths=depths,
            field=field,
            mode_id=mode_id,
            save_path=beta_v_dir / f"beta_{mode_id:02d}_vs_V.png",
        )
        plot_beta_vs_h(
            speeds=speeds,
            depths=depths,
            field=field,
            mode_id=mode_id,
            save_path=beta_h_dir / f"beta_{mode_id:02d}_vs_H.png",
        )
        plot_beta_map(
            speeds=speeds,
            depths=depths,
            field=field,
            mode_id=mode_id,
            save_path=beta_map_dir / f"beta_{mode_id:02d}_VH_map.png",
        )

    # ========================================================
    # 10. Ridge残差V-H地图
    # ========================================================
    _, _, residual_grid = build_beta_grid(VH, residuals, require_full_grid=True)
    for mode_idx in range(n_modes):
        mode_id = mode_idx + 1
        plot_beta_map(
            speeds=speeds,
            depths=depths,
            field=residual_grid[mode_idx],
            mode_id=mode_id,
            save_path=residual_map_dir / f"residual_{mode_id:02d}_VH_map.png",
            title_prefix="Ridge residual",
        )

    # ========================================================
    # 11. 相关性矩阵
    # ========================================================
    plot_correlation_matrix(
        pearson_df,
        save_path=corr_dir / "pearson_correlation.png",
        title="Pearson correlation: physics features vs POD coefficients",
    )
    plot_correlation_matrix(
        spearman_df,
        save_path=corr_dir / "spearman_correlation.png",
        title="Spearman correlation: physics features vs POD coefficients",
    )
    plot_coefficient_std(stats_df, save_path=figure_dir / "coefficient_std.png")

    # ========================================================
    # 12. 终端摘要
    # ========================================================
    show_columns = [
        "mode",
        "pod_energy_ratio",
        "monotonicity_V",
        "monotonicity_H",
        "roughness_V",
        "roughness_H",
        "additive_R2",
        "ridge_LOO_R2",
        "residual_std_ratio",
        "maxV_holdout_NRMSE_std",
        "maxH_holdout_NRMSE_std",
    ]
    show_columns = [col for col in show_columns if col in summary_df.columns]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 4)

    print("=" * 70)
    print("TREND-GP PRE-MODELING DIAGNOSTIC")
    print("=" * 70)
    print(summary_df[show_columns].to_string(index=False))
    print()
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
