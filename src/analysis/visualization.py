from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _prepare_save_path(save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path

# ============================================================
# beta - V
# ============================================================
def plot_beta_vs_v(speeds, depths, field, mode_id, save_path):
    """
    固定H，观察beta随V变化。
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for ih, h in enumerate(depths):
        ax.plot(speeds, field[ih, :], marker="o", markersize=3, label=f"H={h:g}")
    ax.set_xlabel("Speed V (m/s)")
    ax.set_ylabel(rf"$\beta_{{{mode_id}}}$")
    ax.set_title(rf"$\beta_{{{mode_id}}}$ vs V (fixed H)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_path = _prepare_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# beta - H
# ============================================================
def plot_beta_vs_h(speeds, depths, field, mode_id, save_path):
    """
    固定V，观察beta随H变化。
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for iv, v in enumerate(speeds):
        ax.plot(depths, field[:, iv], marker="o", markersize=3, label=f"V={v:g}")
    ax.set_xlabel("Depth H (m)")
    ax.set_ylabel(rf"$\beta_{{{mode_id}}}$")
    ax.set_title(rf"$\beta_{{{mode_id}}}$ vs H (fixed V)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_path = _prepare_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# beta V-H参数地图
# ============================================================
def plot_beta_map(speeds, depths, field, mode_id, save_path, title_prefix="POD coefficient"):
    """
    V-H二维系数分布图。
    """
    V_grid, H_grid = np.meshgrid(speeds, depths)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(V_grid, H_grid, field, shading="auto")
    contour = ax.contour(V_grid, H_grid, field, levels=10)
    ax.clabel(contour, fontsize=7)
    fig.colorbar(mesh, ax=ax, label=rf"$\beta_{{{mode_id}}}$")
    ax.set_xlabel("Speed V (m/s)")
    ax.set_ylabel("Depth H (m)")
    ax.set_title(f"{title_prefix} {rf'$\beta_{{{mode_id}}}$'}")
    fig.tight_layout()
    save_path = _prepare_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 特征相关性图
# ============================================================
def plot_correlation_matrix(df: pd.DataFrame, save_path, title):
    """
    dataframe:
        mode | V | V2 | V4 | ...
    """
    values = df.drop(columns=["mode"]).to_numpy()
    feature_names = [c for c in df.columns if c != "mode"]
    modes = df["mode"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(values, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels([f"beta_{int(i)}" for i in modes])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    save_path = _prepare_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# beta尺度
# ============================================================
def plot_coefficient_std(stats_df, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    modes = stats_df["mode"]
    std = stats_df["std"]
    ax.bar(modes, std)
    ax.set_xlabel("POD mode")
    ax.set_ylabel("Coefficient standard deviation")
    ax.set_title("Scale of POD coefficients")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_path = _prepare_save_path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
