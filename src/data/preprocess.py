import os
import numpy as np
from dataset import WaveDataset

# ==========================
# 1. Tecplot点数据转二维网格
# ==========================
def reshape_wave(X, Y, Z):
    """
    将Tecplot POINT格式转换为二维波场
    输出: (180,300)
    """
    nx = len(np.unique(X))
    ny = len(np.unique(Y))
    wave = Z.reshape(ny, nx)
    return wave

# ==========================
# 2. 构建POD矩阵
# ==========================
def build_wave_matrix(dataset):
    VH = []
    wave_matrix = []
    coordinates = None

    for i, sample in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}")
        V = sample["V"]
        H = sample["H"]
        VH.append([V, H])

        wave = reshape_wave(sample["X"], sample["Y"], sample["Z"])
        # 保存坐标
        if coordinates is None:
            coordinates = np.vstack((sample["X"], sample["Y"])).T
        # 展平成54000
        wave_matrix.append(wave.flatten())

    VH = np.array(VH)
    wave_matrix = np.array(wave_matrix)
    return VH, wave_matrix, coordinates

# ==========================
# 3. 数据检查
# ==========================
def check_matrix(VH, wave_matrix):
    print("====================")
    print("Input parameter matrix:")
    print(VH.shape)
    print("Wave matrix:")
    print(wave_matrix.shape)
    print("Maximum wave height:", wave_matrix.max())
    print("Minimum wave height:", wave_matrix.min())
    print("Mean:", wave_matrix.mean())
    print("NaN number:", np.isnan(wave_matrix).sum())
    print("====================")

# ==========================
# 4. 保存
# ==========================
def save_processed_data(VH, wave_matrix, coordinates, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "VH.npy"), VH)
    np.save(os.path.join(save_dir, "wave_matrix.npy"), wave_matrix)
    np.save(os.path.join(save_dir, "coordinates.npy"), coordinates)
    print("Data saved!")

# ==========================
# main
# ==========================
if __name__ == "__main__":
    train_path = "data/raw/Train"
    save_path = "data/processed"

    # 读取数据
    dataset = WaveDataset(train_path)
    # 构造矩阵
    VH, wave_matrix, coordinates = build_wave_matrix(dataset)
    # 检查
    check_matrix(VH, wave_matrix)
    # 保存
    save_processed_data(VH, wave_matrix, coordinates, save_path)
