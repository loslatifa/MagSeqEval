import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def pos_cmp_plot(
    real_pos: np.ndarray,
    mag_pos: np.ndarray,
    save_path: str,
    title: str = "",
    x_invert: bool = False,
    y_invert: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    background_img: Optional[str] = None
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(real_pos[:, 0], real_pos[:, 1], c='red', marker='o', label='GT', s=80, edgecolors='k')
    ax.scatter(mag_pos[:, 0], mag_pos[:, 1], c='blue', marker='x', label='Pred', s=100)

    for i in range(len(real_pos)):
        ax.annotate(
            "", xy=(mag_pos[i, 0], mag_pos[i, 1]), xytext=(real_pos[i, 0], real_pos[i, 1]),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1)
        )

    for i, (gt, pred) in enumerate(zip(real_pos, mag_pos)):
        ax.text(gt[0] + 5, gt[1] + 5, f"GT_{i}", fontsize=8, color='darkred')
        ax.text(pred[0] + 5, pred[1] + 5, f"Pred_{i}", fontsize=8, color='navy')

    plt.title(title, fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if x_invert: ax.invert_xaxis()
    if y_invert: ax.invert_yaxis()

    background_img="./MagCode/MagLoc-data/SDCS_F4_FURUI/Huawei_MatePad/floor4.png"
    if background_img:
        try:
            img = plt.imread(background_img)
            ax.imshow(img, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], origin='upper')
        except Exception as e:
            print(f"Failed to load background image: {e}")

    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}_mag.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    xlim = (0, 1300)
    ylim = (0, 610)
    y_invert = True

    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_dir = os.path.join(csv_dir, csv_name)
    os.makedirs(save_dir, exist_ok=True)

    for idx, row in df.iterrows():
        real_pos = np.array([
            [row['GT_i_x'], row['GT_i_y']],
            [row['GT_j_x'], row['GT_j_y']]
        ])
        pred_pos = np.array([
            [row['Pred_i_x'], row['Pred_i_y']],
            [row['Pred_j_x'], row['Pred_j_y']]
        ])

        name = f"Sample_{int(row['Sample_i'])}_{int(row['Sample_j'])}"
        save_path = os.path.join(save_dir, name)

        title = f"GT vs Pred Comparison: {name}"
        pos_cmp_plot(
            real_pos=real_pos,
            mag_pos=pred_pos,
            save_path=save_path,
            title=title,
            x_invert=False,
            y_invert=y_invert,
            xlim=xlim,
            ylim=ylim,
            background_img=None  # Optional: set to path if needed
        )
        print(f"Saved: {save_path}_mag.png")


def find_all_csv(root_folder: str):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, file))
    return csv_files

def compare_sequences_from_csv(csv_path, npz_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 加载 CSV 和 npz 文件
    df = pd.read_csv(csv_path)
    data = np.load(npz_path)
    raw_seq = data['raw_seq']  # shape: (N, L, 3)

    assert raw_seq.ndim == 3 and raw_seq.shape[2] == 3, "raw_seq 应为 (N, L, 3)，代表 H/V/Norm"

    for idx, row in df.iterrows():
        i = int(row['Sample_i'])
        j = int(row['Sample_j'])

        seq_i = raw_seq[i]  # shape: (L, 3)
        seq_j = raw_seq[j]

        t = np.arange(seq_i.shape[0])

        h_i, v_i, n_i = seq_i[:, 0], seq_i[:, 1], seq_i[:, 2]
        h_j, v_j, n_j = seq_j[:, 0], seq_j[:, 1], seq_j[:, 2]

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        axs[0].plot(t, h_i, label=f'Sample {i}', color='blue')
        axs[0].plot(t, h_j, label=f'Sample {j}', color='orange')
        axs[0].set_ylabel('H Channel')
        axs[0].legend()

        axs[1].plot(t, v_i, label=f'Sample {i}', color='blue')
        axs[1].plot(t, v_j, label=f'Sample {j}', color='orange')
        axs[1].set_ylabel('V Channel')

        axs[2].plot(t, n_i, label=f'Sample {i}', color='blue')
        axs[2].plot(t, n_j, label=f'Sample {j}', color='orange')
        axs[2].set_ylabel('Norm Channel')
        axs[2].set_xlabel('Time Step')

        fig.suptitle(f'Sample {i} vs Sample {j}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f'pair_{i}_{j}.png'
        plt.savefig(os.path.join(save_dir, fname), dpi=300)
        plt.close()

        print(f"✅ Saved: {fname}")
