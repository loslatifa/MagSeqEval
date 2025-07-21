import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from feature_plot.pos_cmp_plot_pairs_from_csv import pos_cmp_plot_pairs_from_csv
def analyze_csv(file_path, save_dir, title_prefix):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(file_path)

    print(f"\n🟩 Analyzing file: {file_path}")
    print(df.head())

    # === 图1: 特征距离 vs 位置距离 ===
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x='Feature_Distance', y='Position_Distance')
    plt.title(f'{title_prefix}: Feature vs Position Distance')
    plt.xlabel('Feature Distance')
    plt.ylabel('Position Distance (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title_prefix}_scatter_dist.png'))
    plt.close()

    # === 图2: Error_i & Error_j 分布 ===
    plt.figure(figsize=(7, 5))
    sns.histplot(df[['Error_i', 'Error_j']].melt(value_name='Error')['Error'], bins=30, kde=True)
    plt.title(f'{title_prefix}: Error Distribution')
    plt.xlabel('Localization Error')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title_prefix}_error_hist.png'))
    plt.close()

    # === 图3: GT 与 Pred KDE 热力图 ===
    for suffix in ['GT', 'Pred']:
        x_i = df[f'{suffix}_i_x']
        y_i = df[f'{suffix}_i_y']
        x_j = df[f'{suffix}_j_x']
        y_j = df[f'{suffix}_j_y']

        plt.figure(figsize=(6, 5))
        sns.kdeplot(x=pd.concat([x_i, x_j]), y=pd.concat([y_i, y_j]), fill=True, cmap="viridis")
        plt.title(f'{title_prefix}: {suffix} Position KDE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_{suffix}_kde.png'))
        plt.close()

    # === 输出统计指标到 CSV & LOG ===
    stats = {}
    for col in ['Feature_Distance', 'Position_Distance', 'Error_i', 'Error_j']:
        stats[f'{col}_mean'] = df[col].mean()
        stats[f'{col}_median'] = df[col].median()
        stats[f'{col}_std'] = df[col].std()

    # 保存为 CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(save_dir, f'{title_prefix}_stats.csv'), index=False)

    # 保存为 log 文件
    log_path = os.path.join(save_dir, f'{title_prefix}_stats.log')
    with open(log_path, 'w') as f:
        f.write(f"Analysis Timestamp: {datetime.now()}\n")
        f.write(f"File analyzed: {file_path}\n\n")
        for k, v in stats.items():
            f.write(f"{k:<30}: {v:.4f}\n")

    print(f"📄 Stats saved to: {save_dir}")

# 使用示例（请替换为你的路径）
if __name__ == "__main__":
    df1 = 'sdcs_f4_furui-MTCP-600/evaluate_figure/mag_seq_error_analysis/feature_close_position_diff.csv'
    df2 = 'sdcs_f4_furui-MTCP-600/evaluate_figure/mag_seq_error_analysis/feature_diff_position_close.csv'
    sf1 = 'results/sdcs_f4_furui-MTCP-600/close_feat_far_pos'
    sf2 = 'results/sdcs_f4_furui-MTCP-600/far_feat_close_pos'
    bg_image_path="MagCode/MagLoc-data/SDCS_F4_FURUI/Huawei_MatePad/floor4.png"

    analyze_csv(df1, sf1, 'CloseFeatFarPos')
    analyze_csv(df2, sf2, 'FarFeatClosePos')
    pos_cmp_plot_pairs_from_csv(df1, sf1, "CloseFeatFarPos", bg_image_path)
