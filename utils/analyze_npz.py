import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Literal
from .seq_dist_compute import combined_feature_distance_norm_only, combined_feature_distance_all_channels

def analyze_npz(
    npz_path: str,
    save_dir: str,
    top_k: int = 100,
    gt_threshold: float = 15.0,
    pred_threshold: float = 5,
    feat_threshold: float = 1000.0,
    distance_type: Literal["euclidean", "norm_only", "all_channels"] = "euclidean"
):
    os.makedirs(save_dir, exist_ok=True)
    data = np.load(npz_path)

    raw_seq = data['raw_seq']  # shape: (N, 3, T)
    gt_pos = data['ground_truth_positions']
    pred_pos = data['prediction_positions']
    
    N = len(raw_seq)
    idx_i, idx_j = np.triu_indices(N, k=1)

    # === 自定义特征距离矩阵 ===
    if distance_type == "euclidean":
        feat_flat = raw_seq.reshape(N, -1)
        feat_dist = cdist(feat_flat, feat_flat, metric='euclidean')

    elif distance_type == "norm_only":
        feat_dist = np.zeros((N, N))
        for i, j in zip(idx_i, idx_j):
            d = combined_feature_distance_norm_only(raw_seq[i], raw_seq[j])
            feat_dist[i, j] = feat_dist[j, i] = d

    elif distance_type == "all_channels":
        feat_dist = np.zeros((N, N))
        for i, j in zip(idx_i, idx_j):
            d = combined_feature_distance_all_channels(
                raw_seq[i][0], raw_seq[i][1], raw_seq[i][2],
                raw_seq[j][0], raw_seq[j][1], raw_seq[j][2]
            )
            feat_dist[i, j] = feat_dist[j, i] = d
    else:
        raise ValueError(f"Unsupported distance_type: {distance_type}")

    # === 构建数据表 ===
    df = pd.DataFrame({
        'Sample_i': idx_i,
        'Sample_j': idx_j,
        'Feature_Distance': feat_dist[idx_i, idx_j],
        'GT_Position_Distance': np.linalg.norm(gt_pos[idx_i] - gt_pos[idx_j], axis=1),
        'Pred_Position_Distance': np.linalg.norm(pred_pos[idx_i] - pred_pos[idx_j], axis=1),
        'GT_i_x': gt_pos[idx_i, 0],
        'GT_i_y': gt_pos[idx_i, 1],
        'GT_j_x': gt_pos[idx_j, 0],
        'GT_j_y': gt_pos[idx_j, 1],
        'Pred_i_x': pred_pos[idx_i, 0],
        'Pred_i_y': pred_pos[idx_i, 1],
        'Pred_j_x': pred_pos[idx_j, 0],
        'Pred_j_y': pred_pos[idx_j, 1],
    })

    '''# === Top-K 最相似序列 ===
    topk_df = df.sort_values(by='Feature_Distance', ascending=True).head(top_k)
    topk_df.to_csv(os.path.join(save_dir, f'feature_close_pairs_{distance_type}.csv'), index=False)'''

    # === 特征相近但GT远、Pred近的混淆对 ===
    confusing_df = df[
        (df['Feature_Distance'] < np.percentile(df['Feature_Distance'], 10)) &
        (df['GT_Position_Distance'] > gt_threshold) &
        (df['Pred_Position_Distance'] < pred_threshold)
    ]
    confusing_all = df[(df['GT_Position_Distance'] > gt_threshold) &
        (df['Pred_Position_Distance'] < pred_threshold)]
    
    confusing_all.to_csv(os.path.join(save_dir, f'gt_far_pred_close_{distance_type}.csv'), index=False)
    confusing_df.to_csv(os.path.join(save_dir, f'feature_close_gt_far_pred_close_{distance_type}.csv'), index=False)
    confusing_strict = confusing_all[confusing_all['Feature_Distance'] < feat_threshold]
    confusing_strict.to_csv(os.path.join(save_dir, f'confusing_close_features_under_{feat_threshold}_{distance_type}.csv'), index=False)
    print(f"[{os.path.basename(npz_path)}] ✔ {distance_type} 分析完成：")
    #print(f"  Top-K 最相似对: {len(topk_df)}")
    print(f"  混淆（特征近10%）: {len(confusing_df)}")
    print(f"  所有混淆对（GT远Pred近）: {len(confusing_all)}")
    print(f"  其中特征距离 < {feat_threshold} 的混淆对: {len(confusing_strict)}")


def find_and_process_npz(root_dir, result_root='npz_analysis_results'):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.npz'):
                npz_path = os.path.join(dirpath, fname)
                relative_dir = os.path.relpath(dirpath, root_dir)
                save_dir = os.path.join(result_root, relative_dir)
                analyze_npz(npz_path, save_dir)