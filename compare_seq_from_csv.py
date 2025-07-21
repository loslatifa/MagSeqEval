import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.pos_plot import compare_sequences_from_csv

if __name__ == "__main__":
    # 替换成你自己的路径
    csv_path = "./npz_analysis_results/sdcs_f4_TransCP/gt_far_pred_close_euclidean.csv"
    npz_path = "./csv_result/sdcs_f4_TransCP/anomaly_inputs_TransCP.npz"
    save_dir = "seq_comp_plot/TransCP/gt_far_pred_close_euclidean"

    compare_sequences_from_csv(csv_path, npz_path, save_dir)