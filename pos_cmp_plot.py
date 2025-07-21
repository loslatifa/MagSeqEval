import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from utils.pos_plot import visualize_csv, find_all_csv\

def main():
    root_folder = "./npz_analysis_results"  # Change this to your root directory
    csv_list = find_all_csv(root_folder)

    print(f"Found {len(csv_list)} CSV files.")
    for csv_path in csv_list:
        print(f"Processing: {csv_path}")
        try:
            visualize_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")


if __name__ == "__main__":
    main()
