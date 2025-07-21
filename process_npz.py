import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Literal
from utils.analyze_npz import find_and_process_npz

if __name__ == '__main__':
    # 替换成你的根目录
    root_npz_dir = './csv_result'  # 比如 'outputs/sdcs_f4/Magneto'
    find_and_process_npz(root_npz_dir)