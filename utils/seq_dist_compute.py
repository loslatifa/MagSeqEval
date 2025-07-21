import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import os

def combined_feature_distance_norm_only(seq1, seq2, alpha=0.5, beta=0.5, eps=1e-8):
    seq1 = np.asarray(seq1).flatten()
    seq2 = np.asarray(seq2).flatten()

    euclidean = np.linalg.norm(seq1 - seq2)
    norm1 = np.linalg.norm(seq1) + eps
    norm2 = np.linalg.norm(seq2) + eps
    cosine_sim = np.dot(seq1, seq2) / (norm1 * norm2)
    cosine_dist = 1 - cosine_sim

    return alpha * euclidean + beta * cosine_dist


def combined_feature_distance_all_channels(seq_h1, seq_v1, seq_n1,
                                           seq_h2, seq_v2, seq_n2,
                                           alpha=0.5, beta=0.5, eps=1e-8):
    seq1 = np.concatenate([seq_h1, seq_v1, seq_n1])
    seq2 = np.concatenate([seq_h2, seq_v2, seq_n2])

    euclidean = np.linalg.norm(seq1 - seq2)
    norm1 = np.linalg.norm(seq1) + eps
    norm2 = np.linalg.norm(seq2) + eps
    cosine_sim = np.dot(seq1, seq2) / (norm1 * norm2)
    cosine_dist = 1 - cosine_sim

    return alpha * euclidean + beta * cosine_dist
