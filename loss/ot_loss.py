import numpy as np
import ot

def optimal_transport_loss(source_features, target_features):
    distance_matrix = np.linalg.norm(source_features - target_features, axis=1)
    M = ot.emd([], [], distance_matrix)
    return np.sum(M * distance_matrix)