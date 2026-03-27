import pylab as pb
import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from pathlib import Path 
import pandas as pd
from ternary_diagram import TernaryDiagram as Td


def nmf(s_raw: np.array, n_comp: int )-> np.array:
    """
    s_raw: xrd raw data 2D 
    n_comp: number of basis matrix 
    """
    nmf_model = NMF(n_components=n_comp, init='random', random_state=42, max_iter=3000)
    w = nmf_model.fit_transform(s_raw)  #abundances
    h = nmf_model.components_ #components
    xy_reconstructed = np.dot(w, h) 
    return w, h