import numpy as np
import h5py
import os
if __name__ == "__main__":
    os.chdir("..")
    path = "data/KS/visual_features.h5"
    data = h5py.File(path, 'r')
    print(data["vZbWB3jLd20"].shape)