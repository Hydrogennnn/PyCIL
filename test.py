import numpy as np
import torch

import h5py

import pickle



def inspect(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        print(f"{prefix}dict  ({len(obj)} keys)")
        for k, v in obj.items():
            print(f"{prefix}  [{k}]:")
            inspect(v, indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}  (len={len(obj)})")
        for i, v in enumerate(obj[:3]):  # 只看前3个
            print(f"{prefix}  [{i}]:")
            inspect(v, indent + 2)
        if len(obj) > 3:
            print(f"{prefix}  ... ({len(obj)-3} more)")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}ndarray  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"{prefix}{type(obj).__name__}  = {repr(obj)[:80]}")


# with open("data/MOESI/mosei_senti_data.pkl", "rb") as f:
#     data = pickle.load(f)

# inspect(data)

# a = [26,14,23,4,11,25,31,10,29,5,6,9,17,22,2,19,13,1,21,16,8,3,27,28,15,30,0,7,12,18,20,24]

# print(len(a))

# a = np.load("data/visual_pretrained_feature_dict.npy", allow_pickle=True).item()
# for k, v in a.items():
#     print(v.shape)


# with h5py.File('data/KS/visual_features.h5', 'r') as f:
#     # 查看所有 key
#     print(f["zzyTugOJAPc"])
    
with open("data/KS/Kinetics400_data_tasks_10.pkl", "rb") as f:
    data = pickle.load(f)
    # inspect(data)
    print(len(data['train'][0]['shuffling cards']))

    # 读取某个数据集
