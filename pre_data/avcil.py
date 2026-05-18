import numpy as np
import os

if __name__ == '__main__':
    # os.chdir("../data/AVE_features")
    # audio_feature = np.load("audio_pretrained_feature_dict.npy", allow_pickle=True)
    # visual_feature = np.load("visual_pretrained_feature_dict.npy", allow_pickle=True)
    # class_id = np.load("all_classId_vid_dict.npy", allow_pickle=True)
    
    # print(class_id.item(),keys())
    os.chdir("..")
    features = np.load("data/AVE/seq_ave_features.npz")
    print(features["train_audios"].shape)
    