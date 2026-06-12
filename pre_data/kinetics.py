import numpy as np
import h5py
import os

def dict_to_h5(data_dict, h5_path, compression=None):
    with h5py.File(h5_path, "w") as f:
        for key, value in data_dict.items():
            f.create_dataset(
                key,
                data=value
            )


if __name__ == '__main__':
    os.chdir("../data/AVE_features")
    
    # visual_feature = h5py.File("visual_features.h5", 'r')

    # print(visual_feature.item().keys())


    class_id = np.load("all_classId_vid_dict.npy", allow_pickle=True) #样本和类别的对应关系
    visual_feature = np.load("visual_pretrained_feature_dict.npy", allow_pickle=True).item()
    dict_to_h5(visual_feature, "visual_features.h5")


    split = {}
    # cur = 0
    # pre = 0
    labels = {}
    for split_name in ["train", "test", "val"]:
        ids = []
        targets = []
        for cls, sample_ids in class_id.item()[split_name].items():
            for sample_id in sample_ids:
                targets.append(cls)
                # audios.append(audio_feature[sample_id])
                # videos.append(visual_feature[sample_id])
                # cur+=1
                ids.append(sample_id)
        
        split[split_name] = ids
        labels[split_name] = targets
        # pre = cur
    
    np.save("split.npy", split)
    np.save("labels.npy", labels)