import numpy as np
import h5py
import os
if __name__ == '__main__':
    os.chdir("../data/AVE_features")
    
    # visual_feature = h5py.File("visual_features.h5", 'r')

    # print(visual_feature.item().keys())


    class_id = np.load("all_classId_vid_dict.npy", allow_pickle=True) #样本和类别的对应关系

    # targets = []
    # audios = []
    # videos = []
    # for cls, sample_ids in class_id.item().items():
    #     for sample_id in tqdm(sample_ids):
    #         targets[sample_id] = cls
    # np.save("labels.npy", targets)


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

    
    
    # audios = np.stack(audios, axis=0)
    # videos = np.stack(videos, axis=0)

    # targets = np.array(targets)
    
    np.save("split.npy", split)
    np.save("labels.npy", labels)
    # np.save("audio_features.npy", audios)
    # np.save("video_features.npy", videos)

