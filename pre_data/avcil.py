import numpy as np
import os

if __name__ == '__main__':
    # os.chdir("../data/AVE_features")
    # audio_feature = np.load("audio_pretrained_feature_dict.npy", allow_pickle=True)
    # visual_feature = np.load("visual_pretrained_feature_dict.npy", allow_pickle=True)
    # class_id = np.load("all_classId_vid_dict.npy", allow_pickle=True)
    
    # print(class_id.item(),keys())
    os.chdir("..")
    dataroot = "data/AVE"
    data = np.load(os.path.join(dataroot, "seq_ave_features.npz"), allow_pickle=True)
    
    video_features = np.concatenate([data['train_videos'], data['test_videos'], data['val_videos']], axis=0)
    audio_features = np.concatenate([data['train_audios'], data['test_audios'], data['val_audios']], axis=0)
    labels = np.concatenate([data['train_targets'], data['test_targets'], data['val_targets']], axis=0)

    split = {}
    split["train"] = np.arange(len(data['train_targets']))
    split["test"] = np.arange(len(data['train_targets']), len(data['train_targets']) + len(data['test_targets']))
    split["val"] = np.arange(len(data['train_targets']) + len(data['test_targets']), len(labels))
    np.save(os.path.join(dataroot, "split.npy"), split)


    np.save(os.path.join(dataroot, "video_features.npy"), video_features)
    np.save(os.path.join(dataroot, "audio_features.npy"), audio_features)
    np.save(os.path.join(dataroot, "labels.npy"), labels)
