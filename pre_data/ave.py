import pandas as pd
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEModel, AutoModel
import torch
import numpy as np
import torchaudio
from tqdm import tqdm


import timm
num_frames = 16
dataroot="data/AVE"

def preprocessAVE(anno_path):
    
    anno_df = pd.read_csv(anno_path,
                          sep='&',
                          header=None,
                          names=['Category', 'VideoID', 'Quality','StartTime','EndTime'])
    
    # process labels
    categories = anno_df['Category'].to_list()
    class_names = sorted(set(categories))
    class2idx = {cls_name:i for i, cls_name in enumerate(class_names)}
    
    print(class2idx)
    
    # labels = np.array([class2idx[c] for c in categories])
    labels = np.array([class2idx[c] for c in categories])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    video_model = video_model.to(device)

    video_model.eval()

    image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    
    audio_model = timm.create_model("hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True).to(device)
    audio_model = audio_model.eval()
    
    
    video_features_list = []
    audio_features_list = []
    
    
    
    for idx, row in tqdm(anno_df.iterrows(), desc="Processing..."):
        
        video_path = os.path.join(dataroot, "AVE", row["VideoID"]+".mp4")
        
        # 读取视频的视觉信息
        
        vidcap = cv2.VideoCapture(video_path)
        visual_frames = []
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            # Read all frames if less than 16
            success, image = vidcap.read()
            while success:
                visual_frames.append(image)
                success, image = vidcap.read()
        else:
            # 均匀采样16帧
            frame_indices = [int(round(i * total_frames / num_frames)) for i in range(num_frames)]
            frame_indices = [min(fi, total_frames - 1) for fi in frame_indices]
            for i in range(total_frames):
                success, image = vidcap.read()
                if not success:
                    break
                if i in frame_indices:
                    visual_frames.append(image)
            # 如果有的帧没有采到，则补全
            while len(visual_frames) < num_frames and len(visual_frames) > 0:
                visual_frames.append(visual_frames[-1])
        vidcap.release()
        video_inputs = image_processor(visual_frames, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            video_features = video_model(video_inputs)
        
        video_features = video_features.last_hidden_state
        
        video_features = video_features.mean(dim=1).squeeze(0)
        
        video_features = video_features.cpu().numpy()
        
        video_features_list.append(video_features)
            
        
        # 读取音频信息
            
        video = VideoFileClip(video_path)
        audio = video.audio
        # 将 AudioFileClip 转换为 numpy 数组，然后转换为 torch tensor
        audio_array = audio.to_soundarray(fps=16000)  # fps 可以根据需要调整
        waveform = torch.from_numpy(audio_array).float()
        # to_soundarray 返回格式: (samples, channels)
        # 如果是立体声，转换为单声道（取平均值）
        if waveform.dim() > 1 and waveform.shape[-1] > 1:
            waveform = waveform.mean(dim=-1, keepdim=True)  # (samples,1)
        # 确保是一维 (samples,)，然后添加通道维度
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(-1)  # (samples, 1) 
            
        waveform = waveform.transpose(0, 1)  # (1, samples) - 匹配 torchaudio 格式 (channels, samples)
        sr = 16000  # 采样率
        waveform = waveform - waveform.mean()
        
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,          # 使用 HTK 风格的 Mel 滤波器
            sample_frequency=sr,      # 音频采样率
            use_energy=False,         # 不使用能量项
            window_type='hanning',    # 汉宁窗
            num_mel_bins=128,         # Mel 频带数（特征维度）
            dither=0.0,               # 不加抖动噪声
            frame_shift=10            # 帧移 10ms（100 帧 / 秒）
        ) # fbank.shape = (T, 128)
        fbank = fbank.to(device)
        
        p = 1024 - len(fbank)
        # 计算需要 padding 或裁剪的长度

        if p > 0:
            # 如果帧数不足 1024
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            # ZeroPad2d((left, right, top, bottom))
            # 这里只在时间维（bottom）补 0

            fbank = m(fbank)
            # fbank.shape → (1024, 128)

        elif p < 0:
            # 如果帧数超过 1024
            fbank = fbank[:1024, :]
            # 直接裁剪前 1024 帧
            
        
        fbank = fbank.unsqueeze(dim=0).unsqueeze(dim=0)
        with torch.no_grad():
            out = audio_model(fbank).squeeze(dim=0).cpu().numpy()
        
        # 清理资源
        audio.close()
        video.close()
        
        audio_features_list.append(out)
        
        # print(f"处理完成: {row['VideoID']}")
            
            
    
    video_features = np.stack(video_features_list)
    audio_features = np.stack(audio_features_list)
    # 保存这两个feature矩阵
    # np.save(os.path.join(dataroot, "video_features.npy"), video_features)
    # np.save(os.path.join(dataroot, "audio_features.npy"), audio_features)
    
    return video_features, audio_features, labels

    
    

if __name__ == '__main__':
    os.chdir('..')
    train_v, train_a, train_y = preprocessAVE(os.path.join(dataroot, "trainSet.txt"))
    test_v, test_a, test_y = preprocessAVE(os.path.join(dataroot, "testSet.txt"))
    val_v, val_a, val_y = preprocessAVE(os.path.join(dataroot,"valSet.txt"))
    
    np.savez(os.path.join(dataroot, 'ave_features.npz'),
             train_videos=train_v, train_audios=train_a, train_targets=train_y,
             test_videos=test_v, test_audios=test_a, test_targets=test_y,
             val_videos=val_v, val_audios=val_a, val_targets=val_y)
    
    
    


