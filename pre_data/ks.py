import pandas as pd
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import timm

num_frames = 16
dataroot = "data/AVE"
BATCH_SIZE = 8  # 根据显存调整


def load_visual_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    visual_frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        success, image = vidcap.read()
        while success:
            visual_frames.append(image)
            success, image = vidcap.read()
    else:
        frame_indices = set([min(int(round(i * total_frames / num_frames)), total_frames - 1) for i in range(num_frames)])
        for i in range(total_frames):
            success, image = vidcap.read()
            if not success:
                break
            if i in frame_indices:
                visual_frames.append(image)
        while len(visual_frames) < num_frames and len(visual_frames) > 0:
            visual_frames.append(visual_frames[-1])
    vidcap.release()
    return visual_frames


def load_fbank(video_path, device):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_array = audio.to_soundarray(fps=16000)
    waveform = torch.from_numpy(audio_array).float()
    if waveform.dim() > 1 and waveform.shape[-1] > 1:
        waveform = waveform.mean(dim=-1, keepdim=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(-1)
    waveform = waveform.transpose(0, 1)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=16000,
        use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0.0, frame_shift=10
    )
    p = 1024 - len(fbank)
    if p > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
    elif p < 0:
        fbank = fbank[:1024, :]
    audio.close()
    video.close()
    # shape: [1, 1024, 128]
    return fbank.unsqueeze(0)


def process_batch_video(frames_list, image_processor, video_model, device):
    """frames_list: list of list of frames, 每个元素是一个视频的16帧"""
    # image_processor 支持批量输入：传入 list of list of frames
    video_inputs = image_processor(frames_list, return_tensors="pt")["pixel_values"].to(device)
    # shape: [B, 16, 3, 224, 224]
    with torch.no_grad():
        output = video_model(video_inputs).last_hidden_state
    # [B, 1568, 768] → mean → [B, 768]
    return output.mean(dim=1).cpu().numpy()


def process_batch_audio(fbank_list, audio_model, device):
    """fbank_list: list of [1, 1024, 128] tensors"""
    # stack → [B, 1, 1024, 128]
    batch = torch.stack(fbank_list, dim=0).to(device)
    with torch.no_grad():
        out = audio_model(batch).cpu().numpy()
    # shape: [B, feat_dim]
    return out


def preprocessAVE(anno_path, image_processor, video_model, audio_model, device):
    anno_df = pd.read_csv(anno_path, sep='&', header=None,
                          names=['Category', 'VideoID', 'Quality', 'StartTime', 'EndTime'])
    categories = anno_df['Category'].to_list()
    class_names = sorted(set(categories))
    class2idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    labels = np.array([class2idx[c] for c in categories])

    video_features_list = []
    audio_features_list = []

    # 按 batch 处理
    rows = list(anno_df.iterrows())
    for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing batches"):
        batch_rows = rows[batch_start: batch_start + BATCH_SIZE]

        frames_batch = []
        fbank_batch = []

        for _, row in batch_rows:
            video_path = os.path.join(dataroot, "videos", row["VideoID"] + ".mp4")
            frames_batch.append(load_visual_frames(video_path))
            fbank_batch.append(load_fbank(video_path, device))

        # 批量推理
        video_feats = process_batch_video(frames_batch, image_processor, video_model, device)
        audio_feats = process_batch_audio(fbank_batch, audio_model, device)

        for vf in video_feats:
            video_features_list.append(vf)
        for af in audio_feats:
            audio_features_list.append(af)

    return np.stack(video_features_list), np.stack(audio_features_list), labels


if __name__ == '__main__':
    os.chdir('..')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型只初始化一次
    video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
    image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    audio_model = timm.create_model(
        "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True
    ).to(device).eval()

    train_v, train_a, train_y = preprocessAVE(os.path.join(dataroot, "trainSet.txt"), image_processor, video_model, audio_model, device)
    test_v, test_a, test_y = preprocessAVE(os.path.join(dataroot, "testSet.txt"), image_processor, video_model, audio_model, device)
    val_v, val_a, val_y = preprocessAVE(os.path.join(dataroot, "valSet.txt"), image_processor, video_model, audio_model, device)

    np.savez(os.path.join(dataroot, 'seq_ave_features.npz'),
             train_videos=train_v, train_audios=train_a, train_targets=train_y,
             test_videos=test_v, test_audios=test_a, test_targets=test_y,
             val_videos=val_v, val_audios=val_a, val_targets=val_y)