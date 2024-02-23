import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import adjust_brightness, hflip
from torchvision.io import encode_jpeg, decode_jpeg

from .videofact2_dataset import VideoFact2Dataset
from typing import *


class VideoFact2InpaintingVIDNetDataset(Dataset):
    def __init__(
        self,
        videofact2_dataset_obj: VideoFact2Dataset,
        is_training: bool = True,
    ):
        self.dataset = videofact2_dataset_obj
        self.is_training = is_training

        assert self.dataset.return_frame_size is None, "return_frame_size must be None"
        assert self.dataset.return_type == "video", "return_type must be 'video'"

        self.transforms = Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_ela(self, frames):
        ela_frames = []
        for frame in frames:
            frame = (frame * 255).to(torch.uint8)
            reencoded_frame = decode_jpeg(encode_jpeg(frame, quality=50))
            ela_frame = torch.abs(frame - reencoded_frame)
            ela_frame = adjust_brightness(ela_frame, 255.0/ela_frame.max()).int().float() / 255.0
            ela_frames.append(ela_frame)
        return torch.stack(ela_frames)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        frames, label, masks, shape = sample
        ela_frames = self.get_ela(frames)

        frames = self.transforms(frames)
        ela_frames = self.transforms(ela_frames)

        if self.is_training:
            if torch.rand(1)[0] < 0.5:
                frames = hflip(frames)
                ela_frames = hflip(ela_frames)
                masks = hflip(masks)

        return frames, ela_frames, masks, label
