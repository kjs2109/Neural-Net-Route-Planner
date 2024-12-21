import random 

import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms


class RandomFlipPair:
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, image, label):
        # Horizontal Flip 적용
        if random.random() < self.p_h:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)
        
        # Vertical Flip 적용
        if random.random() < self.p_v:
            image = transforms.functional.vflip(image)
            label = transforms.functional.vflip(label)
        
        return image, label


class ChannelwisePad:
    def __init__(self, target_height, target_width, padding_values):
        self.target_height = target_height
        self.target_width = target_width
        self.padding_values = padding_values

    def __call__(self, img):
        # CHW 형식의 Tensor 입력
        channels, height, width = img.shape

        # 각 채널에 대해 다른 패딩 값 적용
        padded_channels = []
        for c in range(channels):
            pad_left = 0
            pad_right = self.target_width - width
            pad_top = 0
            pad_bottom = self.target_height - height

            padded_channel = F.pad(
                img[c:c+1, :, :],  # 특정 채널 선택
                (pad_left, pad_right, pad_top, pad_bottom),  # (좌, 우, 위, 아래)
                mode='constant',
                value=self.padding_values[c]  # 채널별 패딩 값
            )
            padded_channels.append(padded_channel)

        # 패딩된 채널 병합
        return torch.cat(padded_channels, dim=0)