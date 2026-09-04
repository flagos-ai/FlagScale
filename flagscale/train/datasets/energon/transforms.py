# Copyright (c) 2025, BAAI. All rights reserved.
# Image transforms for Bagel data pipeline, ported from Bagel's transforms.py.

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize image so longest/shortest sides are within range, divisible by stride."""

    def __init__(
        self,
        max_size,
        min_size,
        stride,
        max_pixels,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    """Standard image transform for Bagel: resize + to_tensor + normalize."""

    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    ):
        self.stride = image_stride
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=list(image_mean), std=list(image_std), inplace=True
        )

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


class AudioTransform:
    """Whisper-style mel spectrogram feature extractor for audio.

    Extracts log-mel spectrogram from a raw waveform (1-D numpy float32).
    Truncates audio longer than ``chunk_length`` seconds and zero-pads
    shorter audio so every sample produces a fixed-size spectrogram.

    Args:
        feature_size: Number of mel bins.
        sampling_rate: Expected sample rate (Hz).
        hop_length: STFT hop length in samples.
        chunk_length: Max audio duration in seconds — longer audio is
            truncated, shorter audio is zero-padded.
        n_fft: FFT window size.
    """

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, waveform: np.ndarray) -> tuple:
        """
        Args:
            waveform: 1-D numpy float32 array of raw audio samples at
                ``self.sampling_rate`` Hz.
        Returns:
            (mel_features, feature_length) where ``mel_features`` is a numpy
            array of shape ``(n_mels, T_fixed)`` and ``feature_length`` is the
            number of *real* (non-padded) mel frames.
        """
        waveform = np.asarray(waveform, dtype=np.float32)

        # Record the real length before truncate/pad
        real_length = min(len(waveform), self.n_samples)

        # Truncate if longer than chunk_length
        if len(waveform) > self.n_samples:
            waveform = waveform[: self.n_samples]

        # # Pad with zeros if shorter than chunk_length
        # if len(waveform) < self.n_samples:
        #     waveform = np.pad(waveform, (0, self.n_samples - len(waveform)))

        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )
        log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # feature_length = number of real (non-padded) mel frames
        feature_length = real_length // self.hop_length

        return log_spec, feature_length
