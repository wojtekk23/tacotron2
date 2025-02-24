import numpy as np
from scipy.io.wavfile import read
import scipy.signal as sps
import librosa
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path, final_sr):
    sampling_rate, data = read(full_path)
    number_of_samples = round(len(data) * final_sr / sampling_rate)
    data = sps.resample(data, number_of_samples)
    data, _ = librosa.effects.trim(data)
    return torch.FloatTensor(data.astype(np.float32)), final_sr


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
