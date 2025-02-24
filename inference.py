import matplotlib
import matplotlib.pylab as plt

import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import torchaudio
import argparse
import os
from scipy.io.wavfile import write

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


def plot_data(filepath, data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    fig.savefig(filepath)


def main(args):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    
    checkpoint_path = args.checkpoint_path
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()
    
    waveglow_path = args.waveglow_path
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    
    text = args.text
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    #torch.save(mel_outputs_postnet.float().data.cpu(), os.path.join(args.output_dir, 'mels.pt'))

    if args.plot:
        plot_data(os.path.join(args.output_dir, 'data.png'),
                (mel_outputs.float().data.cpu().numpy()[0],
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T))

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=args.sigma)
        audio = audio.data.cpu()
        audio = audio / torch.abs(audio).max()
        audio = audio.to(torch.float32)
        filename = f"sigma_{args.sigma}_gate_{args.gate}.wav" 
        
        write(os.path.join(args.output_dir, filename),
            hparams.sampling_rate, audio.numpy().T)
        
    if not args.no_denoise:
        audio_denoised = denoiser(audio, strength=0.05)[:, 0]
        audio_denoised = audio_denoised.data.cpu()
        audio_denoised = audio_denoised / torch.abs(audio_denoised).max()
        audio_denoised = audio_denoised.to(torch.float32)
        filename = f"sigma_{args.sigma}_gate_{args.gate}_denoised.wav"
        
        write(os.path.join(args.output_dir, filename),
            hparams.sampling_rate, audio_denoised.numpy().T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path',
                        help='Path to tacotron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    #parser.add_argument('-f', '--file', help='File with lines to read', 
    #type=str, default="")
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.66, type=float)
    # TODO: czy wykorzystujemy gdzieś gate?
    parser.add_argument("-g", "--gate", default=0.5, type=float)
    #parser.add_argument("-d", "--decoder_steps", default=0, type=int)
    parser.add_argument("--plot", action='store_true', help='Plot the mel outputs and the alignment chart')
    parser.add_argument("--no_denoise", action='store_true', help='Do not denoise the audio output')
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    main(args)

 
