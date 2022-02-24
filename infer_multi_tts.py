import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import json
import re

import sys
sys.path.append('waveglow/')
import numpy as np
from numpy import finfo
import torch
from distributed import apply_gradient_allreduce

from resemblyzer.audio import preprocess_wav, trim_long_silences
from resemblyzer import VoiceEncoder
from hparams import create_hparams
from model import Tacotron2
from model_multi_tts import MultiSpeakerTacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from text import text_to_sequence, _clean_text
from denoiser import Denoiser
import torchaudio
import argparse
from scipy.io.wavfile import write

FRAME_PER_TOKEN = 9
regexp = re.compile(r'\s+([0-9]|km\/h|m\/s|kg|h|m|s)(\s|,|\.)+')

def load_model(hparams):
    model = MultiSpeakerTacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

        if hparams.distributed_run:
            model = apply_gradient_allreduce(model)

    return model

def get_embeds(audiopaths, speaker_embedding):
    audios = [trim_long_silences(preprocess_wav(audiopath)) for audiopath in audiopaths]
    min_audio_length = min([len(audio) for audio in audios])
    audios = [audio[:min_audio_length] for audio in audios]
    with torch.no_grad():
        embeds = []
        for wav in audios:
            embeds.append(speaker_embedding.embed_utterance(wav))
            embeds = torch.tensor(embeds).cuda()
    return embeds

def check_line(line):
    # return not bool(regexp.search(line))
    return True

def main(args):
    # Prepare output dirs and 
    if args.file == "": 
        args.file = None
    else:
        with open(args.file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
    os.makedirs(args.output_dir, exist_ok=True)

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    
    if args.decoder_steps > 0:
        hparams.max_decoder_steps = min(2000, args.decoder_steps)
        # hparams.max_decoder_steps = min(1000, args.decoder_steps)
        # if not args.file:
        #     hparams.max_decoder_steps = args.decoder_steps
        # else:
        #     max_length = max([len(line) for line in lines])

    checkpoint_path = args.checkpoint_path
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    # _ = model.cuda().eval()

    waveglow_path = args.waveglow_path
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    # waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()

    speaker_embedding = VoiceEncoder()
    denoiser = Denoiser(waveglow)

    def infer_text(text, filename=None):
        # TODO: brzydko
        # if args.file:
        model.decoder.max_decoder_steps = FRAME_PER_TOKEN * len(_clean_text(text, hparams.text_cleaners))
        print(_clean_text(text, hparams.text_cleaners))

        # print(_clean_text(text, hparams.text_cleaners))
        sequence = np.array(text_to_sequence(text, hparams.text_cleaners))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        def plot_data(data, figsize=(16, 4)):
            fig, axes = plt.subplots(1, len(data), figsize=figsize)
            for i in range(len(data)):
                axes[i].imshow(data[i], aspect='auto', origin='lower', 
                            interpolation='none')
            fig.savefig(os.path.join(args.output_dir, 'data.png'))

        embeds = get_embeds([args.embed], speaker_embedding)
        embeds = embeds.cuda().half()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, wavs=embeds)
        if not args.file:
            plot_data((mel_outputs.float().data.cpu().numpy()[0],
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T))

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=args.sigma)
        audio_denoised = denoiser(audio, strength=0.1)[:, 0]    

        if not args.file:
            audio = audio.data.cpu()
            audio = audio / torch.abs(audio).max()
            audio = audio.to(torch.float32)
            print(audio.shape)
            print(audio.dtype)
            print(audio)
            write(os.path.join(args.output_dir, filename),
                hparams.sampling_rate, audio.numpy().T)

        filename_denoised = '{}_denoised.wav'.format(filename.rsplit('.wav', 1)[0])
        audio_denoised = audio_denoised.data.cpu().numpy()
        audio_denoised = audio_denoised / np.abs(audio_denoised).max()

        return audio_denoised, filename_denoised

    if not args.file:
        text = args.text
        audio, filename = infer_text(text, 'sigma{}_gate_{}.wav'.format(args.sigma, args.gate))
        write(os.path.join(args.output_dir, filename),
            hparams.sampling_rate, audio.T)
    else:        
        file_dict = {}
        file_cnt = 0
        for i, line in enumerate(lines):
            if file_cnt > 50000:
                print("Last: {}".format(i))
                break

            if check_line(line):
                file_cnt += 1

                words = line.split()
                sentences = line.split('.')
                text = ""
                counter = 0
                audios = []
                filename = 'audio_{}.wav'.format(i)
                cleaned_text = ""
                for ix, word in enumerate(sentences):
                    if text == "":
                        text = word
                    else:
                        # text += " " + word
                        text += ". " + word

                    if len(text) > 80 and (ix == len(sentences)-1 or len(text + ". " + sentences[ix+1]) > 130):
                        if cleaned_text == "":
                            cleaned_text = _clean_text(text, hparams.text_cleaners)
                        else:
                            cleaned_text += ". " + _clean_text(text, hparams.text_cleaners)

                        audio, _ = infer_text(text, filename=filename)
                        text = ""
                        counter += 1
                        audios.append(audio)

                if len(text) > 0:
                    if cleaned_text == "":
                        cleaned_text = _clean_text(text, hparams.text_cleaners)
                    else:
                        cleaned_text += ". " + _clean_text(text, hparams.text_cleaners)

                    audio, _ = infer_text(text, filename=filename)
                    audios.append(audio)

                if len(audios) > 0:
                    # audio = torch.cat(audios, dim=1)
                    audio = np.concatenate(audios, axis=1)

                    write(os.path.join(args.output_dir, filename),
                        hparams.sampling_rate, audio.T)

                    file_dict[filename] = {
                        'speaker_id': 42, 
                        'text': line, 
                        'cleaned_text': cleaned_text
                    }
                    
        with open(os.path.join(args.output_dir, 'info.json'), 'w') as info_json:
            json.dump(file_dict, info_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path',
                        help='Path to tacotron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-e', '--embed', help='Path to reference speaker audio', type=str)
    parser.add_argument('-f', '--file', help='File with lines to read', 
    type=str, default="")
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument("-g", "--gate", default=0.5, type=float)
    parser.add_argument("-d", "--decoder_steps", default=0, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    main(args)
