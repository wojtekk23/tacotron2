from hparams import create_hparams
from resemblyzer.audio import preprocess_wav, trim_long_silences
from resemblyzer import VoiceEncoder
from utils import load_wav_to_torch, load_filepaths_and_text
from tqdm import tqdm
import torch
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save embeddings')

    args = parser.parse_args()
    hparams = create_hparams(None, True)
    
    speaker_encoder = VoiceEncoder().cuda()
    train_paths = load_filepaths_and_text(hparams.training_files)
    valid_paths = load_filepaths_and_text(hparams.validation_files)
    
    #print("Training files:")
    #for filename, _ in tqdm(train_paths):
        #audio = trim_long_silences(preprocess_wav(filename))
        #embed = speaker_encoder.embed_utterance(audio)
        #torch.save(embed, os.path.join(args.output_directory, os.path.basename(filename).rsplit('.', 1)[0]))
    
    print("Validation files:")
    for filename, _ in tqdm(valid_paths):
        audio = trim_long_silences(preprocess_wav(filename))
        embed = speaker_encoder.embed_utterance(audio)
        torch.save(embed, os.path.join(args.output_directory, os.path.basename(filename).rsplit('.', 1)[0]))
