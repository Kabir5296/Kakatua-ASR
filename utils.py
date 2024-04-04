import os
import re
import glob
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset

# import wandb
import evaluate
import transformers
from transformers import Trainer, Seq2SeqTrainingArguments
# from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

def get_latest_checkpoint(model_folder):
    number = []
    for all in os.listdir(model_folder):
        if all.startswith('checkpoint'): 
            number.append(all.split('-')[-1])
    return f'checkpoint-{max(number)}'

def remove_punctuation(text):
    punctuation_pattern = re.compile(r'[^\w\s]')
    return punctuation_pattern.sub('', text)
class SprintDataset(Dataset):
    def __init__(self, df, processor, audioConverter, feature_extractor, loopDataset=1):
        self.df = df
        self.paths = df['path']
        self.sentences = df['sentence']
        self.len = len(self.df) * loopDataset

        self.processor = processor
        self.ac = audioConverter
        self.feature_extractor = feature_extractor

    def __len__(self):
        return self.len

    def loadSample(self, idx):
        idx %= len(self.df)
        audio_paths = self.paths[idx]
        sentence = self.sentences[idx]
        if not audio_paths.startswith("["): 
            audio_paths = [audio_paths]
        else:
            audio_paths = eval(audio_paths)
        waves = [torch.from_numpy(self.feature_extractor(self.ac.getAudio(audio_path)[0], sampling_rate=16000).input_features[0]) for audio_path in audio_paths]
        input_values = torch.cat(waves, axis = 0) #[0]
        # print(len(input_values))

        input_length = len(input_values)
        # with self.processor.as_target_processor():
        labels = self.processor.tokenizer(remove_punctuation(sentence)).input_ids
        # print(processor.decode(labels))
        return {
            'input_features':input_values,
            'input_length':input_length,
            'labels':labels
        }

    def __getitem__(self, idx): 
        if idx >= self.len:
            raise IndexError('index out of range')
        return self.loadSample(idx)

    def __getitem__(self, idx): 
        if idx >= self.len:
            raise IndexError('index out of range')
        return self.loadSample(idx)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
STANDARDIZE_ZW = re.compile(r'(?<=\u09b0)[\u200c\u200d]+(?=\u09cd\u09af)')
DELETE_ZW = re.compile(r'(?<!\u09b0)[\u200c\u200d](?!\u09cd\u09af)')
PUNC = re.compile(r'([\?\.।;:,!])')

def removeOptionalZW(text):
    """
    Removes all optional occurrences of ZWNJ or ZWJ from Bangla text.
    """
    text = STANDARDIZE_ZW.sub('\u200D', text)
    text = DELETE_ZW.sub('', text)
    return text

def separatePunc(text):
    """
    Checks for punctuation puts a space between the punctuation
    and the adjacent word.
    """
    text = PUNC.sub(r" \1 ", text)
    text = " ".join(text.split())
    return text

def removePunc(text):
    """
    Remove for punctuations from text.
    """
    text = PUNC.sub(r"", text)
    return text

def remove_multiple_strings(cur_string):
  for cur_word in ['"', "'", '”', '\u200d']:
    cur_string = cur_string.replace(cur_word, '')
  for cur_word in ['-', '—']:
    cur_string = cur_string.replace(cur_word, ' ')
  return cur_string
    
def normalizeUnicodextra(text):
    """
    Normalizes unicode strings using the Normalization Form Canonical
    Composition (NFC) scheme where we first decompose all characters and then
    re-compose combining sequences in a specific order as defined by the
    standard in unicodedata module. Finally all zero-width joiners are
    removed.
    """
    text = text.replace(u"\u098c", u"\u09ef")
    text = remove_multiple_strings(text)
    text = removeOptionalZW(text)
    text = removePunc(text)
    return text

def normalizeUnicode(text):
    """
    Normalizes unicode strings using the Normalization Form Canonical
    Composition (NFC) scheme where we first decompose all characters and then
    re-compose combining sequences in a specific order as defined by the
    standard in unicodedata module. Finally all zero-width joiners are
    removed.
    """
    text = text.replace(u"\u098c", u"\u09ef")
    text = removeOptionalZW(text)
    text = removePunc(text)
    return text

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics_wav2vec2(processor, pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # We do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids)
    print(f'This is prediction: {pred_str}')
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    # print(f'This is prediction: {label_str}')
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

def compute_metrics_whisper(tokenizer, pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # print(f'This is prediction: {pred_str}')
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    print(f"At this evaluation, WER is: {wer} and CER is: {cer}")
    return {"wer": wer, "cer": cer}