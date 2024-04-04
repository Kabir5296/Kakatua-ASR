import os, re, glob, random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
import torch, torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
import evaluate
from transformers import Trainer, Seq2SeqTrainingArguments, pipeline
from transformers.utils import is_flash_attn_2_available
from sklearn.model_selection import train_test_split
# from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, EarlyStoppingCallback
from audio_converter import AudioConverter
from utils import SprintDataset, DataCollatorSpeechSeq2SeqWithPadding, normalizeUnicode, normalizeUnicodextra, remove_punctuation, get_latest_checkpoint

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

class CONFIG:
    debug=False
    do_test = True
    do_train = True
    run_checkpoint=False
    wandb_log = False
    patience = 5
    device = 'cuda'
    base_model = "openai/whisper-tiny"
    output_dir = 'Train/NoiseDataIntroducedTraining'
    checkpoint_model = os.path.join(output_dir, get_latest_checkpoint(output_dir)) if run_checkpoint else None
    normalized_csv='data/data.csv'
    sample_rate=16000
    noisefiles =glob('noise_files/audio/audio/*.wav') + glob('noise_files/audio/audio/*/*.wav')
    per_device_train_batch_size=4
    per_device_eval_batch_size=4
    per_device_test_batch_size=12
    gradient_accumulation_steps=1
    learning_rate=1e-6
    warmup_steps=500
    save_steps=1 if debug else 1000
    eval_steps=1 if debug else 250
    log_steps=1 if debug else 100
    save_limit = 4
    num_train_epochs=30
    loop_train_dataset = 1
    loop_val_datset = 1
    test_size = 0.2
    valid_size = 0.2
    seed = 42
    stratify_column_name = None

import wandb
if not CONFIG.wandb_log:
    wandb.init(mode="disabled") 
elif CONFIG.wandb_log:
    if CONFIG.wandblog:
        os.environ["WANDB_PROJECT"] ="ASR Call Centre Whisper"
        wandb.login(key='9b38a904019934d720374c5f915943338425e6a6')

print('\n' + '='*70 + '\n')
print(f'Running Training on data: "{CONFIG.normalized_csv}"')

if not os.path.exists(CONFIG.output_dir):
    os.mkdir(CONFIG.output_dir)

# Load data and run text preprocessings.
print('\n' + '='*70 +'\n')
print('Running text preprocessings...')

data = pd.read_csv(CONFIG.normalized_csv)
data["sentence"] = data["sentence"].progress_apply(normalizeUnicodextra)
new_df = data

print('\n'+'='*70)

# Generate train test and validation split
train_val, test_df = train_test_split(new_df, 
                                      test_size=CONFIG.test_size, 
                                      stratify=CONFIG.stratify_column_name if CONFIG.stratify_column_name else None, 
                                      random_state=CONFIG.seed
                                      )
train_df, valid_df = train_test_split(train_val, 
                                     test_size=CONFIG.valid_size,
                                     random_state=CONFIG.seed,
                                     stratify=CONFIG.stratify_column_name if CONFIG.stratify_column_name else None
                                     ) 
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# Load Model, Processor and Tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(CONFIG.base_model)
tokenizer = WhisperTokenizer.from_pretrained(CONFIG.base_model, language="bn", task="transcribe")
processor = WhisperProcessor.from_pretrained(CONFIG.base_model,  language="bn", task="transcribe")

try:
    if isinstance(CONFIG.noisefiles, list) and os.path.isfile(CONFIG.noisefiles[0]):
        noise_file_paths = CONFIG.noisefiles 
    elif os.path.dir(CONFIG.noisefiles):
        noise_file_paths = glob(CONFIG.noisefiles + '*.wav')
except:
    noise_file_paths = None
    raise Warning(f"Noise files at {CONFIG.noisefiles} are not accessible. Turning off noise file settings.")

# Creating dataset objects for train and val.
train_ac = AudioConverter(sampleRate=CONFIG.sample_rate, 
                          disableAug=False,
                          noiseFileList=noise_file_paths,
                          )
valid_ac = AudioConverter(sampleRate=CONFIG.sample_rate, 
                        disableAug=True
                        )
# test_ac = AudioConverter(sampleRate=CONFIG.sample_rate, 
#                         disableAug=True)

train_dataset = SprintDataset(train_df, 
                              processor, 
                              train_ac, 
                              feature_extractor,
                              CONFIG.loop_train_dataset)

valid_dataset = SprintDataset(valid_df, 
                              processor, 
                              valid_ac, 
                              feature_extractor,
                              CONFIG.loop_val_datset)

# test_dataset = SprintDataset(test_df, 
#                               processor, 
#                               test_ac, 
#                               feature_extractor,
#                               CONFIG.loop_val_datset)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics_whisper(pred):
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

if CONFIG.do_train:
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG.output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=CONFIG.per_device_train_batch_size,
        per_device_eval_batch_size=CONFIG.per_device_eval_batch_size,
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=CONFIG.learning_rate,
        warmup_steps=CONFIG.warmup_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        dataloader_pin_memory=False,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=CONFIG.save_steps,
        eval_steps=CONFIG.eval_steps,
        logging_steps=CONFIG.log_steps,
        num_train_epochs=CONFIG.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        lr_scheduler_type='cosine',
        save_total_limit=CONFIG.save_limit,
    )

    if not CONFIG.run_checkpoint:
        model = WhisperForConditionalGeneration.from_pretrained(CONFIG.base_model).to(CONFIG.device)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(CONFIG.checkpoint_model,local_files_only=True).to(CONFIG.device)


    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_param = {total_param/1000000}")
    print(f"trainable = {trainable_param/1000000}")
    print('\n'+'='*70)

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_whisper,
        tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=CONFIG.patience)],
        
    )

    trainer.train(
        resume_from_checkpoint=CONFIG.checkpoint_model if CONFIG.run_checkpoint else False,
    )

    model.save_pretrained(CONFIG.output_dir)
    tokenizer.save_pretrained(CONFIG.output_dir)
    trainer.save_pretrained(CONFIG.output_dir)

if CONFIG.do_test:
    print('\n'+'Running Test on Test Dataset' + '\n')
    pipe = pipeline(task='automatic-speech-recognition',
                    model=CONFIG.output_dir,
                    device = CONFIG.device,
                    torch_dtype=torch.float16,
                    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"}
                    )
        
    inference_list = pipe(test_df['path'].tolist(),
                        batch_size=CONFIG.per_device_test_batch_size,
                        chunk_length_s=30,
                        return_timestamps=False,
                        )
    
    inference_dict = {'inference':[], 'label':[]}
    for inf_text, label in zip(inference_list, test_df['sentence'].tolist()):
        inference_dict['inference'].append(remove_punctuation(inf_text['text']).lower())
        inference_dict['label'].append(remove_punctuation(label).lower())
        
    inf_csv = pd.DataFrame.from_dict(inference_dict)
    inf_csv.to_csv('inference.csv', index=False)
    
    wer = wer_metric.compute(predictions=inference_dict['inference'], references=inference_dict['label'])
    print(f'WER for test dataset is: {wer}')
    cer = cer_metric.compute(predictions=inference_dict['inference'], references=inference_dict['label'])
    print(f'CER for test dataset is: {cer}')