import os, re, random
from glob import glob
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
from transformers import (
    WhisperFeatureExtractor, 
    BitsAndBytesConfig, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    EarlyStoppingCallback,
    TrainerCallback, 
    TrainingArguments, 
    TrainerState, 
    TrainerControl,
    )
from audio_converter import AudioConverter
from utils import SprintDataset, DataCollatorSpeechSeq2SeqWithPadding, normalizeUnicode, normalizeUnicodextra, get_latest_checkpoint
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, 
    PeftModel, 
    LoraModel, 
    LoraConfig, 
    get_peft_model,
    PeftConfig
)
from torchmetrics.text import WordErrorRate, CharErrorRate
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

class CONFIG:
    debug=False
    do_aug = False
    do_train = True
    run_checkpoint=True
    wandb_log = True if not debug else False
    do_noise = False
    patience = 5
    device = 'cuda'
    base_model = 'tugstugi' #"openai/whisper-tiny"
    output_dir = 'Train/TugsTugiWithLora-v0.1'
    checkpoint_model = os.path.join(output_dir, get_latest_checkpoint(output_dir)) if run_checkpoint else None
    train_csv= 'clean_data.csv' #'ben10/ben10/train.csv'
    sample_rate=16000
    noisefiles = glob('audio/audio/*.wav') + glob('audio/audio/*/*.wav')
    per_device_train_batch_size=14
    per_device_eval_batch_size=14
    gradient_accumulation_steps=1
    learning_rate=1e-6
    warmup_steps=500
    save_steps= 250
    eval_steps=10 if debug else 250
    log_steps=1 if debug else 100
    save_limit = 4
    num_train_epochs=50 if not debug else 1
    loop_train_dataset = 1
    loop_val_datset = 1
    # test_size = 0.1
    valid_size = 0.2
    seed = 42

import wandb
if not CONFIG.wandb_log:
    wandb.init(mode="disabled") 
elif CONFIG.wandb_log:
    os.environ["WANDB_PROJECT"] ="KakatuaASR"
    wandb.login(key='9b38a904019934d720374c5f915943338425e6a6')

print('\n' + '='*70 + '\n')
print(f'Running Training on data: "{CONFIG.train_csv}"')

if not os.path.exists(CONFIG.output_dir):
    os.mkdir(CONFIG.output_dir)

# Load data and run text preprocessings.
print('\n' + '='*70 +'\n')
print('Loading data...')

data = pd.read_csv(CONFIG.train_csv)
new_df = data

print(f'Data loading done, total datapoints: {len(data)}')
print('\n'+'='*70)

# Generate train test and validation split
train_df, valid_df = train_test_split(new_df, 
                                     test_size=CONFIG.valid_size,
                                     random_state=CONFIG.seed,
                                     ) 
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# Load Model, Processor and Tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(CONFIG.base_model)
if CONFIG.base_model.endswith('small') or CONFIG.base_model.endswith('tiny'):
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path='tugstugi', 
                                                 task='transcribe',
                                                 language = 'bn')
    print(f'Tokenizer loaded from TUGSTUGI for training "{CONFIG.base_model}"')
else:
    tokenizer = WhisperTokenizer.from_pretrained(CONFIG.base_model, language="bn", task="transcribe")
processor = WhisperProcessor.from_pretrained(CONFIG.base_model,  language="bn", task="transcribe")

try:
    if isinstance(CONFIG.noisefiles, list) and os.path.isfile(CONFIG.noisefiles[0]):
        noise_file_paths = CONFIG.noisefiles 
    elif os.path.dir(CONFIG.noisefiles):
        noise_file_paths = glob(CONFIG.noisefiles + '*.wav')
    if CONFIG.do_noise:
        print('\n'+f'Noise files will be added during training.'+'\n')
except:
    noise_file_paths = None
    raise Warning('\n'+f"Noise files at {CONFIG.noisefiles} are not accessible. Turning off noise file settings."+'\n')

# Creating dataset objects for train and val.
train_ac = AudioConverter(sampleRate=CONFIG.sample_rate, 
                          disableAug=CONFIG.do_aug,
                          noiseFileList=noise_file_paths if CONFIG.do_noise else None,
                          )
valid_ac = AudioConverter(sampleRate=CONFIG.sample_rate, 
                        disableAug=True
                        )

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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# wer_metric = evaluate.load("wer")
# cer_metric = evaluate.load("cer")

# def compute_metrics_whisper(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids

#     # replace -100 with the pad_token_id
#     label_ids[label_ids == -100] = tokenizer.pad_token_id

#     # we do not want to group tokens when computing the metrics
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     # print(f'This is prediction: {pred_str}')
#     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     wer = wer_metric.compute(predictions=pred_str, references=label_str)
#     cer = cer_metric.compute(predictions=pred_str, references=label_str)
#     print(f"At this evaluation, WER is: {wer} and CER is: {cer}")
#     return {"wer": wer, "cer": cer}

cer = CharErrorRate()
wer = WordErrorRate()

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_res = wer(pred_str, label_str)
    cer_res = cer(pred_str, label_str)
    
    """
        uncomment the next 3 lines if you want to see how the examples look like during eval 
    """
    print("WER:",wer_res,"| CER:", cer_res)
    # print("Pred:",pred_str[0])
    # print("Label:",label_str[0])
    
    return {"wer": wer_res, "cer": cer_res}

if CONFIG.do_train:    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if not CONFIG.run_checkpoint:
        model = WhisperForConditionalGeneration.from_pretrained(CONFIG.base_model, quantization_config=quant_config, device_map="auto")
    else:
        print('='*99 + '\n')
        print(f'The Training is Running from a checkpoint at: {CONFIG.checkpoint_model}')
        print('\n' + '='*99)
        model = WhisperForConditionalGeneration.from_pretrained(CONFIG.checkpoint_model, quantization_config=quant_config,local_files_only=True, device_map="auto")
    
    model = prepare_model_for_kbit_training(model)
    
    """
        whisper uses CNNs in the encoder
        LoRA freezes all layers, but let's make the first layer (receiving layer) trainable.
    """
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        greater_is_better=False,
        num_train_epochs=CONFIG.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        lr_scheduler_type='cosine',
        save_total_limit=CONFIG.save_limit,
        optim = "paged_adamw_8bit",
        remove_unused_columns=False,
    )

    model.generation_config.language = "bn"
    model.generation_config.task = "transcribe"
    model.generation_config.lang_to_id = {"<|bn|>": 50302}
    model.generation_config.task_to_id = {"transcribe": 50359}

    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    from transformers import Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=CONFIG.patience)],
        
    )
    
    if not CONFIG.run_checkpoint:
        print('\n' + f'Starting training with train dataset of length: {train_dataset.__len__()}')
        print(f'And valid dataset of length: {valid_dataset.__len__()}')
        print('\n' + '='*70)
    else:
        print(f'Training resuming from {CONFIG.checkpoint_model}')
        
    trainer.train(
        resume_from_checkpoint=CONFIG.checkpoint_model if CONFIG.run_checkpoint else False,
    )

    model.save_pretrained(CONFIG.output_dir)
    tokenizer.save_pretrained(CONFIG.output_dir)
    out_logs = pd.DataFrame(trainer.state.log_history)
    out_logs.to_csv("logs.csv", index = False)
    # trainer.save_pretrained(CONFIG.output_dir)