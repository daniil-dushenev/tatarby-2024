import os
import torchaudio
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration # Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import Dataset, DatasetDict, Audio, Features
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchaudio
import numpy as np
import tqdm
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import tqdm
import wandb
from transformers import get_scheduler
from torch.optim import AdamW
from jiwer import wer  # Ensure you have jiwer installed: pip install jiwer



import warnings


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="It is strongly recommended to pass the `sampling_rate` argument to this function. Failing to do so can result in silent errors that might be hard to debug.")


def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return {
        "audio": {
            "array": waveform.numpy().squeeze(),
            "sampling_rate": sample_rate,
            "path": file_path
        }
    }

def load_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()
    return transcript

audio_files = []
transcript_files = []

def collect_files(audio_files, transcript_files, dir):
    audio_files += sorted([dir + '/' + f for f in os.listdir(dir) if f.endswith('.mp3')])
    transcript_files += sorted([dir + '/' + f for f in os.listdir(dir) if f.endswith('.txt')])
    return audio_files, transcript_files


for dir in ['hack', 
            'TatSC/crowdsourced_tg/train',
            'TatSC/crowdsourced_web/train',
            'TatSC/crowdsourced_web/test',
            'TatSC/crowdsourced_web/dev',
            'TatSC/audiobooks/train',
            'TatSC/audiobooks/test',
            'TatSC/audiobooks/dev'
           ]:
    audio_files, transcript_files = collect_files(audio_files, transcript_files, dir)

SIZE_DATASET = 0.1
aud_all = audio_files
tr_all = transcript_files
n_samples = len(audio_files)
audio_files = audio_files[:int(n_samples*SIZE_DATASET)]
transcript_files = transcript_files[:int(n_samples*SIZE_DATASET)]

audio_test = aud_all[int(n_samples*SIZE_DATASET)+1000:int(n_samples*SIZE_DATASET)+1000+int(n_samples*0.1) ]
tr_test = tr_all[int(n_samples*SIZE_DATASET)+1000:int(n_samples*SIZE_DATASET)+1000+int(n_samples*0.1) ]

processor = AutoProcessor.from_pretrained("sanchit-gandhi/whisper-small-tt-1k-steps")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

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
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=50258,
)


class WhisperDataset(Dataset):
    def __init__(self, audio_files, transcript_files, processor, dir):
        self.audio_files = audio_files
        self.transcript_files = transcript_files
        self.processor = processor
        self.dir = dir

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if isinstance(idx, int):  # Check if idx is a single integer
            return self._get_item_single(idx)
        elif isinstance(idx, list):  # Check if idx is a list of integers
            return [self._get_item_single(i) for i in idx]
        else:
            raise TypeError("Index must be either an integer or a list of integers")

    def _get_item_single(self, idx):
        audio_path = self.audio_files[idx]
        transcript_path = self.transcript_files[idx]
        speech, sampling_rate = torchaudio.load(audio_path)
        # Processing audio
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            resampled_waveform = resampler(speech[0])
        else:
            resampled_waveform = speech[0]
        audio_input = self.processor.feature_extractor(
            resampled_waveform, 
            sampling_rate=16000,
            return_tensors="pt", 
            padding="max_length", 
            max_length=480000, 
            truncation=True
        )
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        
        tokenized = self.processor.tokenizer(transcript, return_tensors='pt', padding='max_length', return_attention_mask=True, max_length=448)
        labels, attention_mask = tokenized['input_ids'][0], tokenized['attention_mask'][0]
        
        # Loading text
        return {
            "input_features": audio_input.input_features[0],
            "sentence": transcript,
            "labels": labels,
            "attention_mask": attention_mask
        }


ds = WhisperDataset(audio_files, transcript_files, processor, dir)

ds_test = WhisperDataset(audio_test, tr_test, processor, dir)


def data_collate_fn(data_list):
    input_features = [item["input_features"] for item in data_list]
    labels = [item["labels"] for item in data_list]
    attention_mask = [item["attention_mask"] for item in data_list]

    return {
        'input_features': torch.stack(input_features),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }



def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



# wandb.init(project="tatar-hack")
def validate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    num_batches = 0

    for batch in tqdm.tqdm(dataloader, desc="Validation Batch", leave=False):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_features, labels=labels)
            logits = outputs.logits

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Calculate WER
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_texts = processor.batch_decode(labels, skip_special_tokens=True)
            batch_wer = wer(label_texts, pred_texts)
            total_wer += batch_wer

        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_wer = total_wer / num_batches
    return avg_loss, avg_wer



def train(rank, world_size):
# Определение устройства

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained("sanchit-gandhi/whisper-small-tt-1k-steps")
    teacher_model.config.forced_decoder_ids = None
    student_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    student_model.config.forced_decoder_ids = None
    # Настройка моделей
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    

    
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    
    # Оберните модели в DataParallel для использования нескольких GPU
    # teacher_model = torch.nn.DataParallel(teacher_model, device_ids=[rank])
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    sampler_test = DistributedSampler(ds_test, num_replicas=world_size, rank=rank)
    student_model = DDP(student_model, device_ids=[rank], output_device=rank)

    dataloader = DataLoader(ds, batch_size=8, collate_fn=data_collator, pin_memory=True, persistent_workers=True, num_workers=12, sampler=sampler)


    dataloader_test = DataLoader(ds_test, batch_size=8, collate_fn=data_collator, pin_memory=True, persistent_workers=True, num_workers=12, sampler=sampler_test)

    
    # Определение оптимизатора
    learning_rate = 1e-5
    lr_scheduler_warmup_steps=50
    optimizer_betas=(0.9, 0.999)
    optimizer_epsilon = 1e-8
    optimizer = AdamW(student_model.parameters(), lr=learning_rate, betas=optimizer_betas, eps=optimizer_epsilon)
    scheduler = get_scheduler(
        'constant_with_warmup',
        optimizer,
        num_warmup_steps=lr_scheduler_warmup_steps,
    )
    
    # Определение функции потерь
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Обучающий цикл
    num_epochs = 10
    teacher_model.eval()
    student_model.train()
    
    # Инициализация wandb
    # Задание конфигурации (гиперпараметров)
    # config = wandb.config
    # config.learning_rate = learning_rate
    # config.num_epochs = num_epochs
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0
        for batch in tqdm.tqdm(dataloader, desc="Batch", leave=False):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Teacher predictions (логиты teacher модели)
            try:
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_features, labels=labels)
                    teacher_logits = teacher_outputs.logits
            except:
                continue
    
            # Student predictions (логиты student модели)
            student_outputs = student_model(input_features, labels=labels)
            student_logits = student_outputs.logits
    
            # Loss calculation (дистилляция + файнтюнинг)
            # Дистилляционная компонента потерь (Mean Squared Error между логитами teacher и student моделей)
            distillation_loss = torch.nn.functional.mse_loss(student_logits, teacher_logits)
    
            # Файнтюнинговая компонента потерь (кросс-энтропия между предсказаниями student модели и реальными метками)
            fine_tuning_loss = loss_fn(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
            # Общая потеря
            loss = distillation_loss + fine_tuning_loss
            epoch_loss += loss.item()
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            # cleanup()
        # Логирование потерь в wandb после каждой эпохи
        avg_epoch_loss = epoch_loss / len(dataloader)
        # wandb.log({"epoch": epoch, "loss": avg_epoch_loss})
    
        print(f"Epoch: {epoch}, Loss: {avg_epoch_loss}")
        val_loss, val_wer = validate(student_model, dataloader_test, device, loss_fn)
        print(f"Validation Loss: {val_loss}, Validation WER: {val_wer}")

        # Сохранение модели после каждой эпохи
        student_model.module.save_pretrained(f"whisperv2-base-distilled-epoch-{epoch}")
    cleanup()

# Завершение сеанса wandb



def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    # wandb.finish()

if __name__ == "__main__":
    main()
    # wandb.finish()
