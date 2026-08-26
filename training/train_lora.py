import argparse, json, os
from pathlib import Path

import torch
import torchaudio
from datasets import Dataset
from snac import SNAC
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from unsloth import FastLanguageModel

BASE_AUDIO_TOKEN = 128266
OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]


def read_manifest(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if not rows: raise ValueError('Dataset vacío')
    return rows


def encode_audio(path, snac, device):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000: wav = torchaudio.functional.resample(wav, sr, 24000)
    wav = wav.unsqueeze(0).to(device)
    with torch.inference_mode(): codes = snac.encode(wav)
    result=[]
    frames=codes[0].shape[1]
    for i in range(frames):
        vals=[codes[0][0,i],codes[1][0,2*i],codes[2][0,4*i],codes[2][0,4*i+1],codes[1][0,2*i+1],codes[2][0,4*i+2],codes[2][0,4*i+3]]
        result.extend(int(v.item()) + BASE_AUDIO_TOKEN + OFFSETS[j] for j,v in enumerate(vals))
    return result


def build_example(row, tokenizer, snac, device, max_seq_length):
    text=row['text'].strip()
    codes=encode_audio(row['audio'], snac, device)
    text_ids=tokenizer.encode(text, add_special_tokens=False)
    # Orpheus special-token layout used by public fine-tuning implementations.
    ids=[128259,128000] + text_ids + [128009,128260,128261,128257] + codes + [128258,128262]
    ids=ids[:max_seq_length]
    return {'input_ids':ids,'labels':ids.copy(),'attention_mask':[1]*len(ids)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base-model', default='canopylabs/orpheus-3b-0.1-pretrained')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--max-seq-length', type=int, default=4096)
    args=p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('El entrenamiento LoRA requiere una GPU CUDA en esta configuración.')
    device='cuda'
    rows=read_manifest(args.dataset)
    tokenizer=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token

    snac=SNAC.from_pretrained('hubertsiuzdak/snac_24khz').to(device).eval()
    prepared=[]
    for i,row in enumerate(rows,1):
        print(f'[dataset] {i}/{len(rows)} {Path(row["audio"]).name}', flush=True)
        prepared.append(build_example(row,tokenizer,snac,device,args.max_seq_length))
    del snac
    torch.cuda.empty_cache()
    ds=Dataset.from_list(prepared)

    model,_=FastLanguageModel.from_pretrained(model_name=args.base_model,max_seq_length=args.max_seq_length,dtype=None,load_in_4bit=True)
    model=FastLanguageModel.get_peft_model(model,r=args.lora_r,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_alpha=args.lora_r*2,lora_dropout=0,bias='none',use_gradient_checkpointing='unsloth',random_state=3407)

    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ta=TrainingArguments(output_dir=str(out),per_device_train_batch_size=args.batch_size,gradient_accumulation_steps=args.grad_accum,max_steps=args.steps,learning_rate=args.lr,warmup_steps=min(10,max(1,args.steps//20)),logging_steps=1,save_strategy='steps',save_steps=max(25,args.steps//4),save_total_limit=2,optim='adamw_8bit',fp16=not torch.cuda.is_bf16_supported(),bf16=torch.cuda.is_bf16_supported(),report_to='none',seed=3407)
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,return_tensors='pt')
    trainer=Trainer(model=model,args=ta,train_dataset=ds,data_collator=collator)
    trainer.train()
    adapter=out/'adapter'
    model.save_pretrained(str(adapter)); tokenizer.save_pretrained(str(adapter))
    print(f'LATIDOS_ADAPTER={adapter.resolve()}', flush=True)

if __name__=='__main__': main()
