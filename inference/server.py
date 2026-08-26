import io, os, threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from peft import PeftModel
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv('LATIDOS_BASE_MODEL', 'canopylabs/orpheus-3b-0.1-pretrained')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if DEVICE == 'cuda' else torch.float32)

app = FastAPI(title='LatidosIA Voice LoRA Inference', version='1.3.0')
_lock = threading.Lock()
_state = {'adapter': None, 'model': None, 'tokenizer': None, 'snac': None}


def load_snac():
    if _state['snac'] is None:
        _state['snac'] = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').to(DEVICE).eval()
    return _state['snac']


def load_model(adapter_path: str):
    adapter_path = str(Path(adapter_path).resolve())
    if _state['adapter'] == adapter_path and _state['model'] is not None:
        return _state['model'], _state['tokenizer']

    if DEVICE != 'cuda':
        raise RuntimeError('La inferencia LoRA local está configurada para GPU CUDA. En CPU sería demasiado lenta.')

    if not Path(adapter_path).exists():
        raise FileNotFoundError(f'Adaptador no encontrado: {adapter_path}')

    if _state['model'] is not None:
        del _state['model']
        _state['model'] = None
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map='cuda',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    _state.update({'adapter': adapter_path, 'model': model, 'tokenizer': tokenizer})
    return model, tokenizer


def decode_snac(generated_ids, snac_model):
    token_to_find = 128257
    token_to_remove = 128258
    indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    if len(indices) > 1 and len(indices[1]) > 0:
        last_idx = indices[1][-1].item()
        cropped = generated_ids[:, last_idx + 1:]
    else:
        cropped = generated_ids

    row = cropped[0]
    row = row[row != token_to_remove]
    valid = row[(row >= 128266) & (row < 128266 + 7 * 4096)]
    new_length = (valid.numel() // 7) * 7
    if new_length < 7:
        raise RuntimeError('Orpheus no produjo suficientes tokens de audio válidos.')
    code_list = [int(x.item()) - 128266 for x in valid[:new_length]]

    l1, l2, l3 = [], [], []
    for i in range(0, len(code_list), 7):
        frame = code_list[i:i+7]
        l1.append(frame[0])
        l2.extend([frame[1] - 4096, frame[4] - 16384])
        l3.extend([frame[2] - 8192, frame[3] - 12288, frame[5] - 20480, frame[6] - 24576])

    codes = [
        torch.tensor(l1, device=DEVICE).unsqueeze(0),
        torch.tensor(l2, device=DEVICE).unsqueeze(0),
        torch.tensor(l3, device=DEVICE).unsqueeze(0),
    ]
    with torch.inference_mode():
        audio = snac_model.decode(codes).squeeze().detach().float().cpu().numpy()
    return np.clip(audio, -1.0, 1.0)


def make_prompt(text, tokenizer):
    text_ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids
    start = torch.tensor([[128259]], dtype=torch.long)
    end = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.long)
    return torch.cat([start, text_ids, end], dim=1).to(DEVICE)


@app.get('/health')
def health():
    return {
        'ok': True,
        'version': '1.3.0',
        'device': DEVICE,
        'cuda': torch.cuda.is_available(),
        'active_adapter': _state['adapter'],
        'base_model': BASE_MODEL,
    }


@app.post('/v1/audio/speech')
def speech(payload: dict):
    text = (payload.get('input') or '').strip()
    adapter = payload.get('voice_adapter')
    if not text:
        raise HTTPException(400, 'Falta input')
    if not adapter:
        raise HTTPException(400, 'Falta voice_adapter')

    temperature = float(payload.get('temperature', 0.7))
    top_p = float(payload.get('top_p', 0.9))
    repetition_penalty = float(payload.get('repetition_penalty', 1.15))
    max_new_tokens = int(payload.get('max_new_tokens', max(700, min(5000, len(text) * 18))))

    try:
        with _lock:
            model, tokenizer = load_model(adapter)
            snac_model = load_snac()
            input_ids = make_prompt(text, tokenizer)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=128258,
                    use_cache=True,
                )
            audio = decode_snac(generated, snac_model)
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format='WAV', subtype='PCM_16')
            return Response(buf.getvalue(), media_type='audio/wav')
    except Exception as e:
        raise HTTPException(500, str(e))
