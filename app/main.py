from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import base64, json, os, requests, uuid, shutil, time, subprocess, sys

BASE = Path(__file__).resolve().parent.parent
DATA = BASE/'data'; VOICE_ROOT = DATA/'voices'; OUT = BASE/'outputs'; TRAINING = BASE/'training'; DB = DATA/'voices.json'
for p in [VOICE_ROOT, OUT, TRAINING]: p.mkdir(parents=True, exist_ok=True)
if not DB.exists(): DB.write_text('[]', encoding='utf-8')
ORPHEUS_URL=os.getenv('ORPHEUS_URL','http://127.0.0.1:5005')
TRAIN_PYTHON=os.getenv('TRAIN_PYTHON',sys.executable)
JOBS={}
app=FastAPI(title='LatidosIA Voice Multi',version='1.2.0')

def load_db(): return json.loads(DB.read_text(encoding='utf-8'))
def save_db(x): DB.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def get_voice(voice_id):
    v=next((x for x in load_db() if x['id']==voice_id),None)
    if not v: raise HTTPException(404,'Voz no encontrada')
    return v

def patch_voice(voice_id, **changes):
    db=load_db()
    for v in db:
        if v['id']==voice_id:
            v.update(changes); save_db(db); return v
    raise HTTPException(404,'Voz no encontrada')

@app.get('/api/health')
def health():
    try:
        r=requests.get(f'{ORPHEUS_URL}/health',timeout=3); connected=r.ok
    except Exception: connected=False
    return {'app':True,'orpheus':connected,'url':ORPHEUS_URL,'version':'1.2.0'}

@app.get('/api/voices')
def voices(): return load_db()

@app.post('/api/voices')
def create_voice(payload:dict):
    name=(payload.get('name') or '').strip()
    if not name: raise HTTPException(400,'Nombre obligatorio')
    vid=uuid.uuid4().hex[:10]; (VOICE_ROOT/vid).mkdir(parents=True,exist_ok=True)
    item={'id':vid,'name':name,'language':payload.get('language') or 'es','style':payload.get('style') or 'natural','notes':payload.get('notes') or '','created_at':int(time.time()),'training_status':'not_trained','adapter_path':None,'samples':[]}
    db=load_db(); db.append(item); save_db(db); return item

@app.patch('/api/voices/{voice_id}')
def update_voice(voice_id:str,payload:dict):
    allowed={k:v for k,v in payload.items() if k in ['name','language','style','notes','training_status','adapter_path']}
    return patch_voice(voice_id,**allowed)

@app.delete('/api/voices/{voice_id}')
def delete_voice(voice_id:str):
    get_voice(voice_id); folder=VOICE_ROOT/voice_id
    if folder.exists(): shutil.rmtree(folder)
    save_db([x for x in load_db() if x['id']!=voice_id]); return {'ok':True}

@app.post('/api/voices/{voice_id}/samples')
async def add_sample(voice_id:str, transcript:str=Form(...), audio:UploadFile=File(...)):
    db=load_db(); v=next((x for x in db if x['id']==voice_id),None)
    if not v: raise HTTPException(404,'Voz no encontrada')
    ext=Path(audio.filename or 'reference.wav').suffix.lower() or '.wav'
    if ext not in ['.wav','.mp3','.m4a','.flac','.ogg']: raise HTTPException(400,'Formato no admitido')
    sid=uuid.uuid4().hex[:10]; folder=VOICE_ROOT/voice_id; folder.mkdir(parents=True,exist_ok=True); target=folder/f'{sid}{ext}'
    with target.open('wb') as f: shutil.copyfileobj(audio.file,f)
    v['samples'].append({'id':sid,'transcript':transcript.strip(),'audio_path':str(target.relative_to(BASE)).replace('\\','/'),'filename':audio.filename or target.name})
    v['training_status']='not_trained'; v['adapter_path']=None; save_db(db); return v['samples'][-1]

@app.delete('/api/voices/{voice_id}/samples/{sample_id}')
def delete_sample(voice_id:str,sample_id:str):
    db=load_db(); v=next((x for x in db if x['id']==voice_id),None)
    if not v: raise HTTPException(404,'Voz no encontrada')
    s=next((x for x in v['samples'] if x['id']==sample_id),None)
    if not s: raise HTTPException(404,'Muestra no encontrada')
    p=BASE/s['audio_path'];
    if p.exists(): p.unlink()
    v['samples']=[x for x in v['samples'] if x['id']!=sample_id]; v['training_status']='not_trained'; v['adapter_path']=None; save_db(db); return {'ok':True}

@app.post('/api/voices/{voice_id}/prepare-training')
def prepare_training(voice_id:str):
    v=get_voice(voice_id)
    if len(v['samples'])<3: raise HTTPException(400,'Añade al menos 3 muestras antes de preparar entrenamiento')
    out=TRAINING/f'{voice_id}_dataset.jsonl'
    with out.open('w',encoding='utf-8') as f:
        for s in v['samples']:
            row={'audio':str((BASE/s['audio_path']).resolve()),'text':s['transcript'],'speaker':v['name'],'language':v['language'],'style':v['style']}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    patch_voice(voice_id,training_status='dataset_ready',dataset_path=str(out.relative_to(BASE)).replace('\\','/'))
    return {'ok':True,'dataset':str(out),'samples':len(v['samples'])}

@app.post('/api/voices/{voice_id}/start-training')
def start_training(voice_id:str,payload:dict={}):
    v=get_voice(voice_id)
    if voice_id in JOBS and JOBS[voice_id].poll() is None: raise HTTPException(409,'Esta voz ya se está entrenando')
    if not v.get('dataset_path'):
        prepare_training(voice_id); v=get_voice(voice_id)
    dataset=BASE/v['dataset_path']; out=TRAINING/'runs'/voice_id; out.mkdir(parents=True,exist_ok=True); log=out/'training.log'
    steps=max(10,min(int(payload.get('steps',200)),5000)); lora_r=max(4,min(int(payload.get('lora_r',16)),128)); lr=float(payload.get('lr',2e-4))
    cmd=[TRAIN_PYTHON,str(TRAINING/'train_lora.py'),'--dataset',str(dataset),'--output',str(out),'--steps',str(steps),'--lora-r',str(lora_r),'--lr',str(lr)]
    fh=log.open('w',encoding='utf-8')
    try: proc=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,cwd=str(BASE))
    except Exception as e: fh.close(); raise HTTPException(500,f'No se pudo iniciar entrenamiento: {e}')
    JOBS[voice_id]=proc; patch_voice(voice_id,training_status='training',training_started_at=int(time.time()),training_log=str(log.relative_to(BASE)).replace('\\','/'))
    return {'ok':True,'pid':proc.pid,'steps':steps,'lora_r':lora_r}

@app.get('/api/voices/{voice_id}/training-status')
def training_status(voice_id:str):
    v=get_voice(voice_id); proc=JOBS.get(voice_id); status=v.get('training_status','not_trained')
    if proc is not None:
        rc=proc.poll()
        if rc is None: status='training'
        elif rc==0:
            adapter=TRAINING/'runs'/voice_id/'adapter'; status='trained'; patch_voice(voice_id,training_status='trained',adapter_path=str(adapter.relative_to(BASE)).replace('\\','/'),training_finished_at=int(time.time()))
            JOBS.pop(voice_id,None)
        else:
            status='failed'; patch_voice(voice_id,training_status='failed',training_finished_at=int(time.time())); JOBS.pop(voice_id,None)
    log_text=''
    lp=v.get('training_log')
    if lp and (BASE/lp).exists():
        try: log_text='\n'.join((BASE/lp).read_text(encoding='utf-8',errors='replace').splitlines()[-30:])
        except Exception: pass
    return {'status':status,'adapter_path':get_voice(voice_id).get('adapter_path'),'log':log_text}

@app.post('/api/generate')
def generate(payload:dict):
    text=(payload.get('text') or '').strip()
    if not text: raise HTTPException(400,'Falta el texto')
    voice_id=payload.get('voice_id'); sample_id=payload.get('sample_id'); body={'input':text}
    if voice_id:
        v=get_voice(voice_id)
        # Adapter use depends on the inference server supporting voice_adapter. Otherwise fall back to zero-shot sample.
        if v.get('adapter_path') and v.get('training_status')=='trained' and payload.get('use_adapter',True): body['voice_adapter']=str((BASE/v['adapter_path']).resolve())
        else:
            if not v['samples']: raise HTTPException(400,'Esta voz no tiene muestras')
            s=next((x for x in v['samples'] if x['id']==sample_id),None) if sample_id else v['samples'][0]
            if not s: raise HTTPException(404,'Muestra no encontrada')
            body['ref_audio']=base64.b64encode((BASE/s['audio_path']).read_bytes()).decode('ascii'); body['ref_text']=s['transcript']
    else: body['voice']=payload.get('preset') or 'tara'
    try: r=requests.post(f'{ORPHEUS_URL}/v1/audio/speech',json=body,timeout=600)
    except Exception as e: raise HTTPException(503,f'Orpheus no disponible: {e}')
    if not r.ok: raise HTTPException(r.status_code,r.text[:500])
    fn=OUT/f'latidos_voice_{uuid.uuid4().hex[:8]}.wav'; fn.write_bytes(r.content); return {'ok':True,'url':f'/api/audio/{fn.name}','file':fn.name}

@app.get('/api/audio/{filename}')
def audio(filename:str):
    p=OUT/Path(filename).name
    if not p.exists(): raise HTTPException(404,'No encontrado')
    return FileResponse(p,media_type='audio/wav',filename=p.name)

app.mount('/',StaticFiles(directory=str(BASE/'app'/'static'),html=True),name='static')
