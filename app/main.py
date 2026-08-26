from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import base64, json, os, requests, uuid, shutil, time

BASE = Path(__file__).resolve().parent.parent
DATA = BASE/"data"
VOICE_ROOT = DATA/"voices"
OUT = BASE/"outputs"
TRAINING = BASE/"training"
DB = DATA/"voices.json"

for p in [VOICE_ROOT, OUT, TRAINING]:
    p.mkdir(parents=True, exist_ok=True)
if not DB.exists():
    DB.write_text("[]", encoding="utf-8")

ORPHEUS_URL = os.getenv("ORPHEUS_URL","http://127.0.0.1:5005")
app = FastAPI(title="LatidosIA Voice Multi", version="1.1.0")

def load_db():
    return json.loads(DB.read_text(encoding="utf-8"))

def save_db(x):
    DB.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def get_voice(voice_id):
    v = next((x for x in load_db() if x["id"] == voice_id), None)
    if not v:
        raise HTTPException(404, "Voz no encontrada")
    return v

@app.get("/api/health")
def health():
    try:
        r = requests.get(f"{ORPHEUS_URL}/health", timeout=3)
        return {"app":True,"orpheus":r.ok,"url":ORPHEUS_URL,"version":"1.1.0"}
    except Exception as e:
        return {"app":True,"orpheus":False,"url":ORPHEUS_URL,"version":"1.1.0","detail":str(e)}

@app.get("/api/voices")
def voices():
    return load_db()

@app.post("/api/voices")
def create_voice(payload: dict):
    name=(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400,"Nombre obligatorio")
    vid=uuid.uuid4().hex[:10]
    folder=VOICE_ROOT/vid
    folder.mkdir(parents=True, exist_ok=True)
    item={
        "id":vid,
        "name":name,
        "language":payload.get("language") or "es",
        "style":payload.get("style") or "natural",
        "notes":payload.get("notes") or "",
        "created_at":int(time.time()),
        "training_status":"not_trained",
        "adapter_path":None,
        "samples":[]
    }
    db=load_db(); db.append(item); save_db(db)
    return item

@app.patch("/api/voices/{voice_id}")
def update_voice(voice_id: str, payload: dict):
    db=load_db()
    found=False
    for v in db:
        if v["id"] == voice_id:
            for k in ["name","language","style","notes","training_status","adapter_path"]:
                if k in payload:
                    v[k]=payload[k]
            found=True
    if not found:
        raise HTTPException(404,"Voz no encontrada")
    save_db(db)
    return next(v for v in db if v["id"]==voice_id)

@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    db=load_db()
    v=next((x for x in db if x["id"]==voice_id),None)
    if not v:
        raise HTTPException(404,"Voz no encontrada")
    folder=VOICE_ROOT/voice_id
    if folder.exists():
        shutil.rmtree(folder)
    save_db([x for x in db if x["id"]!=voice_id])
    return {"ok":True}

@app.post("/api/voices/{voice_id}/samples")
async def add_sample(voice_id: str, transcript: str = Form(...), audio: UploadFile = File(...)):
    db=load_db()
    v=next((x for x in db if x["id"]==voice_id),None)
    if not v:
        raise HTTPException(404,"Voz no encontrada")
    ext=Path(audio.filename or "reference.wav").suffix.lower() or ".wav"
    if ext not in [".wav",".mp3",".m4a",".flac",".ogg"]:
        raise HTTPException(400,"Formato no admitido")
    sid=uuid.uuid4().hex[:10]
    folder=VOICE_ROOT/voice_id
    folder.mkdir(parents=True, exist_ok=True)
    target=folder/f"{sid}{ext}"
    with target.open("wb") as f:
        shutil.copyfileobj(audio.file,f)
    sample={
        "id":sid,
        "transcript":transcript.strip(),
        "audio_path":str(target.relative_to(BASE)).replace("\\","/"),
        "filename":audio.filename or target.name
    }
    v["samples"].append(sample)
    save_db(db)
    return sample

@app.delete("/api/voices/{voice_id}/samples/{sample_id}")
def delete_sample(voice_id: str, sample_id: str):
    db=load_db()
    v=next((x for x in db if x["id"]==voice_id),None)
    if not v:
        raise HTTPException(404,"Voz no encontrada")
    s=next((x for x in v["samples"] if x["id"]==sample_id),None)
    if not s:
        raise HTTPException(404,"Muestra no encontrada")
    p=BASE/s["audio_path"]
    if p.exists():
        p.unlink()
    v["samples"]=[x for x in v["samples"] if x["id"]!=sample_id]
    save_db(db)
    return {"ok":True}

@app.post("/api/generate")
def generate(payload: dict):
    text=(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400,"Falta el texto")
    voice_id=payload.get("voice_id")
    sample_id=payload.get("sample_id")
    body={"input":text}
    if voice_id:
        v=get_voice(voice_id)
        if v.get("adapter_path") and v.get("training_status")=="trained":
            body["voice_adapter"]=v["adapter_path"]
        else:
            if not v["samples"]:
                raise HTTPException(400,"Esta voz no tiene muestras")
            s = next((x for x in v["samples"] if x["id"]==sample_id), None) if sample_id else v["samples"][0]
            if not s:
                raise HTTPException(404,"Muestra no encontrada")
            body["ref_audio"]=base64.b64encode((BASE/s["audio_path"]).read_bytes()).decode("ascii")
            body["ref_text"]=s["transcript"]
    else:
        body["voice"]=payload.get("preset") or "tara"

    try:
        r=requests.post(f"{ORPHEUS_URL}/v1/audio/speech",json=body,timeout=600)
    except Exception as e:
        raise HTTPException(503,f"Orpheus no disponible: {e}")
    if not r.ok:
        raise HTTPException(r.status_code,r.text[:500])
    fn=OUT/f"latidos_voice_{uuid.uuid4().hex[:8]}.wav"
    fn.write_bytes(r.content)
    return {"ok":True,"url":f"/api/audio/{fn.name}","file":fn.name}

@app.post("/api/voices/{voice_id}/prepare-training")
def prepare_training(voice_id: str):
    v=get_voice(voice_id)
    if len(v["samples"]) < 3:
        raise HTTPException(400,"Añade al menos 3 muestras antes de preparar entrenamiento")
    manifest=[]
    for s in v["samples"]:
        manifest.append({
            "audio":str((BASE/s["audio_path"]).resolve()),
            "text":s["transcript"],
            "speaker":v["name"],
            "language":v["language"],
            "style":v["style"]
        })
    out=TRAINING/f"{voice_id}_dataset.jsonl"
    with out.open("w",encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    db=load_db()
    for x in db:
        if x["id"]==voice_id:
            x["training_status"]="dataset_ready"
            x["dataset_path"]=str(out.relative_to(BASE)).replace("\\","/")
    save_db(db)
    return {"ok":True,"dataset":str(out),"samples":len(manifest)}

@app.get("/api/audio/{filename}")
def audio(filename: str):
    p=OUT/Path(filename).name
    if not p.exists():
        raise HTTPException(404,"No encontrado")
    return FileResponse(p,media_type="audio/wav",filename=p.name)

app.mount("/",StaticFiles(directory=str(BASE/"app"/"static"),html=True),name="static")
