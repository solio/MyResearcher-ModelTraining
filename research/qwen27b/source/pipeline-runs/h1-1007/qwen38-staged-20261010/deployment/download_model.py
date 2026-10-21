import json,time,os
from pathlib import Path
from huggingface_hub import snapshot_download,model_info
repo="Qwen/Qwen3.8-27B"
rev="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
p=Path("models/Qwen3.8-27B");p.mkdir(parents=True,exist_ok=True)
start=time.time()
info=model_info(repo,revision=rev,files_metadata=True)
Path("logs/model-manifest.json").write_text(json.dumps({"repo":repo,"revision":info.sha,"started_at_epoch":start,"files":[{"path":x.rfilename,"size":x.size,"lfs":x.lfs.__dict__ if x.lfs else None} for x in info.siblings]},indent=2))
print("Downloading",repo,rev,flush=True)
snapshot_download(repo_id=repo,revision=rev,local_dir=p,max_workers=4,allow_patterns=["*.json","*.safetensors","*.jinja","*.txt","*.model"])
print("DOWNLOAD_COMPLETE",time.time()-start,flush=True)
