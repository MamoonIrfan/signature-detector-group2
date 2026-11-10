"""Build an isolated Colab environment without replacing the notebook's PyTorch."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path
from common import YOLOX_COMMIT,write_json,safe_relative,sha256_file


def download(url,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".part")
    request=urllib.request.Request(url,headers={"User-Agent":"signature-detection-coursework/1.0"})
    with urllib.request.urlopen(request,timeout=120) as response,temporary.open("wb") as output:
        shutil.copyfileobj(response,output)
    os.replace(temporary,destination)
    return destination


def checked(command,**kwargs):
    subprocess.run([str(c) for c in command],check=True,**kwargs)


def setup(project,device="auto"):
    project=Path(project).resolve();project.mkdir(parents=True,exist_ok=True)
    if sys.platform!="linux":
        raise RuntimeError("Run training in Linux: Colab, Kaggle, GitHub Actions or WSL2.")
    checked([sys.executable,"-m","pip","install","--disable-pip-version-check","-q","uv==0.7.13"])
    if device not in ("auto","cpu","cuda","cuda:0"):
        raise ValueError("device must be auto, cpu or cuda")
    gpu_present=False
    if device=="auto":
        try:
            result=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True,timeout=10)
            gpu_present=result.returncode==0 and bool(result.stdout.strip())
        except (FileNotFoundError,subprocess.TimeoutExpired):
            pass
    variant="cuda" if device in ("cuda","cuda:0") or (device=="auto" and gpu_present) else "cpu"
    environment=project/(".runtime-py310-"+variant)
    python=environment/"bin"/"python"
    if not python.is_file():
        checked([sys.executable,"-m","uv","venv","--python","3.10",environment])
    requirements=project/"requirements_training.txt"
    fingerprint=sha256_file(requirements)
    ready=environment/"signature_ready.json"
    complete=ready.is_file() and json.loads(ready.read_text()).get("requirements_sha256")==fingerprint and json.loads(ready.read_text()).get("variant")==variant
    if not complete:
        print("Installing pinned PyTorch",variant,"wheels; first setup can take several minutes.",flush=True)
        index="https://download.pytorch.org/whl/"+("cu118" if variant=="cuda" else "cpu")
        checked([sys.executable,"-m","uv","pip","install","--python",python,"--no-cache",
                 "--index-url",index,"torch==2.6.0","torchvision==0.21.0"])
        checked([sys.executable,"-m","uv","pip","install","--python",python,"--no-cache","-r",requirements])
        write_json(ready,{"requirements_sha256":fingerprint,"python":"3.10","torch":"2.6.0","torchvision":"0.21.0","variant":variant})
    source=project/"vendor"/"YOLOX"
    revision_file=source/".source_revision"
    if not revision_file.is_file():
        if source.exists() and any(source.iterdir()):
            raise ValueError("An unverified YOLOX source folder exists. Choose a new project directory.")
        archive=download("https://github.com/Megvii-BaseDetection/YOLOX/archive/"+YOLOX_COMMIT+".zip",project/"downloads"/"yolox_source.zip")
        source.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if member.is_dir():continue
                relative=safe_relative(member.filename)
                parts=relative.parts[1:]
                if not parts:continue
                target=source.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True)
                with zipped.open(member) as input_stream,target.open("wb") as output:shutil.copyfileobj(input_stream,output)
        if "Apache License" not in (source/"LICENSE").read_text(encoding="utf-8"):
            raise RuntimeError("The pinned YOLOX source license could not be verified")
        revision_file.write_text(YOLOX_COMMIT,encoding="utf-8")
    if revision_file.read_text().strip()!=YOLOX_COMMIT:
        raise RuntimeError("YOLOX source revision does not match this notebook")
    runtime={"python":str(python),"source_dir":str(source),"yolox_commit":YOLOX_COMMIT,"device":variant}
    write_json(project/"runtime.json",runtime)
    env={**os.environ,"PYTHONPATH":str(project)+os.pathsep+str(source),"YOLOX_SOURCE_DIR":str(source)}
    probe="import os,sys,torch,torchvision\nfrom modeling import build_model\nfrom common import resolve_device_name\ndevice=resolve_device_name(sys.argv[1],torch.cuda.is_available())\nif device=='cpu':torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))\nboxes=torch.tensor([[0.,0.,2.,2.]],device=device)\ntorchvision.ops.nms(boxes,torch.tensor([.9],device=device),.5)\nmodel=build_model('s').to(device).eval()\nsize=640 if device.startswith('cuda') else 128\nwith torch.inference_mode():\n output=model(torch.zeros(1,3,size,size,device=device))\nexpected=sum((size//stride)**2 for stride in (8,16,32))\nassert tuple(output.shape)==(1,expected,6),output.shape\nmodel.train()\nimages=torch.zeros(2,3,128,128,device=device)\ntargets=torch.zeros(2,50,5,device=device);targets[:,0,:]=torch.tensor([0,64,64,20,12],device=device)\nwith torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',enabled=device.startswith('cuda')):\n loss=model(images,targets)['total_loss']\nassert torch.isfinite(loss),'Model smoke-check loss is non-finite'\nloss.backward()\nprint('Device:',device,'| YOLOX forward/backward and NMS checks passed.')\n"
    checked([python,"-c",probe,variant],cwd=project,env=env)
    print("Isolated YOLOX environment ready. No Ultralytics package is used.",flush=True)
    return runtime

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    setup(Path(__file__).resolve().parent,parser.parse_args().device)
