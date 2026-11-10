"""Download a pinned HF snapshot; validate COCO and image/split integrity."""
from __future__ import annotations
import collections, concurrent.futures, hashlib, json, math, os, shutil, stat, zipfile
from pathlib import Path
from common import read_json, write_json, sha256_file, safe_relative

SCHEMA_VERSION = 1
SPLITS = {"train":"train2017", "val":"val2017", "test":"test2017"}


def validate_coco(raw):
    category_names = {int(c["id"]): str(c["name"]).strip().lower() for c in raw["categories"]}
    if len(category_names) != 1 or set(category_names.values()) != {"signature"}:
        raise ValueError("Expected exactly one COCO category named signature")
    images, image_ids, names = [], {}, set()
    for image in raw["images"]:
        row = dict(image)
        name = str(safe_relative(row["file_name"]))
        if int(row["id"]) in image_ids or name in names:
            raise ValueError("Duplicate image ID or filename in a COCO split")
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise ValueError("Invalid image dimensions")
        row.update(id=int(row["id"]),file_name=name,width=int(row["width"]),height=int(row["height"]))
        images.append(row); image_ids[row["id"]]=row; names.add(name)
    annotations, seen, clipped = [], set(), 0
    for annotation in raw["annotations"]:
        row = dict(annotation)
        if row["id"] in seen or row["image_id"] not in image_ids or int(row["category_id"]) not in category_names:
            raise ValueError("Duplicate or orphan annotation")
        seen.add(row["id"])
        if row.get("iscrowd",0):
            raise ValueError("Crowd annotations are not supported in this single-signature pipeline")
        box = row["bbox"]
        if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("Invalid COCO box")
        x,y,w,h=map(float,box); image=image_ids[row["image_id"]]
        x1,y1=max(0,x),max(0,y); x2,y2=min(image["width"],x+w),min(image["height"],y+h)
        if w<=0 or h<=0 or x2<=x1 or y2<=y1:
            raise ValueError("A COCO box has zero/negative area or lies outside the image")
        clipped += int((x1,y1,x2,y2)!=(x,y,x+w,y+h))
        row.update(category_id=1,bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0)
        row.pop("segmentation",None)
        annotations.append(row)
    result={"info":raw.get("info",{}),"images":images,"annotations":annotations,
            "categories":[{"id":1,"name":"signature","supercategory":"signature"}]}
    return result, clipped


def extract_images(archive_path, destination, split_directory):
    destination=Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative=safe_relative(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Dataset ZIP contains a symbolic link")
            if relative.parts[0] != split_directory:
                raise ValueError("Expected ZIP members beneath " + split_directory + "/")
            target=destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_file() and target.stat().st_size==member.file_size:
                continue
            temporary=target.with_name(target.name+".part")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source,output)
            os.replace(temporary,target)


def _check_image(row):
    from PIL import Image
    path,width,height=row
    if not path.is_file():
        raise FileNotFoundError("COCO image missing: "+str(path))
    with Image.open(path) as image:
        if image.size!=(width,height):
            raise ValueError("Pixel dimensions disagree with annotations: "+str(path))
        image.load()
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Normalize EXIF orientation and corresponding boxes before training: '+str(path))
        pixels=image.convert('RGB')
        digest=hashlib.sha256(str(pixels.size).encode()+pixels.tobytes()).hexdigest()
    return digest


def audit_local_dataset(dataset, verify_pixels=True):
    dataset=Path(dataset)
    reports, hash_splits = {}, {}
    for split,directory in SPLITS.items():
        path=dataset/"annotations"/("instances_"+directory+".json")
        coco,clipped=validate_coco(read_json(path))
        counts=collections.Counter(a["image_id"] for a in coco["annotations"])
        paths=[(dataset/directory/i["file_name"],i["width"],i["height"]) for i in coco["images"]]
        if verify_pixels:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                hashes=list(pool.map(_check_image,paths))
            for digest in hashes:
                previous=hash_splits.get(digest)
                if previous and previous!=split:
                    raise ValueError("Identical decoded pixels occur across "+previous+" and "+split+". Fix the dataset split before training.")
                hash_splits[digest]=split
        else:
            hashes=[]
        reports[split]={"images":len(coco["images"]),"boxes":len(coco["annotations"]),
                       "negative_images":sum(counts[i["id"]]==0 for i in coco["images"]),
                       "max_boxes_per_image":max(counts.values(),default=0),"clipped_boxes":clipped,
                       "annotation_sha256":sha256_file(path)}
        write_json(path,coco)
        reports[split]["annotation_sha256"]=sha256_file(path)
    return {"splits":reports,"all_pixels_decoded":verify_pixels,"exact_split_overlap_checked":verify_pixels,
            "near_duplicate_audit":"Exact decoded pixels checked; near-duplicates and form-template overlap need separate review"}


def prepare_dataset(project, repo, revision="main", smoke_test=False):
    if smoke_test:
        raise ValueError('Day 2 uses complete train/val/test splits. Dataset subsetting is disabled.')
    from huggingface_hub import HfApi, snapshot_download
    project=Path(project)
    token=os.environ.get("HF_TOKEN") or None
    info=HfApi(token=token).dataset_info(repo_id=repo,revision=revision)
    resolved=info.sha
    if not resolved:
        raise RuntimeError("Hugging Face did not return an immutable dataset revision")
    source=project/"downloads"/resolved
    dataset=project/"datasets"/(resolved+("_smoke" if smoke_test else ""))
    dataset.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(project).free < 12*1024**3 and not (dataset/"audit.json").is_file():
        raise RuntimeError("At least 12 GiB free disk is required for the downloaded and extracted dataset")
    print("Downloading HF dataset:",repo,"revision:",resolved,flush=True)
    snapshot_download(repo_id=repo,repo_type="dataset",revision=resolved,token=token,local_dir=str(source),
        allow_patterns=["README.md","annotations/*.json","train2017.zip","val2017.zip","test2017.zip",
                        "train2017/**","val2017/**","test2017/**"],max_workers=3)
    source_clips={}
    for split,directory in SPLITS.items():
        original=source/"annotations"/("instances_"+directory+".json")
        if not original.is_file():
            raise FileNotFoundError('Cleaned snapshot must contain annotations/'+original.name+
                                    '. Inspect list_repo_files; no older dataset is substituted.')
        coco,source_clips[split]=validate_coco(read_json(original))
        if (source/(directory+'.zip')).is_file():
            extract_images(source/(directory+'.zip'),dataset,directory)
        elif (source/directory).is_dir():
            shutil.copytree(source/directory,dataset/directory,dirs_exist_ok=True)
        else:
            raise FileNotFoundError('Missing '+directory+'.zip or '+directory+'/ in pinned snapshot')
        if not coco['images']:
            raise ValueError('The '+split+' split is empty')
        write_json(dataset/"annotations"/original.name,coco)
    report=audit_local_dataset(dataset,verify_pixels=True)
    for split in SPLITS:
        report["splits"][split]["source_boxes_clipped"]=source_clips[split]
    card=(source/"README.md").read_text(encoding="utf-8") if (source/"README.md").is_file() else ""
    (dataset/"DATASET_CARD.md").write_text(card,encoding="utf-8")
    card_data=getattr(info,"card_data",None) or {}
    license_value=card_data.get("license","unspecified") if hasattr(card_data,"get") else "unspecified"
    report.update(schema_version=SCHEMA_VERSION,repo=repo,revision=resolved,smoke_test=smoke_test,
        dataset_license=str(license_value),license_note='License copied from the pinned dataset card at runtime')
    write_json(dataset/"audit.json",report)
    return dataset,report
