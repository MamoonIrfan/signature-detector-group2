"""COCO evaluation with the standard pycocotools evaluator, no compiled YOLOX extension."""
from __future__ import annotations
import contextlib, io, time
from pathlib import Path
from common import write_json
from geometry import operating_metrics


def coco_metrics(coco_dict,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        ground=COCO(); ground.dataset=coco_dict; ground.createIndex()
        if predictions:
            proposed=ground.loadRes(predictions)
        else:
            proposed=COCO(); proposed.dataset={"images":coco_dict["images"],"categories":coco_dict["categories"],"annotations":[]}; proposed.createIndex()
        evaluator=COCOeval(ground,proposed,"bbox")
        evaluator.params.imgIds=[row["id"] for row in coco_dict["images"]]
        evaluator.params.catIds=[1]
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    return {"mAP50_95":max(0,float(evaluator.stats[0])),"mAP50":max(0,float(evaluator.stats[1])),
            "average_recall_100":max(0,float(evaluator.stats[8]))},capture.getvalue()


def evaluate(model,dataset,config,output,split="val"):
    import torch
    from tqdm.auto import tqdm
    from data_loading import make_loader
    from modeling import postprocessing
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    was_training=model.training
    model.eval()
    predictions=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for images,records in tqdm(make_loader(dataset,config["batch_size"],config.get("workers",2)),desc=split+" evaluation"):
            images=images.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,enabled=bool(config.get("fp16",True) and device.type=="cuda")):
                outputs=model(images)
            results=postprocessing(outputs.float(),0.001,config["nms_threshold"])
            for detections,row in zip(results,records):
                if detections is None:
                    continue
                boxes=detections.detach().float().cpu().numpy()
                # NMS is score ordered. Cap to COCO's maximum detection count per image.
                order=(boxes[:,4]*boxes[:,5]).argsort()[::-1][:100]
                for box in boxes[order]:
                    x1=max(0,min(row["width"],float(box[0])/row["ratio"]))
                    y1=max(0,min(row["height"],float(box[1])/row["ratio"]))
                    x2=max(0,min(row["width"],float(box[2])/row["ratio"]))
                    y2=max(0,min(row["height"],float(box[3])/row["ratio"]))
                    if x2<=x1 or y2<=y1:
                        continue
                    predictions.append({"image_id":row["image_id"],"category_id":1,
                        "bbox":[x1,y1,x2-x1,y2-y1],"score":float(box[4]*box[5])})
    metrics,summary=coco_metrics(dataset.coco,predictions)
    metrics.update(operating_metrics(predictions,dataset.coco["annotations"],config["conf_threshold"],0.5))
    metrics.update(images=len(dataset),candidate_confidence=0.001,nms_threshold=config["nms_threshold"],
                   elapsed_seconds=time.perf_counter()-started)
    write_json(output/(split+"_metrics.json"),metrics)
    write_json(output/(split+"_predictions.json"),predictions)
    (output/(split+"_coco_summary.txt")).write_text(summary,encoding="utf-8")
    model.train(was_training)
    return metrics
