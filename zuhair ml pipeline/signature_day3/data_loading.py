"""Document-safe training transforms and batch loading."""
from __future__ import annotations
import collections, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import read_json
from modeling import preprocessing


class SignatureDataset(Dataset):
    def __init__(self,root,split,size=640,training=False):
        self.root=Path(root); self.split=split; self.size=size; self.training=training; self.augment=training
        self.coco=read_json(self.root/"annotations"/("instances_"+split+"2017.json"))
        self.images=self.coco["images"]
        self.boxes=collections.defaultdict(list)
        for ann in self.coco["annotations"]:
            self.boxes[ann["image_id"]].append(ann)
        self.max_labels=max(50,max((len(v) for v in self.boxes.values()),default=0))

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        import cv2
        row=self.images[index]
        image=cv2.imread(str(self.root/(self.split+"2017")/row["file_name"]))
        if image is None:
            raise ValueError("Cannot decode "+row["file_name"])
        if image.shape[1]!=row["width"] or image.shape[0]!=row["height"]:
            raise ValueError("Image dimensions changed after dataset audit")
        if not self.training:
            processed,ratio=preprocessing(image,self.size)
            return torch.from_numpy(processed),{"image_id":row["id"],"width":row["width"],"height":row["height"],"ratio":ratio}
        from yolox.data.data_augment import TrainTransform, preproc, random_affine
        targets=np.asarray([[a["bbox"][0],a["bbox"][1],a["bbox"][0]+a["bbox"][2],a["bbox"][1]+a["bbox"][3],0]
                            for a in self.boxes[row["id"]]],dtype=np.float32).reshape(-1,5)
        if self.augment:
            # Letterbox first so affine augmentation never crops away an entire high-resolution page.
            canvas,ratio=preproc(image,(self.size,self.size),swap=(0,1,2))
            targets[:,:4]*=ratio
            image,targets=random_affine(canvas.astype(np.uint8),targets,target_size=(self.size,self.size),
                degrees=3.0,translate=0.03,scales=(0.90,1.10),shear=0.0)
            if len(targets):
                targets=targets[(targets[:,2]>targets[:,0]) & (targets[:,3]>targets[:,1])]
        transform=TrainTransform(max_labels=self.max_labels,flip_prob=0.0,hsv_prob=0.20 if self.augment else 0.0)
        processed,labels=transform(image,targets,(self.size,self.size))
        return torch.from_numpy(processed),torch.from_numpy(labels)


def eval_collate(rows):
    return torch.stack([r[0] for r in rows]),[r[1] for r in rows]


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


def make_loader(dataset,batch_size,workers=2,seed=42,epoch=0):
    generator=torch.Generator().manual_seed(seed+epoch)
    return DataLoader(dataset,batch_size=batch_size,shuffle=dataset.training,num_workers=workers,
        pin_memory=torch.cuda.is_available(),drop_last=False,generator=generator,
        worker_init_fn=seed_worker,persistent_workers=False,
        collate_fn=None if dataset.training else eval_collate)
