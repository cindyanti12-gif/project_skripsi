import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from transformers import DetrImageProcessor


class CocoDetectionDataset(Dataset):

    def __init__(self, image_dir, annotation_file):

        self.image_dir = image_dir
        self.coco = COCO(annotation_file)

        self.image_ids = list(self.coco.imgs.keys())

        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
            size={"shortest_edge": 480, "longest_edge": 640}
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image_info = self.coco.loadImgs(image_id)[0]

        image_path = os.path.join(
            self.image_dir,
            image_info["file_name"]
        )

        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        target = {
            "image_id": image_id,
            "annotations": anns
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze()

        labels = encoding["labels"][0]

        return pixel_values, labels