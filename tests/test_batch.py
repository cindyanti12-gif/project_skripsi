from torch.utils.data import DataLoader

from dataset_loader import CocoDetectionDataset
from collate_fn import collate_fn

dataset = CocoDetectionDataset(
    image_dir="dataset/train",
    annotation_file="dataset/train/_annotations.coco.json"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

images, labels = next(iter(loader))

print(images.shape)

print(type(labels))
print(len(labels))