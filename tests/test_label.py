from dataset_loader import CocoDetectionDataset

dataset = CocoDetectionDataset(
    image_dir="dataset/train",
    annotation_file="dataset/train/_annotations.coco.json"
)

pixel_values, labels = dataset[0]

print("Class Labels:")
print(labels["class_labels"])

print("\nBoxes:")
print(labels["boxes"])