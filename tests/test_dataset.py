import json

for split in ["train","val","test"]:
    with open(f"dataset/{split}/_annotations.coco.json") as f:
        data = json.load(f)

    print(split, len(data["images"]))