# DETR-VGG Fusion for Bivalvia Parasite Detection

Penelitian deteksi penyakit parasit pada cangkang kerang Bivalvia menggunakan kombinasi DETR dan VGG19.

## Dataset
- Format COCO
- 2 kelas:
  - parasite
  - non_parasite

## Model
- DETR
- VGG19 Feature Extractor
- Hungarian Matching
- GIoU Loss + L1 Loss

## Training

```bash
python main/train.py
```

## Evaluation

```bash
python main/evaluate.py
```

## Inference

```bash
python main/inference.py
```