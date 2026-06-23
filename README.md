# DETR-VGG Fusion for Bivalvia Parasite Detection

Penelitian deteksi penyakit parasit pada cangkang kerang Bivalvia menggunakan kombinasi DETR (Detection Transformer) dan VGG19.

## Dataset
Dataset menggunakan format COCO.

## Model Architecture
- DETR
- VGG19 Backbone Fusion
- Hungarian Matching
- GIoU + L1 Loss

## Project Structure

main/
- train.py
- evaluate.py
- inference.py

model/
- detr_fusion.py
- vgg_feature.py

loss/
- detr_loss.py

tests/
- unit testing files

evaluation/
- confusion matrix
- evaluation metrics

## Training

```bash
python main/train.py