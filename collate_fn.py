import torch


def collate_fn(batch):

    pixel_values = []
    labels = []

    for item in batch:
        pixel_values.append(item[0])
        labels.append(item[1])

    max_h = max(img.shape[1] for img in pixel_values)
    max_w = max(img.shape[2] for img in pixel_values)

    padded_images = []

    for img in pixel_values:

        c, h, w = img.shape

        padded = torch.zeros(
            (c, max_h, max_w),
            dtype=img.dtype
        )

        padded[:, :h, :w] = img

        padded_images.append(padded)

    pixel_values = torch.stack(padded_images)

    return pixel_values, labels