import torch

from scipy.optimize import linear_sum_assignment


class HungarianMatcher:

    def __init__(
        self,
        class_cost=1,
        bbox_cost=5
    ):
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost

    @torch.no_grad()
    def __call__(self, outputs, targets):

        bs = outputs["pred_logits"].shape[0]

        indices = []

        for b in range(bs):

            pred_logits = outputs["pred_logits"][b]
            pred_boxes = outputs["pred_boxes"][b]

            tgt_labels = targets[b]["class_labels"]
            tgt_boxes = targets[b]["boxes"]

            prob = pred_logits.softmax(-1)

            class_cost = -prob[:, tgt_labels]

            bbox_cost = torch.cdist(
                pred_boxes,
                tgt_boxes,
                p=1
            )

            cost_matrix = (
                self.class_cost * class_cost
                +
                self.bbox_cost * bbox_cost
            )

            cost_matrix = cost_matrix.cpu()

            pred_idx, tgt_idx = linear_sum_assignment(
                cost_matrix
            )

            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.long, device=pred_logits.device),
                    torch.as_tensor(tgt_idx, dtype=torch.long, device=pred_logits.device)
                )
            )

        return indices