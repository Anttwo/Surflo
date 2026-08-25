"""Flow-matching training loss.

Consumes the dict returned by :meth:`surflo.model.ffm.FFM.forward` and produces
a scalar ``objective`` (an L2 loss on either the predicted target points or the
predicted velocity) plus a handful of scalar diagnostics for logging.

Only :class:`FlowLoss` is provided: mean-flow (iMF) training is not part of the
released method.
"""
from dataclasses import dataclass

import torch


def l2_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.pow(pred - gt, 2).mean()


@dataclass(eq=False)
class FlowLoss(torch.nn.Module):
    """L2 flow-matching loss for training flow-based models."""

    def __init__(
        self,
        estimate_in_freq_enc: bool = True,
        prediction_mode: str = "target",  # "target", "velocity"
        compute_loss_in_velocity: bool = False,  # loss between predicted/target velocities when prediction_mode == "target"
        **kwargs,
    ):
        super().__init__()
        self.estimate_in_freq_enc = estimate_in_freq_enc
        assert prediction_mode in ["target", "velocity"], f"Invalid prediction mode: {prediction_mode}"
        self.prediction_mode = prediction_mode
        self.compute_loss_in_velocity = compute_loss_in_velocity
        assert not (self.compute_loss_in_velocity and prediction_mode != "target"), \
            "compute_loss_in_velocity can only be True when prediction_mode is 'target'"

        self.loss_fn = l2_loss

    def forward(self, predictions, batch, **kwargs) -> dict:
        """Compute the flow-matching objective.

        Args:
            predictions: dict returned by ``FFM.forward``.
            batch: (unused here) training batch dict; kept for signature parity.

        Returns:
            dict with per-term diagnostics and ``objective`` (the scalar to
            optimize).
        """
        total_loss = 0
        loss_dict = {}

        # Get predictions and ground truth
        if self.prediction_mode == "target" and not self.compute_loss_in_velocity:
            assert predictions["surface_estimates"] is not None, "Surface estimates are required for target prediction mode"
            # We use predictions["target_points"] (already normalized/lifted), not
            # batch["target_points"] (raw, un-lifted).
            pred = predictions["surface_estimates"]  # (B, P, D)
            gt = predictions["target_points"]  # (B, P, D)
        elif self.prediction_mode == "velocity" or (self.prediction_mode == "target" and self.compute_loss_in_velocity):
            assert predictions["velocity_estimates"] is not None, "Velocity estimates are required for velocity prediction mode"
            assert predictions["conditional_velocity"] is not None, "Conditional velocity is required for velocity prediction mode"
            pred = predictions["velocity_estimates"]  # (B, P, D)
            gt = predictions["conditional_velocity"]  # (B, P, D)
        else:
            raise ValueError(f"Invalid prediction mode: {self.prediction_mode}")

        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"

        # Log metrics -- keep as tensors to avoid GPU->CPU sync stalls. The
        # trainer calls .item() only at log frequency, not every step.
        with torch.no_grad():
            loss_dict["surface_points_mean"] = predictions["target_points"].mean()
            loss_dict["surface_points_std"] = predictions["target_points"].std()
            loss_dict["initial_points_mean"] = predictions["source_points"].mean()
            loss_dict["initial_points_std"] = predictions["source_points"].std()
            if self.prediction_mode == "target":
                loss_dict["predicted_surface_points_mean"] = predictions["surface_estimates"].mean()
                loss_dict["predicted_surface_points_std"] = predictions["surface_estimates"].std()
                loss_dict["predicted_velocity_mean"] = 0.
                loss_dict["predicted_velocity_std"] = 0.
                loss_dict["conditional_velocity_mean"] = 0.
                loss_dict["conditional_velocity_std"] = 0.
            else:
                loss_dict["predicted_surface_points_mean"] = 0.
                loss_dict["predicted_surface_points_std"] = 0.
                loss_dict["predicted_velocity_mean"] = predictions["velocity_estimates"].mean()
                loss_dict["predicted_velocity_std"] = predictions["velocity_estimates"].std()
                loss_dict["conditional_velocity_mean"] = predictions["conditional_velocity"].mean()
                loss_dict["conditional_velocity_std"] = predictions["conditional_velocity"].std()

        # Compute loss
        if not self.estimate_in_freq_enc:
            # Estimated directly in xyz (or r6) space.
            cm_loss = self.loss_fn(pred, gt)
            total_loss += cm_loss

            loss_dict["loss_cm"] = cm_loss.detach()
            loss_dict["loss_cm_sqrt"] = torch.sqrt(cm_loss.detach())
        else:
            # Estimated in frequency-encoded space; also report the xyz-only loss.
            cm_loss = self.loss_fn(pred, gt)
            total_loss += cm_loss

            loss_dict["loss_cm_freqenc"] = cm_loss.detach()
            loss_dict["loss_cm_freqenc_sqrt"] = torch.sqrt(cm_loss.detach())

            with torch.no_grad():
                pred_xyz = pred[..., :3]
                gt_xyz = gt[..., :3]
                cm_loss_xyz = self.loss_fn(pred_xyz, gt_xyz)
                loss_dict["loss_cm"] = cm_loss_xyz
                loss_dict["loss_cm_sqrt"] = torch.sqrt(cm_loss_xyz)

        loss_dict["objective"] = total_loss
        return loss_dict
