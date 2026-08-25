"""Flow-matching inference driver + public entry points.

Both inference variants share a single Euler loop, :func:`run_guided_flow`,
which drives an opaque *runner* object behind a small fixed interface (``x``,
``ode_steps``, ``flow_velocity``, ``guide``, ``euler_step``, ``polish``,
``build_result``). Two public entry points dispatch to a runner:

  * :func:`guided_inference`  -> :class:`surflo.inference.guided._GuidedRun`
  * :func:`plain_inference`   -> :class:`surflo.inference.plain._PlainRun`

Guided inference reads as a plain Euler solver: sample a source cloud, then per
step query the flow velocity and optionally *bend* it with rendering guidance
before stepping. :func:`plain_inference` uses the same runner interface but is
driven by :func:`run_plain_flow`, which drops the ``guide`` / ``polish`` stages
and adds an outer point-chunking loop (a memory knob). Both return the same
result-dict schema.

The runner classes are imported lazily inside the entry points so that
:mod:`surflo.inference.guided` can import :class:`_OdeStep` from here without
an import cycle.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch


class PhaseTimer:
    """Accumulate CUDA-synchronised wall-clock per named phase.

    Guidance is *interleaved* with the ODE (:func:`run_guided_flow` calls
    ``guide`` inside every Euler step), so a runtime split cannot come from
    timers wrapped around two sequential blocks -- it has to be accumulated per
    call. That is what this does: every ``phase("ode")`` / ``phase("guidance")``
    block adds to a running total keyed by name.

    Disabled by default. A meaningful GPU timing needs a
    ``torch.cuda.synchronize()`` on both sides of every block, and that
    serialisation is not free, so inference only builds an enabled timer when
    the caller passes ``profile=True``.
    """

    def __init__(self, enabled: bool = False, device=None):
        self.enabled = bool(enabled)
        self.device = device
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - t0)
            self.counts[name] = self.counts.get(name, 0) + 1

    def as_dict(self) -> Dict[str, object]:
        """``{"enabled": bool, "seconds": {...}, "calls": {...}}``."""
        return {
            "enabled": self.enabled,
            "seconds": dict(self.totals),
            "calls": dict(self.counts),
        }


@dataclass
class _OdeStep:
    """Mutable per-step scratchpad threaded through the ODE loop."""

    idx: int
    t_curr: torch.Tensor        # clamped time fed to the flow model
    true_t_curr: torch.Tensor   # unclamped time (gating / logging)
    dt: torch.Tensor
    render_loss: Optional[float] = None
    apply_guidance: bool = False


def materialize_deferred_losses(
    values: List[Optional[object]],
) -> List[Optional[float]]:
    """Convert a list that may hold deferred 0-dim loss tensors to Python floats.

    The guided runners keep each render loss as a detached GPU scalar during the
    hot inner loop (avoiding a per-step ``.item()`` device sync) and only realise
    the Python floats here, with a **single** device→host copy, preserving order
    and ``None`` entries (pre-guidance steps). Do not reintroduce a per-step
    ``float(loss)``: the values are the same, but the sync is not free.
    """
    tensor_pos = [i for i, v in enumerate(values) if torch.is_tensor(v)]
    if not tensor_pos:
        return list(values)
    materialized = torch.stack([values[i] for i in tensor_pos]).tolist()  # one sync
    out: List[Optional[float]] = list(values)
    for i, v in zip(tensor_pos, materialized):
        out[i] = float(v)
    return out


def run_guided_flow(run) -> Dict[str, object]:
    """Drive a guided runner through the flow-matching Euler loop.

    ``run`` is any object exposing the runner interface: an initial source
    sample ``run.x`` plus the methods :meth:`ode_steps`, :meth:`flow_velocity`,
    :meth:`guide`, :meth:`euler_step`, :meth:`polish` and
    :meth:`build_result`. Per Euler step it queries the flow velocity and
    optionally bends it with rendering guidance (under the hood) before taking
    the Euler step; an optional aux-only refinement (phase C) then runs before
    the result dict is assembled.
    """
    # Timing lives here rather than inside the runner's methods so the
    # ODE / guidance split is accumulated in one place. Inert unless the
    # runner was built with profile=True (see :class:`PhaseTimer`).
    prof = getattr(run, "profiler", None) or PhaseTimer(enabled=False)
    x = run.x
    for step in run.ode_steps():
        with prof.phase("ode"):
            velocity = run.flow_velocity(x, step)   # plain FM velocity (no grad)
        with prof.phase("guidance"):
            velocity = run.guide(x, velocity, step)  # rendering guidance, under the hood
        with prof.phase("ode"):
            x = run.euler_step(x, velocity, step)   # Euler update + callbacks / logging
    with prof.phase("polish"):
        run.polish(x)                               # optional aux-only refinement (phase C)
    return run.build_result(x)


def run_plain_flow(run) -> Dict[str, object]:
    """Drive a plain (guidance-free) runner through the flow-matching Euler loop.

    The unguided companion to :func:`run_guided_flow`. ``run`` exposes a trimmed
    runner interface: :meth:`chunks` (a memory knob that yields once per
    point-batch, re-seeding ``run.x`` with a fresh source sample) plus the same
    :meth:`ode_steps`, :meth:`flow_velocity` and :meth:`euler_step` stages as the
    guided runners -- only without the ``guide`` / ``polish`` guidance hooks.
    Each chunk is an independent full ODE solve, so per-chunk results are simply
    accumulated (via :meth:`finish_chunk`) and concatenated in
    :meth:`build_result`.
    """
    prof = getattr(run, "profiler", None) or PhaseTimer(enabled=False)
    for _ in run.chunks():
        x = run.x
        for step in run.ode_steps():
            with prof.phase("ode"):
                velocity = run.flow_velocity(x, step)   # plain FM velocity (no grad)
                x = run.euler_step(x, velocity, step)   # Euler update
        with prof.phase("decode"):
            run.finish_chunk(x)                      # unlift + accumulate this chunk
    return run.build_result()


def guided_inference(
    model,
    batch: dict,
    scene_idx: int = 0,
    start_time: float = 0.0,
    # --- FM inference params ---
    # Every default below matches ``configs/guided/default.yaml``, so a bare
    # ``SceneState.reconstruct(mode="guided")`` (no ``config_block``) reproduces
    # ``guided=default``. Callers route presets through
    # ``_filter_kwargs_to_signature``, which drops any key this signature does
    # not accept; the shipped presets contain none.
    num_steps: int = 100,
    use_biphase: bool = True,
    num_steps_phase_1: int = 50,
    num_steps_phase_2: int = 100,
    phase_switch_frac: float = 0.95,
    num_query_points: int = 100_000,
    # --- Rendering guidance for x ---
    learn_normals: bool = True,
    decouple_normals: bool = True,
    initial_n_gaussians_per_anchor: int = 1,
    learn_camera_pose: bool = True,
    cam_rot_lr: float = 1e-5,
    cam_trans_lr: float = 1e-5,
    k_inner_loop: int = 32,
    n_additional_iterations: int = 500,
    # ---- 3DGS densify/prune/opacity-reset ----
    densify_from_step: int = 500,
    densify_until_step: int = 2000,
    densification_interval: int = 250,
    opacity_reset_interval: int = 1000,
    densify_grad_threshold: float = 0.0002,
    densify_min_opacity: float = 0.05,
    densify_percent_dense: float = 0.01,
    densify_split_n: int = 2,
    densify_use_abs_grad: bool = True,
    densify_max_screen_size_after_first_reset: Optional[int] = 20,
    opacity_reset_value: float = 0.01,
    log_densify_diagnostics: bool = False,
    # ---- GW-style position-LR schedule ----
    use_position_lr_schedule: bool = True,
    position_lr_init: float = 0.00016,
    position_lr_final: float = 1.6e-6,
    position_lr_delay_mult: float = 0.01,
    position_lr_delay_steps: int = 0,
    position_lr_max_steps: int = 30_000,
    # ---- GW-style SH-degree warmup ----
    use_sh_degree_warmup: bool = True,
    sh_degree_warmup_interval: int = 500,
    render_guidance_scale: float = 100.0,        # used as on/off gate (>0)
    render_guidance_start_frac: float = 0.95,
    # --- Auxiliary Gaussian losses ---
    use_rgb_loss: bool = True,
    use_dn_loss: bool = True,
    start_dn_loss_at_step: int = 100,
    use_vggt_depth_loss: bool = True,
    use_mask_loss: bool = True,
    use_normal_loss: bool = True,
    start_normal_loss_at_step: int = 100,
    use_anisotropy_penalty: bool = True,
    use_confidence: bool = True,
    confidence_threshold: float = 2.0,
    use_smooth_confidence_mask: bool = True,
    smooth_confidence_mask_min_value: float = 0.01,
    use_exposure_compensation: bool = True,
    use_entropy_loss: bool = False,
    start_entropy_loss_at_step: int = 125,
    use_monodepth_guidance: bool = True,
    start_monodepth_guidance_at_step: int = 0,
    depth_ratio_for_monodepth_guidance: float = 0.6,
    monodepths: Optional[torch.Tensor] = None,
    use_normal_guidance: bool = True,
    start_normal_guidance_at_step: int = 0,
    normal_guidances: Optional[torch.Tensor] = None,
    depth_ratio_for_normal_guidance: float = 0.6,
    use_curvature_loss_for_normal_guidance: bool = True,
    lambda_rgb: float = 1.0,
    lambda_depth: float = 5.0,
    lambda_mask_loss: float = 1.0,
    lambda_normal_loss: float = 0.05,
    lambda_anisotropy_penalty: float = 0.1,
    lambda_entropy_loss: float = 0.1,
    lambda_monodepth_guidance: float = 1.0,
    lambda_normal_guidance: float = 0.1,
    lambda_curvature_loss: float = 0.025,
    use_random_bg: bool = True,
    mask_bg: bool = True,
    initial_opacity: float = 0.1,
    initial_scale_factor: float = 1.0,
    sh_degree: int = 3,
    aux_lr_multiplier: float = 1.0,
    gaussian_radius: float = 0.003,
    initialize_isotropic_gaussians: bool = True,
    k_neighbors: int = 10,
    camera_indices: Optional[List[int]] = None,
    # Optional high-res rendering supervision: pass `(N, 3, H, W)` images larger
    # than the VGGT inputs and the guidance loop renders/optimises against them,
    # resizing the cameras and upsampling the depth / confidence / cull tensors
    # to match. `None` (the default, and what every shipped preset uses) keeps
    # everything at VGGT resolution.
    guidance_images: Optional[torch.Tensor] = None,
    target_camera_extent: Optional[float] = 5.0,
    step_callback: Optional[Callable[..., None]] = None,
    lambda_depth_schedule_values: Optional[List[float]] = (0.5,),
    lambda_depth_schedule_steps: Optional[List[int]] = (125,),
    lambda_monodepth_schedule_values: Optional[List[float]] = (0.5,),
    lambda_monodepth_schedule_steps: Optional[List[int]] = (140,),
    lambda_normal_guidance_schedule_values: Optional[List[float]] = (0.05,),
    lambda_normal_guidance_schedule_steps: Optional[List[int]] = (140,),
    lambda_curvature_loss_schedule_values: Optional[List[float]] = (0.0125,),
    lambda_curvature_loss_schedule_steps: Optional[List[int]] = (140,),
    save_intermediates: bool = False,
    # Accumulate per-phase wall-clock ("ode" / "guidance" / "polish") into the
    # result dict under "timings". Off by default: a correct GPU timing needs a
    # CUDA sync on both sides of every phase. See :class:`PhaseTimer`.
    profile: bool = False,
):
    """Flow-matching inference with rendering-based guidance + 3DGS-style
    densification / pruning / opacity reset.

    All parameter defaults mirror ``configs/guided/default.yaml`` (the shipped
    ``guided=default`` preset), so calling this with only ``model`` / ``batch``
    reproduces that preset. Because ``default`` enables the monodepth / normal
    priors, a bare guided run still needs ``monodepths`` / ``normal_guidances``
    (the API injects them from ``expert_cfg``), exactly as if you passed
    ``guided=default`` on the CLI.
    """
    params = dict(locals())
    from .guided import _GuidedRun

    return run_guided_flow(_GuidedRun(params))


@torch.no_grad()
def plain_inference(
    model,
    batch: dict,
    scene_idx: int = 0,
    start_time: float = 0.0,
    # --- FM inference params ---
    num_steps: int = 100,
    num_query_points: int = 100_000,
    num_points_per_batch: int = 100_000,
    seed: int = 42,
    return_source: bool = False,
    save_intermediates: bool = False,
    # See :func:`guided_inference`; the phases here are "source" / "ode" /
    # "decode" (there is no guidance stage).
    profile: bool = False,
):
    """Plain (guidance-free) flow-matching inference, runner-style.

    Solves the same flow-matching Euler ODE as
    :func:`surflo.inference.plain.run_plain_inference` -- sample the source
    cloud, integrate the flow, decode ``r6 -> points (+ normals)`` -- but exposes
    the guided-style runner/driver structure for uniformity with
    :func:`guided_inference`. There is no rendering guidance.

    Reproducible: the per-chunk source sampling draws from a single seeded
    ``torch.Generator`` (a fixed ``seed`` + ``num_points_per_batch`` yields
    identical output). Memory-bounded: points are flowed in chunks of
    ``num_points_per_batch``, each an independent full ODE solve.

    By default the returned dict has exactly the same keys as
    :func:`guided_inference`; the ``aux_*`` / ``supervision_images``
    fields are empty / ``None`` and ``render_losses`` is ``[]`` since no
    guidance runs.
    ``intermediates`` (gated by ``save_intermediates``) holds the flow-space
    trajectory ``(num_steps + 1, num_query_points, D)`` with ``[0]`` the source
    sample -- matching the guided variants (note: flow space, not world space).

    ``return_source`` (opt-in, off by default so the default schema stays uniform
    with the guided variants) additionally returns the *world-space* source cloud
    as ``source_points`` / ``source_normals`` -- cheap (one extra decoded cloud,
    not the whole trajectory) and equivalent to
    :func:`run_plain_inference`'s ``source_points`` (used for ``initial.ply``).
    """
    params = dict(locals())
    from .plain import _PlainRun

    return run_plain_flow(_PlainRun(params))
