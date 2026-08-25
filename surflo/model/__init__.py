"""Surflo model assembly.

Exports are lazy (PEP 562) so that importing a lightweight submodule (e.g.
``surflo.model.surface_net``) does not eagerly pull the full inference stack
(flow_matching / torch_geometric / the VGGT-1B backbone) that ``FFM`` needs.
"""
from typing import TYPE_CHECKING

__all__ = ["FFM", "SurfaceNet", "VelocityModel"]

if TYPE_CHECKING:
    from surflo.model.ffm import FFM
    from surflo.model.surface_net import SurfaceNet
    from surflo.model.flow import VelocityModel


def __getattr__(name):
    if name == "FFM":
        from surflo.model.ffm import FFM
        return FFM
    if name == "SurfaceNet":
        from surflo.model.surface_net import SurfaceNet
        return SurfaceNet
    if name == "VelocityModel":
        from surflo.model.flow import VelocityModel
        return VelocityModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
