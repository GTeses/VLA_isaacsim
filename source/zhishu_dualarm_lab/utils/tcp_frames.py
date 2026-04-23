from __future__ import annotations

from isaaclab.sensors import FrameTransformerCfg, OffsetCfg


def make_tcp_frame(
    *,
    prim_path: str,
    name: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> FrameTransformerCfg.FrameCfg:
    """Create a virtual TCP frame from a link plus configurable offset."""
    return FrameTransformerCfg.FrameCfg(
        prim_path=prim_path,
        name=name,
        offset=OffsetCfg(pos=pos, rot=rot),
    )

