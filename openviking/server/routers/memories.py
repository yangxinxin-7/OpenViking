# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory maintenance endpoints (principles consolidation and gating)."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from openviking.core.namespace import user_space_fragment
from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking_cli.utils import get_logger

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])
logger = get_logger(__name__)


class GateReport(BaseModel):
    """Verdict of an external gate run over the pending principles candidate."""

    pending_sha256: str
    selection_id: str = ""
    net_wins: int = 0
    score: Optional[float] = None
    baseline_score: Optional[float] = None
    notes: str = ""


def _space(ctx: RequestContext) -> str:
    return user_space_fragment(ctx) if ctx and ctx.user else "default"


async def _read_principles_file(ctx: RequestContext, filename: str) -> dict:
    from openviking.session.memory.consolidation import principles_dir
    from openviking.storage.viking_fs import get_viking_fs

    uri = f"{principles_dir(_space(ctx))}/{filename}"
    try:
        content = (await get_viking_fs().read(uri, ctx=ctx)).decode("utf-8")
    except Exception:
        content = ""
    return {"uri": uri, "content": content}


@router.post("/principles/consolidate")
async def consolidate_principles(
    _ctx: RequestContext = Depends(get_request_context),
):
    """Run one proposal pass over the accumulated trajectory memories.

    Produces a pending candidate (``pending.md``) for external gating; the
    injected ``default.md`` only changes through ``/principles/promote``.
    """
    from openviking.session.memory.consolidation import PrinciplesConsolidator
    from openviking_cli.utils.config import get_openviking_config

    config = get_openviking_config()
    consolidator = PrinciplesConsolidator(config.vlm.get_vlm_instance())
    try:
        result = await consolidator.propose(user_space=_space(_ctx), ctx=_ctx)
    except Exception as exc:
        logger.exception("principles proposal failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(status="ok", result=result)


@router.get("/principles")
async def get_principles(
    _ctx: RequestContext = Depends(get_request_context),
):
    """Return the current (gate-verified) principles document."""
    from openviking.session.memory.consolidation import PRINCIPLES_FILENAME

    return Response(status="ok", result=await _read_principles_file(_ctx, PRINCIPLES_FILENAME))


@router.get("/principles/pending")
async def get_pending_principles(
    _ctx: RequestContext = Depends(get_request_context),
):
    """Return the pending candidate awaiting its gate verdict ("" when none)."""
    from openviking.session.memory.consolidation import PENDING_FILENAME

    return Response(status="ok", result=await _read_principles_file(_ctx, PENDING_FILENAME))


@router.post("/principles/promote")
async def promote_pending_principles(
    report: GateReport,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Promote the pending candidate to default.md after a passing gate run."""
    from openviking.session.memory.consolidation import promote_principles

    result = await promote_principles(user_space=_space(_ctx), report=report.model_dump(), ctx=_ctx)
    if result.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=result.get("reason"))
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("reason"))
    return Response(status="ok", result=result)


@router.post("/principles/reject")
async def reject_pending_principles(
    report: GateReport,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Discard the pending candidate after a failing gate run."""
    from openviking.session.memory.consolidation import reject_principles

    result = await reject_principles(user_space=_space(_ctx), report=report.model_dump(), ctx=_ctx)
    if result.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=result.get("reason"))
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("reason"))
    return Response(status="ok", result=result)
