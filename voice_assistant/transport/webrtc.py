from __future__ import annotations

import asyncio
from dataclasses import dataclass

try:
    from aiortc import MediaStreamTrack, RTCPeerConnection
except Exception:  # pragma: no cover - optional runtime import
    MediaStreamTrack = object  # type: ignore[assignment]
    RTCPeerConnection = object  # type: ignore[assignment]


from opentelemetry import context as otel_context, trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


@dataclass(slots=True)
class WebRTCSession:
    pc: RTCPeerConnection
    incoming_audio: asyncio.Queue[bytes]
    outgoing_audio: asyncio.Queue[bytes]


async def create_webrtc_session(traceparent: str | None = None) -> WebRTCSession:
    if RTCPeerConnection is object:
        raise RuntimeError("aiortc is required for WebRTC mode")
        
    token = None
    if traceparent:
        carrier = {"traceparent": traceparent}
        extracted = TraceContextTextMapPropagator().extract(carrier=carrier)
        token = otel_context.attach(extracted)

    try:
        return WebRTCSession(
            pc=RTCPeerConnection(), 
            incoming_audio=asyncio.Queue(), 
            outgoing_audio=asyncio.Queue()
        )
    finally:
        if token:
            otel_context.detach(token)
