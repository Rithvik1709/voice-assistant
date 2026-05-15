from __future__ import annotations

import asyncio
from dataclasses import dataclass

try:
    from aiortc import MediaStreamTrack, RTCPeerConnection
except Exception:  # pragma: no cover - optional runtime import
    MediaStreamTrack = object  # type: ignore[assignment]
    RTCPeerConnection = object  # type: ignore[assignment]


from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


@dataclass(slots=True)
class WebRTCSession:
    pc: RTCPeerConnection
    incoming_audio: asyncio.Queue[bytes]
    outgoing_audio: asyncio.Queue[bytes]


async def create_webrtc_session(traceparent: str | None = None) -> WebRTCSession:
    if RTCPeerConnection is object:
        raise RuntimeError("aiortc is required for WebRTC mode")
        
    if traceparent:
        carrier = {"traceparent": traceparent}
        extracted = TraceContextTextMapPropagator().extract(carrier=carrier)
        import opentelemetry.context as otel_context
        otel_context.attach(extracted)

    return WebRTCSession(
        pc=RTCPeerConnection(), 
        incoming_audio=asyncio.Queue(), 
        outgoing_audio=asyncio.Queue()
    )
