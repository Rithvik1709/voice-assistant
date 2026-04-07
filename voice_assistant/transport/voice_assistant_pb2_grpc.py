from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import grpc

from voice_assistant.transport import voice_assistant_pb2 as pb2


class VoiceAssistantStub:
    def __init__(self, channel: grpc.Channel) -> None:
        self.StreamVoice = channel.stream_stream(
            "/voice_assistant.VoiceAssistant/StreamVoice",
            request_serializer=pb2.AudioChunk.SerializeToString,
            response_deserializer=pb2.AudioResponse.FromString,
        )


class VoiceAssistantServicer:
    def StreamVoice(self, request_iterator, context):  # pragma: no cover - interface stub
        raise NotImplementedError()


def add_VoiceAssistantServicer_to_server(servicer: VoiceAssistantServicer, server: grpc.Server) -> None:
    rpc_method_handlers = {
        "StreamVoice": grpc.stream_stream_rpc_method_handler(
            servicer.StreamVoice,
            request_deserializer=pb2.AudioChunk.FromString,
            response_serializer=pb2.AudioResponse.SerializeToString,
        )
    }
    generic_handler = grpc.method_handlers_generic_handler("voice_assistant.VoiceAssistant", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
