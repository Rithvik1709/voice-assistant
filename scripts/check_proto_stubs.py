#!/usr/bin/env python3
"""Fail if the committed gRPC stubs do not match voice_assistant.proto.

Compares serialized file descriptors, so the check does not depend on which
grpcio-tools version generated the committed stubs.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2
from grpc_tools import protoc

from voice_assistant.transport import voice_assistant_pb2

PROTO = "voice_assistant/transport/voice_assistant.proto"


def _strip_json_names(fd: descriptor_pb2.FileDescriptorProto) -> None:
    # protoc adds json_name to descriptor sets, Python gencode omits it.
    for message in fd.message_type:
        for field in message.field:
            field.ClearField("json_name")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "descriptor.pb"
        if protoc.main(["protoc", "-I.", f"--descriptor_set_out={out}", PROTO]) != 0:
            print("protoc failed", file=sys.stderr)
            return 1
        fresh = descriptor_pb2.FileDescriptorSet.FromString(out.read_bytes()).file[0]

    committed = descriptor_pb2.FileDescriptorProto.FromString(voice_assistant_pb2.DESCRIPTOR.serialized_pb)
    _strip_json_names(fresh)
    _strip_json_names(committed)
    if fresh != committed:
        print(
            "voice_assistant_pb2.py is out of date with the .proto. Regenerate with:\n"
            "  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. " + PROTO,
            file=sys.stderr,
        )
        return 1
    print("protobuf stubs are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
