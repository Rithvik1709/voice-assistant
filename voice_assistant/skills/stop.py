INTENTS = {
    "stop": {
        "keywords": [
            "stop",
            "pause",
            "ruk",
            "band karo",
            "music band karo",
        ],
    },
}


def handle(intent: str, text: str, entities: dict | None = None) -> dict:
    return {"response": "Stopping...", "intent": intent}
