INTENTS = {
    "greeting": {
        "keywords": [
            "hello",
            "hi",
            "hey",
            "namaste",
            "namaskar",
        ],
    },
}


def handle(intent: str, text: str, entities: dict | None = None) -> dict:
    return {"response": "Hello! How can I help you?", "intent": intent}
