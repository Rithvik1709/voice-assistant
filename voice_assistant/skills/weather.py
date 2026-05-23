INTENTS = {
    "weather": {
        "keywords": [
            "weather",
            "kaa mausam",
            "mausam",
            "mosam",
            "weather batao",
            "mausam batao",
        ],
    },
}


def handle(intent: str, text: str, entities: dict | None = None) -> dict:
    return {"response": f"You asked about the weather: {text}", "intent": intent}
