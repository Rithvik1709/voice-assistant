INTENTS = {
    "play_music": {
        "keywords": [
            "play",
            "play music",
            "song",
            "gaana",
            "play the song",
            "gaana chala do",
            "music chala do",
            "song baja do",
        ],
    },
}


def handle(intent: str, text: str, entities: dict | None = None) -> dict:
    return {"response": "Playing music...", "intent": intent}
