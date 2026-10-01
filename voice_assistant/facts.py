"""Long-term memory: facts about the user that last across sessions.

Conversation history is a sliding window, so after a few turns (or a
restart) Vaani forgets who it is talking to. This keeps the durable facts
separately, in a small JSON file, and adds them to the system prompt:

- Statements are picked up as they are said: "my name is Asha", "I live in
  Pune", "I'm vegetarian", "my favourite colour is green", "I love cricket".
- "Remember that ..." stores anything verbatim.
- "What do you know about me?" lists what is stored, "forget that I like
  cricket" removes one fact and "forget everything about me" clears them all.

Extraction is rule-based (English and some Hinglish), so it is predictable,
instant and works offline with any LLM. Everything stays on this machine.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_LIST_ITEMS = 20
MAX_NOTES = 50
# Longer values are probably not a fact ("my point is that we should ...").
MAX_VALUE_WORDS = 6

_QUESTION_WORDS = frozenset(
    "what who whom whose where when why how which do does did is are am was were can could would will shall should may might".split()
)
# Values that say nothing on their own ("I like that", "I love you").
_EMPTY_VALUES = frozenset("it that this you them him her these those so too one".split())
# "my X is Y" is only kept for these kinds of X, so "my phone is broken" or
# "my question is ..." are not stored as facts.
_KEY_WORDS = frozenset(
    """
    name birthday anniversary age city hometown town country job profession occupation
    work company school college university language languages pronouns email
    """.split()
)
_KEY_PREFIXES = ("favourite ", "favorite ")

# Cut a value at the first of these: "I live in Pune and I work ..." -> "Pune".
_VALUE_END = re.compile(
    r"\s*(?:,|;|\.|!|\?|\bbut\b|\band (?:i|my|you|it)\b|\bbecause\b|\bso\b|\bwhich\b|\bthough\b|\bhai\b|\bhoon\b|\bhun\b)",
    re.IGNORECASE,
)


_TRAILING_FILLER = re.compile(
    r"(?:\s+(?:now|too|also|as well|a lot|very much|so much|these days|anymore|nowadays))+$", re.IGNORECASE
)


def _clean_value(raw: str, max_words: int = MAX_VALUE_WORDS) -> str | None:
    value = _VALUE_END.split(raw, maxsplit=1)[0].strip(" \"'")
    value = re.sub(r"^(?:actually|really|currently|now)\s+", "", value, flags=re.IGNORECASE)
    value = _TRAILING_FILLER.sub("", value)
    words = value.split()
    if not words or len(words) > max_words or words[0].lower() in _EMPTY_VALUES:
        return None
    return value


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]


def _is_question(sentence: str) -> bool:
    words = sentence.lower().split()
    return sentence.rstrip().endswith("?") or (bool(words) and words[0] in _QUESTION_WORDS)


# (key, pattern): the pattern's first group is the value.
_KEYED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r"\b(?:call me|you can call me)\s+(.+)", re.I)),
    ("name", re.compile(r"\bmera naam\s+(.+?)\s+(?:hai|h)\b", re.I)),
    ("location", re.compile(r"\bi\s+(?:live|stay|am living|'m living)\s+in\s+(.+)", re.I)),
    ("location", re.compile(r"\bmain\s+(.+?)\s+(?:mein|me)\s+(?:rehta|rehti)\b", re.I)),
    ("hometown", re.compile(r"\bi(?:'m| am)\s+(?:originally\s+)?from\s+(.+)", re.I)),
    ("age", re.compile(r"\bi(?:'m| am)\s+(\d{1,3})\s+(?:years?\s+old|yrs?\s+old)\b", re.I)),
    ("job", re.compile(r"\bi\s+work\s+as\s+(?:an?\s+)?(.+)", re.I)),
    ("job", re.compile(r"\bi(?:'m| am)\s+an?\s+((?:software |data |web )?(?:engineer|developer|doctor|teacher|student|designer|nurse|lawyer|writer|scientist|manager|researcher|artist|chef|accountant)\b.*)", re.I)),
    ("workplace", re.compile(r"\bi\s+work\s+(?:at|for)\s+(.+)", re.I)),
    ("diet", re.compile(r"\bi(?:'m| am)\s+(?:an?\s+)?(vegetarian|vegan|pescatarian|eggetarian|jain)\b", re.I)),
]
_MY_X_IS_Y = re.compile(r"\bmy\s+((?:[\w']+\s+){0,2}[\w']+)\s+is\s+(.+)", re.I)
_LIKES = re.compile(r"\bi\s+(?:really\s+|absolutely\s+)?(?:like|love|enjoy|adore)\s+(.+)", re.I)
_DISLIKES = re.compile(r"\bi\s+(?:really\s+)?(?:hate|dislike|can't stand|cannot stand|don't like|do not like)\s+(.+)", re.I)
_ALLERGIES = re.compile(r"\bi(?:'m| am)\s+allergic\s+to\s+(.+)", re.I)

_REMEMBER = re.compile(r"^(?:please\s+|hey\s+)?(?:remember|note|keep in mind)\s+(?:that\s+)?(.+)", re.I)
_FORGET_ALL = re.compile(
    r"^(?:please\s+)?(?:forget|erase|delete|clear)\s+(?:everything|all)(?:\s+(?:you know|you remember))?(?:\s+about me)?\W*$",
    re.I,
)
_FORGET = re.compile(r"^(?:please\s+)?forget\s+(?:that\s+|about\s+)?(.+)", re.I)
_RECALL = re.compile(
    r"\b(?:what do you (?:know|remember) about me|what have you remembered|what do you remember|"
    r"what you know about me|tell me what you know about me)\b",
    re.I,
)

_KEY_LABELS = {
    "name": ("Their name is {}.", "your name is {}"),
    "location": ("They live in {}.", "you live in {}"),
    "hometown": ("They are from {}.", "you're from {}"),
    "age": ("They are {} years old.", "you're {} years old"),
    "job": ("They work as {}.", "you work as {}"),
    "workplace": ("They work at {}.", "you work at {}"),
    "diet": ("They are {}.", "you're {}"),
}


@dataclass
class UserFacts:
    """Facts about the user, saved as JSON at `path`."""

    path: Path
    facts: dict[str, str] = field(default_factory=dict)
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._load()

    # ----- storage -----------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read user facts from %s; starting empty", self.path)
            return
        self.facts = {str(k): str(v) for k, v in (data.get("facts") or {}).items()}
        for name in ("likes", "dislikes", "allergies", "notes"):
            setattr(self, name, [str(v) for v in data.get(name) or []])

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "facts": self.facts,
            "likes": self.likes,
            "dislikes": self.dislikes,
            "allergies": self.allergies,
            "notes": self.notes,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    @property
    def empty(self) -> bool:
        return not (self.facts or self.likes or self.dislikes or self.allergies or self.notes)

    # ----- learning ----------------------------------------------------

    def observe(self, text: str) -> bool:
        """Store any facts stated in `text`. Returns whether anything changed."""
        with self._lock:
            changed = self._observe(text)
            if changed:
                self._save()
            return changed

    def _observe(self, text: str) -> bool:
        changed = False
        for sentence in _sentences(text):
            if _is_question(sentence):
                continue
            changed |= self._observe_sentence(sentence)
        return changed

    def _observe_sentence(self, sentence: str) -> bool:
        changed = False
        for key, pattern in _KEYED_PATTERNS:
            match = pattern.search(sentence)
            if match and (value := _clean_value(match.group(1))):
                changed |= self._set(key, value)

        match = _MY_X_IS_Y.search(sentence)
        if match:
            key = match.group(1).lower()
            last = key.split()[-1].removesuffix("'s")
            if (last in _KEY_WORDS or key.startswith(_KEY_PREFIXES)) and (value := _clean_value(match.group(2))):
                changed |= self._set("location" if key in {"city", "home city"} else key, value)

        for pattern, items in ((_ALLERGIES, self.allergies), (_DISLIKES, self.dislikes)):
            match = pattern.search(sentence)
            if match and (value := _clean_value(match.group(1), max_words=8)):
                changed |= self._add(items, value)
        if not _DISLIKES.search(sentence):
            match = _LIKES.search(sentence)
            if match and (value := _clean_value(match.group(1), max_words=8)):
                # Someone who now likes it no longer dislikes it, and vice versa.
                changed |= self._add(self.likes, value, remove_from=self.dislikes)
        return changed

    def _set(self, key: str, value: str) -> bool:
        if self.facts.get(key) == value:
            return False
        self.facts[key] = value
        logger.info("Remembered: %s = %s", key, value)
        return True

    def _add(self, items: list[str], value: str, remove_from: list[str] | None = None) -> bool:
        lowered = value.lower()
        if remove_from is not None:
            remove_from[:] = [v for v in remove_from if v.lower() != lowered]
        if any(v.lower() == lowered for v in items):
            return False
        items.append(value)
        del items[:-MAX_LIST_ITEMS]
        logger.info("Remembered: %s", value)
        return True

    # ----- commands ----------------------------------------------------

    def handle_command(self, text: str) -> str | None:
        """Answer "remember that", "forget that" and "what do you know about me", or None."""
        stripped = text.strip()
        with self._lock:
            if _RECALL.search(stripped):
                return self._recall()
            if _FORGET_ALL.match(stripped):
                self.facts.clear()
                for items in (self.likes, self.dislikes, self.allergies, self.notes):
                    items.clear()
                self._save()
                return "Okay, I've forgotten everything I knew about you."
            match = _FORGET.match(stripped)
            if match and match.group(1).strip(" .!").lower() not in {"it", "this", "that", "about it"}:
                forgot = self._forget(match.group(1))
                if forgot:
                    self._save()
                    return "Okay, I've forgotten that."
                return "I didn't have that remembered."
            match = _REMEMBER.match(stripped)
            if match:
                content = match.group(1).strip().rstrip(".!")
                # "Remember that my name is Asha" is a fact; anything else is kept verbatim.
                if not self._observe(content):
                    if content.lower() not in (n.lower() for n in self.notes):
                        self.notes.append(content)
                        del self.notes[:-MAX_NOTES]
                self._save()
                return "Okay, I'll remember that."
        return None

    def _forget(self, phrase: str) -> bool:
        words = set(re.findall(r"[\w']+", phrase.lower())) - {"i", "my", "me", "that", "the", "a", "an", "about"}
        phrase_l = phrase.lower()
        if not words:
            return False

        def mentioned(item: str) -> bool:
            item_l = item.lower()
            return item_l in phrase_l or set(re.findall(r"[\w']+", item_l)) <= words

        removed = False
        for key in list(self.facts):
            if mentioned(key) or mentioned(self.facts[key]):
                del self.facts[key]
                removed = True
        for items in (self.likes, self.dislikes, self.allergies, self.notes):
            kept = [v for v in items if not mentioned(v)]
            removed |= len(kept) != len(items)
            items[:] = kept
        return removed

    def _recall(self) -> str:
        if self.empty:
            return "I don't know anything about you yet."
        parts = []
        for key, value in self.facts.items():
            parts.append(_KEY_LABELS[key][1].format(value) if key in _KEY_LABELS else f"your {key} is {value}")
        if self.likes:
            parts.append("you like " + _join(self.likes))
        if self.dislikes:
            parts.append("you don't like " + _join(self.dislikes))
        if self.allergies:
            parts.append("you're allergic to " + _join(self.allergies))
        parts.extend(self.notes)
        return "Here's what I remember: " + "; ".join(parts) + "."

    # ----- prompt ------------------------------------------------------

    def prompt_section(self) -> str:
        """The facts as a system-prompt paragraph ("" when there are none)."""
        with self._lock:
            if self.empty:
                return ""
            lines = []
            for key, value in self.facts.items():
                lines.append(_KEY_LABELS[key][0].format(value) if key in _KEY_LABELS else f"Their {key} is {value}.")
            if self.likes:
                lines.append("They like " + _join(self.likes) + ".")
            if self.dislikes:
                lines.append("They dislike " + _join(self.dislikes) + ".")
            if self.allergies:
                lines.append("They are allergic to " + _join(self.allergies) + ".")
            lines.extend(f'They asked you to remember: "{note}"' for note in self.notes)
        return (
            "What you know about the user from earlier conversations (use it when relevant, "
            "and do not list it unprompted):\n- " + "\n- ".join(lines)
        )


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
