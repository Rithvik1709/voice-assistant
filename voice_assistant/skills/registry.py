from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

SKILLS_PACKAGE = __package__

_INTENT_KEYWORDS: dict[str, list[str]] = {}
_HANDLERS: dict[str, Callable[..., dict]] = {}
_SKILL_NAMES: dict[str, str] = {}


@dataclass
class SkillInfo:
    name: str
    intents: dict[str, dict[str, Any]]
    handle: Callable[..., Any] | None = None


def _discover() -> None:
    _INTENT_KEYWORDS.clear()
    _HANDLERS.clear()
    _SKILL_NAMES.clear()

    try:
        package = importlib.import_module(SKILLS_PACKAGE)
    except Exception:
        logger.exception("Failed to import skills package")
        return

    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if modname == "registry" or modname.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{SKILLS_PACKAGE}.{modname}")
            _register_module(modname, module)
        except Exception:
            logger.exception(f"Failed to load skill module: {modname}")


def _register_module(modname: str, module: object) -> None:
    intents = getattr(module, "INTENTS", None)
    if not intents or not isinstance(intents, dict):
        logger.warning(f"Skill '{modname}' has no INTENTS dict, skipping")
        return

    handle_fn = getattr(module, "handle", None)
    for intent_name, config in intents.items():
        if intent_name in _HANDLERS:
            logger.warning(
                f"Duplicate intent '{intent_name}' in skill '{modname}' "
                f"(already registered by '{_SKILL_NAMES[intent_name]}')"
            )
        _SKILL_NAMES[intent_name] = modname
        keywords = config.get("keywords", [])
        if keywords:
            if intent_name in _INTENT_KEYWORDS:
                existing = _INTENT_KEYWORDS[intent_name]
                for kw in keywords:
                    if kw not in existing:
                        existing.append(kw)
            else:
                _INTENT_KEYWORDS[intent_name] = keywords
        if handle_fn:
            _HANDLERS[intent_name] = handle_fn

    logger.info(
        "Registered skill: %s with intents: %s",
        modname,
        list(intents.keys()),
    )


def get_intent_keywords() -> dict[str, list[str]]:
    if not _INTENT_KEYWORDS:
        _discover()
    return dict(_INTENT_KEYWORDS)


def dispatch(intent: str, text: str, entities: dict | None = None) -> dict:
    if not _HANDLERS:
        _discover()
    handler = _HANDLERS.get(intent)
    if handler:
        try:
            result = handler(intent, text, entities or {})
            if isinstance(result, str):
                return {"response": result, "intent": intent}
            return result
        except Exception:
            logger.exception(f"Skill handler failed for intent '{intent}'")
    return {"response": None, "intent": intent}


def get_skill_name(intent: str) -> str | None:
    return _SKILL_NAMES.get(intent)


def reset() -> None:
    _INTENT_KEYWORDS.clear()
    _HANDLERS.clear()
    _SKILL_NAMES.clear()
    _discover()


class SkillRegistry:
    def __init__(self) -> None:
        _discover()

    def get_all_keywords(self) -> dict[str, list[str]]:
        return get_intent_keywords()

    def dispatch(self, intent: str, text: str, entities: dict | None = None) -> dict:
        return dispatch(intent, text, entities)

    def get_skill_for_intent(self, intent: str) -> str | None:
        return get_skill_name(intent)
