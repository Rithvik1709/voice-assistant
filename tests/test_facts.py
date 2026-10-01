from __future__ import annotations

import json

import pytest

from tests.test_orchestrator import FakeLLM, make, run_with, wait_for
from voice_assistant.facts import UserFacts


@pytest.fixture
def facts(tmp_path) -> UserFacts:
    return UserFacts(tmp_path / "facts.json")


@pytest.mark.parametrize(
    ("said", "key", "value"),
    [
        ("My name is Asha.", "name", "Asha"),
        ("you can call me Ash", "name", "Ash"),
        ("mera naam Rohit hai", "name", "Rohit"),
        ("I live in Pune and I work from home", "location", "Pune"),
        ("I'm from Chennai originally", "hometown", "Chennai originally"),
        ("I am 29 years old", "age", "29"),
        ("I work as a data scientist", "job", "data scientist"),
        ("I'm a software engineer", "job", "software engineer"),
        ("I work at Acme Corp", "workplace", "Acme Corp"),
        ("I'm vegetarian", "diet", "vegetarian"),
        ("my favourite colour is green", "favourite colour", "green"),
        ("My wife's name is Priya", "wife's name", "Priya"),
        ("my birthday is on March 3rd", "birthday", "on March 3rd"),
        ("my city is Delhi", "location", "Delhi"),
    ],
)
def test_facts_are_learned_from_statements(facts: UserFacts, said: str, key: str, value: str) -> None:
    assert facts.observe(said)
    assert facts.facts[key] == value


@pytest.mark.parametrize(
    "said",
    [
        "what is my name?",
        "where do I live",
        "my phone is broken",
        "my question is how rainbows form",
        "I like that",
        "I love you",
        "I'd like to know the weather",
        "what's the time",
    ],
)
def test_questions_and_non_facts_are_ignored(facts: UserFacts, said: str) -> None:
    assert not facts.observe(said)
    assert facts.empty


def test_likes_dislikes_and_allergies(facts: UserFacts) -> None:
    facts.observe("I love cricket. I don't like mushrooms. I'm allergic to peanuts.")

    assert facts.likes == ["cricket"]
    assert facts.dislikes == ["mushrooms"]
    assert facts.allergies == ["peanuts"]

    # Changing your mind moves it across.
    facts.observe("actually I like mushrooms now")
    assert facts.dislikes == []
    assert facts.likes == ["cricket", "mushrooms"]


def test_facts_persist_across_sessions(tmp_path) -> None:
    path = tmp_path / "facts.json"
    UserFacts(path).observe("My name is Asha and I live in Pune")

    again = UserFacts(path)

    assert again.facts == {"name": "Asha", "location": "Pune"}
    assert json.loads(path.read_text())["facts"]["name"] == "Asha"


def test_corrupt_file_starts_empty(tmp_path) -> None:
    path = tmp_path / "facts.json"
    path.write_text("{not json")
    assert UserFacts(path).empty


def test_newer_statement_replaces_older_fact(facts: UserFacts) -> None:
    facts.observe("I live in Pune")
    facts.observe("I live in Bangalore")
    assert facts.facts["location"] == "Bangalore"


def test_remember_command_stores_notes_and_facts(facts: UserFacts) -> None:
    assert facts.handle_command("Remember that I parked on level 2.") == "Okay, I'll remember that."
    assert facts.handle_command("remember that my name is Asha") == "Okay, I'll remember that."
    facts.handle_command("Remember that my sister's birthday is March 3")

    # Anything that is not a recognizable fact is kept verbatim.
    assert facts.notes == ["I parked on level 2"]
    assert facts.facts["name"] == "Asha"
    assert facts.facts["sister's birthday"] == "March 3"


def test_recall_command(facts: UserFacts) -> None:
    assert facts.handle_command("What do you know about me?") == "I don't know anything about you yet."

    facts.observe("My name is Asha. I live in Pune. I love cricket.")
    reply = facts.handle_command("what do you remember about me")

    assert reply == "Here's what I remember: your name is Asha; you live in Pune; you like cricket."


def test_forget_one_fact(facts: UserFacts) -> None:
    facts.observe("My name is Asha. I love cricket. I love chess.")

    assert facts.handle_command("forget that I love cricket") == "Okay, I've forgotten that."
    assert facts.likes == ["chess"]
    assert facts.handle_command("forget my name") == "Okay, I've forgotten that."
    assert "name" not in facts.facts
    assert facts.handle_command("forget my shoe size") == "I didn't have that remembered."


def test_forget_everything(tmp_path) -> None:
    facts = UserFacts(tmp_path / "facts.json")
    facts.observe("My name is Asha. I love cricket.")
    facts.handle_command("remember that I park on level 2")

    assert facts.handle_command("Forget everything about me.") == "Okay, I've forgotten everything I knew about you."
    assert facts.empty
    assert UserFacts(tmp_path / "facts.json").empty


def test_casual_forget_it_is_not_a_command(facts: UserFacts) -> None:
    assert facts.handle_command("forget it") is None
    assert facts.handle_command("tell me a joke") is None


def test_prompt_section(facts: UserFacts) -> None:
    assert facts.prompt_section() == ""

    facts.observe("My name is Asha. My favourite food is dosa. I hate traffic.")
    facts.handle_command("remember that I park on level 2")
    section = facts.prompt_section()

    assert "Their name is Asha." in section
    assert "Their favourite food is dosa." in section
    assert "They dislike traffic." in section
    assert 'They asked you to remember: "I park on level 2"' in section


async def test_orchestrator_adds_learned_facts_to_the_system_prompt(tmp_path) -> None:
    llm = FakeLLM(["Nice", " to", " meet", " you", "."])
    orch, asr, _, _ = make(llm, facts=UserFacts(tmp_path / "facts.json"))

    async def body():
        asr.say("Hi, my name is Asha")
        await wait_for(lambda: len(llm.calls) == 1 and not orch.is_responding)

    await run_with(orch, body)

    system = llm.calls[0][0]
    assert system["role"] == "system"
    assert system["content"].startswith("Be brief.")
    assert "Their name is Asha." in system["content"]


async def test_facts_from_an_earlier_session_are_in_the_first_prompt(tmp_path) -> None:
    UserFacts(tmp_path / "facts.json").observe("I live in Pune")
    llm = FakeLLM(["Sunny", "."])
    orch, asr, _, _ = make(llm, facts=UserFacts(tmp_path / "facts.json"))

    async def body():
        asr.say("how's the weather looking")
        await wait_for(lambda: len(llm.calls) == 1 and not orch.is_responding)

    await run_with(orch, body)
    assert "They live in Pune." in llm.calls[0][0]["content"]


async def test_memory_commands_answer_without_the_llm(tmp_path) -> None:
    llm = FakeLLM(["unused"])
    orch, asr, tts, _ = make(llm, facts=UserFacts(tmp_path / "facts.json"))

    async def body():
        asr.say("remember that my locker code is 4512")
        await wait_for(lambda: len(orch.conversation_history) == 3 and not orch.is_responding)
        asr.say("what do you know about me?")
        await wait_for(lambda: len(orch.conversation_history) == 5 and not orch.is_responding)

    await run_with(orch, body)

    assert llm.calls == []
    assert orch.conversation_history[2]["content"] == "Okay, I'll remember that."
    assert "4512" in orch.conversation_history[4]["content"]
    assert "4512" in orch.conversation_history[0]["content"]
