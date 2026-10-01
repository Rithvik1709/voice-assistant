from voice_assistant.doctor import DoctorCheck, format_doctor_report, has_failures


def test_doctor_report_marks_failures() -> None:
    checks = [
        DoctorCheck("one", True, "ready"),
        DoctorCheck("two", False, "missing"),
    ]

    report = format_doctor_report(checks)

    assert "[OK] one: ready" in report
    assert "[FAIL] two: missing" in report
    assert has_failures(checks)


def test_doctor_report_all_ok() -> None:
    checks = [DoctorCheck("one", True, "ready")]

    assert not has_failures(checks)


def test_new_feature_checks(monkeypatch, tmp_path) -> None:
    import voice_assistant.audio.aec as aec
    from voice_assistant.config import Settings
    from voice_assistant.doctor import _echo_cancellation_check, _user_facts_check, _wake_word_checks

    monkeypatch.setattr(aec, "echo_cancellation_available", lambda: False)
    monkeypatch.setenv("ECHO_CANCELLATION", "on")
    assert not _echo_cancellation_check(Settings()).ok
    monkeypatch.setenv("ECHO_CANCELLATION", "auto")
    assert _echo_cancellation_check(Settings()).ok

    monkeypatch.setenv("WAKE_WORD", "hey vaani")
    monkeypatch.setenv("WAKE_WORD_MODEL", str(tmp_path / "missing.onnx"))
    checks = {c.name: c for c in _wake_word_checks(Settings())}
    assert checks["wake_word"].detail.startswith('"hey vaani"')
    assert not checks["wake_word_model"].ok

    monkeypatch.setenv("USER_FACTS_PATH", str(tmp_path / "facts.json"))
    assert "0 remembered" in _user_facts_check(Settings()).detail
