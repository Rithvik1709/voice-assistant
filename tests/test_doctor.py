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
