from scripts.bench.load_asr import ClientResult, quantiles


def test_client_result_includes_ack_and_content_timestamps() -> None:
    result = ClientResult()
    result.first_response_ts = 1.0
    result.first_content_ts = 1.5
    result.ack_responses = 1

    data = result.to_dict()

    assert data["first_response_ts"] == 1.0
    assert data["first_content_ts"] == 1.5
    assert data["ack_responses"] == 1


def test_quantiles_empty_values() -> None:
    assert quantiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
