from ctx_weft.core.models.config import RuntimeConfig


def test_defaults_match_legacy():
    c = RuntimeConfig()
    assert c.hitl_timeout_sec is None
    assert c.hitl_max_resolved == 1000
    assert c.task_max_concurrent == 4
    assert c.task_max_retries == 3
    assert c.default_token_budget == 200_000


def test_overrides():
    c = RuntimeConfig(task_max_concurrent=8, hitl_timeout_sec=30)
    assert c.task_max_concurrent == 8
    assert c.hitl_timeout_sec == 30
