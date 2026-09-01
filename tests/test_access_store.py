"""
Tests for core/access_store.py - the invite-code gate for the public
demo frontend. Uses a fresh AccessStore instance per test rather than
the module-level singleton, so tests don't leak state into each other.
"""
from core.access_store import AccessStore


def test_generated_code_validates_and_is_not_pre_used():
    store = AccessStore()
    invite = store.generate(label="test client")
    fetched = store.validate(invite.code)
    assert fetched is not None
    assert fetched.code == invite.code
    assert fetched.is_used is False
    print("PASS: generated code validates and starts unused")


def test_redeem_marks_code_used_and_blocks_reuse():
    store = AccessStore()
    invite = store.generate()
    redeemed = store.redeem(invite.code, job_id="job-123")
    assert redeemed is True
    assert store.validate(invite.code) is None  # can't validate a used code
    second_attempt = store.redeem(invite.code, job_id="job-456")
    assert second_attempt is False  # can't redeem twice
    print("PASS: redeeming a code blocks reuse")


def test_unknown_code_does_not_validate_or_redeem():
    store = AccessStore()
    assert store.validate("NOPE-NOPE-NOPE") is None
    assert store.redeem("NOPE-NOPE-NOPE", job_id="job-789") is False
    print("PASS: unknown code is rejected by both validate and redeem")


def test_code_lookup_is_case_and_whitespace_insensitive():
    store = AccessStore()
    invite = store.generate()
    lowered = invite.code.lower() + "  "
    assert store.validate(lowered) is not None
    print("PASS: code lookup normalizes case/whitespace")


if __name__ == "__main__":
    test_generated_code_validates_and_is_not_pre_used()
    test_redeem_marks_code_used_and_blocks_reuse()
    test_unknown_code_does_not_validate_or_redeem()
    test_code_lookup_is_case_and_whitespace_insensitive()
