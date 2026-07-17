"""
Unit tests for the pure-logic parts of bot.py (no Telegram or network calls).
Run with:  pytest test_bot.py
"""

import numpy as np
import pandas as pd
import pytest

from bot import _count_stats, compute_signal, _validate_pair


def test_count_stats_basic():
    stats = _count_stats("The quick brown fox jumps over the lazy dog.")
    assert stats["words"] == 9
    assert stats["sentences"] == 1
    assert stats["chars_no_spaces"] > 0


def test_count_stats_multiple_sentences():
    stats = _count_stats("Hello world! How are you? I am fine.")
    assert stats["sentences"] == 3
    assert stats["words"] == 8


def test_count_stats_top_words():
    stats = _count_stats("apple apple banana apple banana cherry")
    top = dict(stats["top_words"])
    assert top["apple"] == 3
    assert top["banana"] == 2


def test_count_stats_empty_string():
    stats = _count_stats("")
    assert stats["words"] == 0
    assert stats["sentences"] == 0
    assert stats["reading_time_seconds"] == 0


def test_validate_pair_valid():
    assert _validate_pair("EURUSD") == ("EUR", "USD")


def test_validate_pair_wrong_length():
    with pytest.raises(ValueError):
        _validate_pair("EUR")


def test_validate_pair_unknown_code():
    with pytest.raises(ValueError):
        _validate_pair("XXXYYY")


def _make_uptrend_df(n=60):
    prices = np.linspace(1.0, 1.1, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices}, index=idx)


def _make_downtrend_df(n=60):
    prices = np.linspace(1.1, 1.0, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices}, index=idx)


def test_compute_signal_uptrend_is_bullish_or_neutral():
    result = compute_signal(_make_uptrend_df())
    assert result["overall"] in ("BUY", "NEUTRAL")
    assert 0 <= result["confidence"] <= 100


def test_compute_signal_downtrend_is_bearish_or_neutral():
    result = compute_signal(_make_downtrend_df())
    assert result["overall"] in ("SELL", "NEUTRAL")


def test_compute_signal_insufficient_data_raises():
    short_df = _make_uptrend_df(n=10)
    with pytest.raises(RuntimeError):
        compute_signal(short_df)
