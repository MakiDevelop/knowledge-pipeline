"""Tests for score.py — routing logic and signal score computation."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from score import (
    compute_route,
    compute_signal_score,
    validate_scores,
    DEFAULT_SCORES,
    VALID_TIME_HORIZONS,
)


class TestComputeRoute:
    def test_high_risk_goes_to_validator(self):
        scores = {"risk_level": 4, "evidence_strength": 3, "emotional_noise": 1,
                  "novelty": 2, "knowledge_density": 2, "actionability": 2}
        assert compute_route(scores) == "validator"

    def test_low_evidence_high_emotion_goes_to_validator(self):
        scores = {"risk_level": 1, "evidence_strength": 1, "emotional_noise": 4,
                  "novelty": 2, "knowledge_density": 2, "actionability": 2}
        assert compute_route(scores) == "validator"

    def test_high_novelty_goes_to_research(self):
        scores = {"risk_level": 1, "evidence_strength": 3, "emotional_noise": 1,
                  "novelty": 5, "knowledge_density": 3, "actionability": 2}
        assert compute_route(scores) == "research"

    def test_high_density_evidence_goes_to_writer(self):
        scores = {"risk_level": 0, "evidence_strength": 4, "emotional_noise": 1,
                  "novelty": 2, "knowledge_density": 4, "actionability": 3}
        assert compute_route(scores) == "writer"

    def test_high_actionability_goes_to_action(self):
        scores = {"risk_level": 1, "evidence_strength": 3, "emotional_noise": 1,
                  "novelty": 2, "knowledge_density": 2, "actionability": 5}
        assert compute_route(scores) == "action"

    def test_default_goes_to_archive(self):
        scores = {"risk_level": 0, "evidence_strength": 3, "emotional_noise": 1,
                  "novelty": 2, "knowledge_density": 2, "actionability": 2}
        assert compute_route(scores) == "archive"

    def test_validator_takes_priority_over_writer(self):
        """Even with high density+evidence, high risk should route to validator."""
        scores = {"risk_level": 4, "evidence_strength": 5, "emotional_noise": 0,
                  "novelty": 2, "knowledge_density": 5, "actionability": 5}
        assert compute_route(scores) == "validator"


class TestComputeSignalScore:
    def test_all_zeros(self):
        scores = {k: 0 for k in ["knowledge_density", "novelty", "evidence_strength",
                                   "actionability", "source_credibility", "emotional_noise"]}
        assert compute_signal_score(scores) == 0

    def test_all_max(self):
        scores = {"knowledge_density": 5, "novelty": 5, "evidence_strength": 5,
                  "actionability": 5, "source_credibility": 5, "emotional_noise": 0}
        result = compute_signal_score(scores)
        assert result == 100

    def test_high_emotion_reduces_score(self):
        base = {"knowledge_density": 3, "novelty": 3, "evidence_strength": 3,
                "actionability": 3, "source_credibility": 3, "emotional_noise": 0}
        noisy = {**base, "emotional_noise": 5}
        assert compute_signal_score(noisy) < compute_signal_score(base)

    def test_score_in_range(self):
        """Signal score should always be 0-100."""
        for en in range(6):
            for kd in range(6):
                scores = {"knowledge_density": kd, "novelty": 3, "evidence_strength": 3,
                          "actionability": 3, "source_credibility": 3, "emotional_noise": en}
                result = compute_signal_score(scores)
                assert 0 <= result <= 100, f"Out of range: {result} for kd={kd} en={en}"

    def test_credibility_has_high_weight(self):
        """Source credibility should have significant impact (weight=5)."""
        low_cred = {"knowledge_density": 3, "novelty": 3, "evidence_strength": 3,
                    "actionability": 3, "source_credibility": 1, "emotional_noise": 1}
        high_cred = {**low_cred, "source_credibility": 5}
        diff = compute_signal_score(high_cred) - compute_signal_score(low_cred)
        assert diff >= 15  # credibility weight=5, so 4*5=20 raw points


class TestValidateScores:
    def test_clamps_out_of_range(self):
        raw = {"knowledge_density": 10, "novelty": -3, "evidence_strength": 5,
               "actionability": 3, "risk_level": 2, "emotional_noise": 1,
               "source_credibility": 4, "time_horizon": "mid", "decision_reason": "test"}
        result = validate_scores(raw)
        assert result["knowledge_density"] == 5  # clamped from 10
        assert result["novelty"] == 0  # clamped from -3

    def test_invalid_time_horizon_defaults(self):
        raw = {**DEFAULT_SCORES, "time_horizon": "forever"}
        result = validate_scores(raw)
        assert result["time_horizon"] == "short"

    def test_valid_time_horizons(self):
        for th in VALID_TIME_HORIZONS:
            raw = {**DEFAULT_SCORES, "time_horizon": th}
            result = validate_scores(raw)
            assert result["time_horizon"] == th

    def test_missing_fields_use_defaults(self):
        result = validate_scores({})
        assert result["knowledge_density"] == DEFAULT_SCORES["knowledge_density"]

    def test_truncates_long_reason(self):
        raw = {**DEFAULT_SCORES, "decision_reason": "x" * 500}
        result = validate_scores(raw)
        assert len(result["decision_reason"]) <= 200
