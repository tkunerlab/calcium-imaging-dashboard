from unittest.mock import patch

import numpy as np

from calcium_imaging_dashboard.cell_registration_dashboard import main


class FakeQualityDatabase:
    def __init__(self):
        self.loads = 0

    def get_sessions_for_mouse(self, mouse, cohort):
        return [{
            "cohort": cohort,
            "mouse": mouse,
            "session_type": "Training",
            "session_name": "Session01",
            "display_name": "Training_Session01",
        }]

    def load_session_quality(self, cohort, mouse, session_type, session_name):
        self.loads += 1
        return {
            "common": {
                "FootprintArea": np.array([4.0, 12.0, 30.0]),
                "TemporalContrast": np.array([1.0, 3.0, 5.0]),
                "FootprintEccentricity": np.array([0.2, 0.95, 0.4]),
            },
            "source": {
                "Accepted": np.array([1, 0, -1], dtype=np.int8),
                "TemporalSNR": np.array([np.nan, 1.5, 8.0]),
            },
        }


def _request(**overrides):
    values = {
        "cohort": "Cohort",
        "mouse": "Animal",
        "is_overview": False,
        "session_type": "Training",
        "session_name": "Session01",
    }
    values.update(overrides)
    return values


def test_candidate_detection_is_non_mutating_and_reports_every_failed_rule():
    fake = FakeQualityDatabase()
    with patch.object(main, "db", fake):
        result = main.run_autoclean(main.AutoCleanRequest(**_request(
            min_footprint_area=10,
            min_temporal_contrast=2,
            max_footprint_eccentricity=0.9,
            flag_source_rejected=True,
        )))
    assert result["mutated"] is False
    assert result["sessions"][0]["candidate_indices"] == [0, 1]
    rules = {
        item["index"]: {failure["rule"] for failure in item["failed_rules"]}
        for item in result["sessions"][0]["candidates"]
    }
    assert rules[0] == {"min_footprint_area", "min_temporal_contrast"}
    assert rules[1] == {"max_footprint_eccentricity", "source_rejected"}
    assert fake.loads == 1


def test_disabled_thresholds_and_missing_source_values_do_not_fail_cells():
    fake = FakeQualityDatabase()
    with patch.object(main, "db", fake):
        disabled = main.run_autoclean(main.AutoCleanRequest(**_request()))
        missing = main.run_autoclean(main.AutoCleanRequest(**_request(
            min_source_snr=2.0
        )))
    assert disabled["total_candidates"] == 0
    assert missing["sessions"][0]["candidate_indices"] == [1]


def test_quality_stats_returns_single_cell_common_and_source_metrics():
    fake = FakeQualityDatabase()
    with patch.object(main, "db", fake):
        result = main.get_cell_quality_stats(main.CellQualityRequest(**_request(
            selected_cell_index=1
        )))
    assert result["selected_cell"]["common"]["FootprintArea"] == 12.0
    assert result["selected_cell"]["source"]["Accepted"] == 0
    assert result["accepted_counts"] == {
        "accepted": 1,
        "rejected": 1,
        "unknown": 1,
    }
    assert set(result["histograms"]) == {
        "FootprintArea",
        "TemporalContrast",
        "FootprintEccentricity",
        "SourceTemporalSNR",
        "SpatialCorrelation",
        "ClassifierScore",
    }
    assert result["histograms"]["FootprintEccentricity"]["counts"]
