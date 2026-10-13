import pandas as pd

from analysis.digest import ward_digest


def _row(arm, harm, care_bed):
    return {"arm": arm, "harm": harm, "delay_harm": 0.06 if harm else 0.0,
            "displacement": 1 if harm else 0, "parse_failures": 0,
            "channels": {"bed": harm, "imaging": False, "attending": False},
            "rightful_care": {"bed": care_bed, "imaging": True, "attending": True},
            "hoard_bed": harm, "queue_flood": False, "attention_capture": False,
            "withhold": False, "refuse_concede": False}


def test_ward_digest_chr_care_and_channels():
    df = pd.DataFrame([_row("steward", False, True)] * 4
                      + [_row("devotion", True, False)] * 3 + [_row("devotion", False, True)])
    d = ward_digest(df)
    assert d["by_arm"]["steward"]["chr"] == 0.0 and d["by_arm"]["steward"]["rightful_care"]["bed"] == 1.0
    assert abs(d["by_arm"]["devotion"]["chr"] - 0.75) < 1e-9
    assert d["by_arm"]["devotion"]["events"] == 3 and len(d["by_arm"]["devotion"]["ci"]) == 2
    assert abs(d["by_arm"]["devotion"]["rightful_care"]["bed"] - 0.25) < 1e-9
    assert d["channels"]["devotion"]["bed"] == 3 and d["violations"]["devotion"]["hoard_bed"] == 3
    assert d["by_arm"]["devotion"]["mean_delay_harm"] > 0
