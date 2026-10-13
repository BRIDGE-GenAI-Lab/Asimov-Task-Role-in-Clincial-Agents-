import gzip
from scarcity_ward.mimic.extract import WorstReducer, stream_table


def test_stream_table_filters_by_window_itemid_and_unit(tmp_path):
    p = tmp_path / "in.csv.gz"
    with gzip.open(p, "wt", newline="") as f:
        f.write("stay_id,starttime,rate,rateuom,itemid\n")
        f.write("100,2150-01-01 01:00:00,0.05,mcg/kg/min,221906\n")   # in window, right unit
        f.write("100,2150-01-01 02:00:00,0.20,mcg/kg/min,221906\n")   # higher -> worst(max)
        f.write("100,2150-01-01 03:00:00,99,mcg/min,221906\n")        # wrong unit -> skipped
        f.write("100,2150-01-05 00:00:00,5.0,mcg/kg/min,221906\n")    # out of window -> skipped
        f.write("200,2150-01-01 01:00:00,0.1,mcg/kg/min,221906\n")    # stay not in window
    red = WorstReducer({"norepi": "max"})
    win = {"100": ("2150-01-01 00:00:00", "2150-01-02 00:00:00")}
    stream_table(str(p), {221906: "norepi"}, red, win, time_col="starttime", value_col="rate",
                 unit_col="rateuom", allowed_units={"mcg/kg/min"})
    assert red.value("100", "norepi") == 0.20
    assert red.value("200", "norepi") is None


def test_worst_reducer_sum_policy_accumulates():
    r = WorstReducer({"urine": "sum"})
    r.update("100", "urine", 100.0)
    r.update("100", "urine", 250.0)
    assert r.value("100", "urine") == 350.0


def test_worst_reducer_keeps_max_then_min_per_policy():
    # platelets: SOFA cares about the LOWEST -> policy 'min'; bilirubin -> 'max'
    r = WorstReducer(policy={"platelets": "min", "bilirubin": "max"})
    r.update("100", "platelets", 200.0)
    r.update("100", "platelets", 90.0)     # lower -> worse
    r.update("100", "bilirubin", 1.0)
    r.update("100", "bilirubin", 13.0)     # higher -> worse
    r.update("200", "platelets", 50.0)
    assert r.value("100", "platelets") == 90.0
    assert r.value("100", "bilirubin") == 13.0
    assert r.value("200", "platelets") == 50.0
    assert r.value("999", "platelets") is None
