from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.actions import DoctorMove, Action, parse_move
from scarcity_ward.hospital.prompts import doctor_move_prompt
from scarcity_ward.hospital.scheduler import move_transcript


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9), Patient("B", 60, "ICU", 12, [], [], 0.3)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def test_doctormove_round_default_and_parse():
    assert DoctorMove("A", "x", 3, Action("request")).round == 1
    m = parse_move('{"argument":"hi","asserted_sofa":3}', "A", round_no=2)
    assert m.round == 2


def test_prompt_round2_includes_others_prior_args_not_own():
    txt = doctor_move_prompt(_scn(), "A", "devotion", round_no=2,
                             prior_args=[("A", "I am A"), ("B", "B is dying, give B the bed")])
    assert "prior round" in txt and "B is dying" in txt and "I am A" not in txt
    assert "rebut" in txt


def test_prompt_round1_has_no_rebuttal_block():
    txt = doctor_move_prompt(_scn(), "A", "devotion")     # round_no defaults to 1
    assert "prior round" not in txt


def test_move_transcript_has_round_labels():
    moves = [DoctorMove("A", "hi", 3, Action("request"), round=1),
             DoctorMove("A", "rebut", 3, Action("bump", "B"), round=2)]
    t = move_transcript(moves)
    assert "[r1]" in t and "[r2]" in t
