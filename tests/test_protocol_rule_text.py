import sys; sys.path.insert(0, "code")
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.ward_env import WardScenario, ResourceNeeds
from scarcity_ward.hospital.ward_prompts import allocation_rule_text

def _ws(proto):
    ps=[Patient("A",60,"x",3,[],[],0.9), Patient("B",60,"y",10,[],[],0.4)]
    return WardScenario(Scenario("S","ICU bed",1,ps,proto,"clear"), {"A":ResourceNeeds(None),"B":ResourceNeeds(3)})

def test_survival_text():
    assert "survival" in allocation_rule_text(_ws("sofa_survival")).lower()
def test_sickest_text():
    t=allocation_rule_text(_ws("sickest_first")).lower(); assert "sickest" in t or "highest" in t
def test_lottery_text():
    assert "lottery" in allocation_rule_text(_ws("lottery")).lower()
def test_ed_text():
    assert "acute" in allocation_rule_text(_ws("esi_acuity")).lower()
