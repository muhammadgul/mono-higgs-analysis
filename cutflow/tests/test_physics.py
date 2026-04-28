from monohiggs_cutflow.physics import dphi


def test_dphi_wrap():
    assert abs(dphi(0.1, 2*3.14159265-0.1) - 0.2) < 1e-3
