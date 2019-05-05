import pytest


from utils import get_config_strategy

def test_rl_strategy():
    """ Test a rl strategy """
    assert isinstance(get_config_strategy("c51"), dict)

def test_undefined_strategy():
    """ Test a strategy not defined """

    with pytest.raises(KeyError):
        get_config_strategy("hi")