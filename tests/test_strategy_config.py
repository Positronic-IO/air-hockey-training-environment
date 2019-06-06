import pytest

from rl.utils import get_config_strategy


class TestStrategyConfig:
    def setup(self):
        pass

    def test_rl_strategy(self):
        """ Test a rl strategy """
        assert isinstance(get_config_strategy("c51"), dict)

    def test_undefined_strategy(self):
        """ Test a strategy not defined """

        with pytest.raises(KeyError):
            get_config_strategy("hi")
