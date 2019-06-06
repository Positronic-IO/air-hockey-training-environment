import pytest

from rl.utils import get_config_strategy


class TestStrategyConfig:
    def setup(self):
        pass

    def test_c51_strategy(self):
        """ Test c51 strategy """
        assert isinstance(get_config_strategy("c51"), dict)

    def test_dueling_strategy(self):
        """ Test dueling strategy """
        assert isinstance(get_config_strategy("dueling"), dict)

    def test_ddqn_strategy(self):
        """ Test ddqn strategy """
        assert isinstance(get_config_strategy("ddqn"), dict)

    def test_qlearner_strategy(self):
        """ Test qlearner strategy """
        assert isinstance(get_config_strategy("q-learner"), dict)

    def test_a2c_strategy(self):
        """ Test a2c strategy """
        assert isinstance(get_config_strategy("a2c"), dict)

    def test_undefined_strategy(self):
        """ Test a strategy not defined """

        with pytest.raises(KeyError):
            get_config_strategy("hi")
