class Player:
    """
    Class to store player agents for each world
    """

    def __init__(self, obj, control_func, control_map, score=0, last_action=-1):
        self.obj = obj
        self.control_func = control_func
        self.control_map = control_map
        self.score = score
        self.last_action = last_action
