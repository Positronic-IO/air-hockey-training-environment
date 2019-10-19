from environment.puck import Puck
from environment.mallet import Mallet


def computer(puck: Puck, opponent: Mallet) -> None:
    """ The 'AI' of the computer """

    if puck.x < opponent.x:
        if puck.x < opponent.left_lim:
            opponent.dx = 1
        else:
            opponent.dx = -2

    if puck.x > opponent.x:
        if puck.x > opponent.right_lim:
            opponent.dx = -1
        else:
            opponent.dx = 2

    if puck.y < opponent.y:
        if puck.y < opponent.u_lim:
            opponent.dy = 1
        else:
            opponent.dy = -6

    if puck.y > opponent.y:
        if puck.y <= 360:  # was 250
            opponent.dy = 6
        # elif puck.y<=350:
        #    left_mallet.dy = 2
        else:
            if opponent.y > 200:
                opponent.dy = -2
            else:
                opponent.dy = 0
        # Addresses situation when the puck and the computer are on top of each other.
        # Breaks loop
        if abs(puck.y - opponent.y) < 40 and abs(puck.x - opponent.x) < 40:
            puck.dx += 2
            puck.dy += 2

    opponent.update()
