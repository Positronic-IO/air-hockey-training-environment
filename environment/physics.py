import numpy as np

from environment import config
from environment.mallet import Mallet
from environment.puck import Puck


def collision(puck: Puck, mallet: Mallet, correction: bool = True) -> bool:
    """ Collision resolution

    Reference:
        https://www.gamedev.net/forums/topic/488102-circlecircle-collision-response/
        https://gamedevelopment.tutsplus.com/tutorials/how-to-create-a-custom-2d-physics-engine-the-basics-and-impulse-resolution--gamedev-6331
    """
    # separation vector
    d_x = mallet.x - puck.x
    d_y = mallet.y - puck.y
    d = np.array([d_x, d_y])

    #  distance between circle centres, squared
    distance_squared = np.dot(d, d)

    # combined radius squared
    radius = mallet.radius + puck.radius
    radius_squared = radius ** 2

    # No collision
    if distance_squared > radius_squared:
        return False

    # distance between circle centres
    distance = np.sqrt(distance_squared)

    # normal of collision
    ncoll = (d / distance) if distance > 0 else d

    # penetration distance
    dcoll = radius - d

    # Sum of inverse masses
    imass_sum = puck.imass + mallet.imass

    # separation vector
    if correction:
        # For floating point corrections
        percent = config.physics["percent"]  # usually 20% to 80%
        slop = config.physics["slop"]  # usually 0.01 to 0.1
        separation_vector = (np.max(dcoll - slop, 0) / imass_sum) * percent * ncoll
    else:
        separation_vector = (dcoll / imass_sum) * ncoll

    # separate the circles
    puck.x -= separation_vector[0] * puck.imass
    puck.y -= separation_vector[1] * puck.imass
    mallet.x += separation_vector[0] * mallet.imass
    mallet.y += separation_vector[1] * mallet.imass

    # combines velocity
    vcoll_x = mallet.dx - puck.dx
    vcoll_y = mallet.dy - puck.dy
    vcoll = np.array([vcoll_x, vcoll_y])

    # impact speed
    vn = np.dot(vcoll, ncoll)

    # obejcts are moving away. dont reflect velocity
    if vn > 0:
        return True  # we did collide

    # coefficient of restitution in range [0, 1].
    cor = config.physics["restitution"]  # air hockey -> high cor

    # collision impulse
    j = -(1.0 + cor) * (vn / imass_sum)

    # collision impusle vector
    impulse = j * ncoll

    # change momentum of the circles
    puck.dx -= impulse[0] * puck.imass
    puck.dy -= impulse[1] * puck.imass

    mallet.dx += impulse[0] * mallet.imass
    mallet.dy += impulse[1] * mallet.imass

    return True
