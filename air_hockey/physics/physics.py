from air_hockey.object.shapes import LineShape, CircleShape
from air_hockey.object.objects import StaticObject
import numpy as np

# Mapping dictionary to call appropriate function based on types of objects provided
f_map = {
    (LineShape, LineShape): lambda x, y: line_line_collision(line_obj_a=x, line_obj_b=y),
    (CircleShape, CircleShape): lambda x, y: circle_circle_collision(circle_obj_a=x, circle_obj_b=y),
    (LineShape, CircleShape): lambda x, y: line_circle_collision(line_obj=x, circle_obj=y),
    (CircleShape, LineShape): lambda x, y: line_circle_collision(line_obj=y, circle_obj=x),
}


def rigid_body_physics(world, dt):
    obj_list = world.get_object_list()

    for o in obj_list:
        if isinstance(o, StaticObject):
            continue

        o.apply_pulse(dt)  # apply_pulse updates the objects velocity resulting from any applied forces
        o.limit_speed()  # Limit speed of each object to their defined maximums

    for i in range(len(obj_list)):
        for j in range(i + 1, len(obj_list)):  # Loop over all pair sof objects and check for collisions
            if obj_list[i].ignore_physics or obj_list[j].ignore_physics:
                continue

            if isinstance(obj_list[i], StaticObject) and isinstance(obj_list[j], StaticObject):
                continue  # Both static, no collision can occur

            f_map[(type(obj_list[i].shape), type(obj_list[j].shape))](obj_list[i], obj_list[j])

    for o in obj_list:  # Update object positions based on new velocities resulting from pulse and collisions
        if isinstance(o, StaticObject):
            continue

        o.apply_new_position(dt)


def circle_circle_collision(circle_obj_a, circle_obj_b):
    """
    Check for and handle collision between two objects using a Circle Shape
    """
    x_a = circle_obj_a.x
    r_a = circle_obj_a.r

    x_b = circle_obj_b.x
    r_b = circle_obj_b.r

    d2 = np.sum((x_a - x_b) ** 2)
    if d2 < (r_a + r_b) ** 2:
        n = (x_b - x_a) / np.sqrt(d2)
        p = (x_a * r_a + x_b * r_b) / (r_a + r_b)
        resolve_impact(circle_obj_a, circle_obj_b, n, p)
    pass


def line_line_collision(line_obj_a, line_obj_b):
    """
    Check for and handle collision between two objects using a LineShape
    """
    lx_a = line_obj_a.x
    line_length_a = line_obj_a.length
    lx0_a = line_obj_a.x0
    lx1_a = line_obj_a.x1

    lx_b = line_obj_b.x
    line_length_b = line_obj_b.length
    lx0_b = line_obj_b.x0
    lx1_b = line_obj_b.x1

    p = line_line_intersection(lx0_a, lx1_a, lx0_b, lx1_b)
    if len(p) < 2:
        return

    dp_a = line_length_a / 2 - np.sqrt((p[0] - lx_a[0]) ** 2 + (p[1] - lx_a[1]) ** 2)
    dp_b = line_length_b / 2 - np.sqrt((p[0] - lx_b[0]) ** 2 + (p[1] - lx_b[1]) ** 2)
    if dp_a > 0 and dp_b > 0:
        if dp_a >= dp_b:  # Impact is closer to edge of line b than line a. Use line a to create impact normal
            n = np.array([lx1_a[1] - lx0_a[1], -1 * (lx1_a[0] - lx0_a[0])]) / line_length_a
            if np.dot(n, lx_b - lx_a) < 0:
                n = n * -1
            resolve_impact(line_obj_a, line_obj_b, n, p)
        else:
            n = np.array([lx1_b[1] - lx0_b[1], -1 * (lx1_b[0] - lx0_b[0])]) / line_length_b
            if np.dot(n, lx_a - lx_b) < 0:
                n = n * -1
            resolve_impact(line_obj_b, line_obj_a, n, p)
    pass


def line_line_intersection(x0_a, x1_a, x0_b, x1_b):
    """
    Find the intersection point for two lines defined by the provided end points
    """

    x1, y1 = x0_a
    x2, y2 = x1_a
    x3, y3 = x0_b
    x4, y4 = x1_b

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if np.abs(denom) > 0:
        p = (
            np.array(
                [
                    (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4),
                    (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4),
                ]
            )
            / denom
        )
    else:
        p = np.array([])

    return p


def line_circle_collision(line_obj, circle_obj):
    """
    Check for and handle collision between an object with a LineShape and one with a CircleShape

    See the following page for more information in calculating distances between a point (e.g., circle center) and line
    https://www.intmath.com/plane-analytic-geometry/perpendicular-distance-point-line.php
    """
    lx = line_obj.x  # Get position of line
    line_length = line_obj.length
    lx0 = line_obj.x0
    lx1 = line_obj.x1

    cx = circle_obj.x  # Get position of circle
    cr = circle_obj.r

    # If circle is not in bounding box of line then no collision is occurring
    if cx[0] < lx0[0] - cr and cx[0] < lx1[0] - cr:
        return
    if cx[0] > lx0[0] + cr and cx[0] > lx1[0] + cr:
        return
    if cx[1] < lx0[1] - cr and cx[1] < lx1[1] - cr:
        return
    if cx[1] > lx0[1] + cr and cx[1] > lx1[1] + cr:
        return

    # Calculate distance from circle center to nearest point on line.
    # If they are sufficiently far apart then assume no collision and return
    A = -(lx1[1] - lx0[1])
    B = lx1[0] - lx0[0]
    C = -(A * lx0[0] + B * lx0[1])
    if (A * cx[0] + B * cx[1] + C) ** 2 / (A * A + B * B) > cr * cr:
        return

    p = cx - [A * C + A * (A * cx[0] + B * cx[1]), B * C + B * (A * cx[0] + B * cx[1])] / (A ** 2 + B ** 2)

    dp = np.dot((lx1 - lx) / (line_length / 2), p - lx)
    if np.abs(dp) > line_length / 2:
        p = lx + (lx1 - lx) * np.sign(dp)

    d2 = (p[0] - cx[0]) ** 2 + (p[1] - cx[1]) ** 2

    if d2 < cr ** 2:  # If true there is a collision between the two objects
        n = [1, 0] if d2 == 0 else (p - cx) / np.sqrt(d2)

        resolve_impact(circle_obj, line_obj, n, p)

    return


def resolve_impact(obj_a, obj_b, n, p):
    """
    If a collision is detected between two objects this function is called to resolve it.
    Collisions resolved by calculating post-collision velocities for each object.

    The following site detailing rigid body collisions was used when writing this function
    https://www.myphysicslab.com/engine2D/collision-en.html
    """
    if isinstance(obj_a, StaticObject):  # Make sure if we have a static object it is obj_b
        obj_a, obj_b = obj_b, obj_a  # this function will (should) never be called for two static objects
        n = n * -1  # Need to swap direction of impact normal if swapping objects

    x_a = obj_a.x  # Properties of object a
    v1_a = obj_a.v
    w1_a = obj_a.w
    m_a = obj_a.mass
    I_a = obj_a.get_moment_of_inertia()
    cor_a = obj_a.cor  # Coefficient of restitution

    x_b = obj_b.x  # Properties of object b
    v1_b = obj_b.v
    w1_b = obj_b.w
    m_b = obj_b.mass
    I_b = obj_b.get_moment_of_inertia()
    cor_b = obj_b.cor  # Coefficient of restitution

    if cor_a == 0 or cor_b == 0:  # Effective coefficient of restitution is harmonic mean or individual CORs
        e = 0
    else:
        e = 2 / (1 / cor_a + 1 / cor_b)

    r_ap = p - x_a  # Vector from center of each object to point of impact
    r_bp = p - x_b

    v_ap1 = v1_a + [-w1_a * r_ap[1], w1_a * r_ap[0]]  # Velocity of body a at impact point p
    v_bp1 = v1_b + [-w1_b * r_bp[1], w1_b * r_bp[0]]  # Velocity of body b at impact point p
    v_ab1 = v_ap1 - v_bp1  # Velocity of body a relative to body b at impact point

    if isinstance(obj_b, StaticObject):  # If one object is static/stationary
        j = (
            -(1 + e) * np.dot(v_ab1, n) / (1 / m_a + (r_ap[0] * n[1] - r_ap[1] * n[0]) ** 2 / I_a)
        )  # Net impulse of collision
        if j < 0:
            obj_a.v = v1_a + j * n / m_a
            obj_a.w = w1_a + j * (r_ap[0] * n[1] - r_ap[1] * n[0]) / I_a
    else:  # If both objects are dynamic
        j = (
            -(1 + e)
            * np.dot(v_ab1, n)
            / (
                1 / m_a
                + 1 / m_b
                + (r_ap[0] * n[1] - r_ap[1] * n[0]) ** 2 / I_a
                + (r_bp[0] * n[1] - r_bp[1] * n[0]) ** 2 / I_b
            )
        )  # Net impulse of collision
        if j < 0:
            obj_a.v = v1_a + j * n / m_a
            obj_a.w = w1_a + j * (r_ap[0] * n[1] - r_ap[1] * n[0]) / I_a
            obj_b.v = v1_b - j * n / m_b
            obj_b.w = w1_b - j * (r_bp[0] * n[1] - r_bp[1] * n[0]) / I_b
    return
