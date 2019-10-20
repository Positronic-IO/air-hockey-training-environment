import numpy as np
from typing import List


class BaseShape:
    """
    Parent class for Shapes. Every shape will include an x-position, angle, and color used when rendering.
    """

    __slots__: List[str] = ["x", "ang", "color"]

    def __init__(self, x, ang, color=None):
        self.x = np.squeeze(np.array(x))
        self.ang = ang
        if color is None:
            color = [255, 255, 255]
        self.color = np.array(color).astype(np.float64)

    def __setattr__(self, name, value):
        if name == "x":  # Ensure proper shape for x
            super().__setattr__(name, np.squeeze(np.array(value)))
        else:
            super().__setattr__(name, value)

    def get_moment_of_inertia(self, m):
        raise Exception("Shape child class needs to define get_moment_of_inertia function")


class CircleShape(BaseShape):
    """
    Circle shapes also require a radius.
    """

    def __init__(self, x, r, ang=0, color=None):

        self.r = float(r)
        super(CircleShape, self).__init__(x=x, ang=ang, color=color)

    def get_moment_of_inertia(self, m):
        return 0.5 * m * self.r ** 2

    def draw(self, arr, scale):
        """
        Render the circle onto the provided array (arr)
        """
        x = self.x * scale
        r = self.r * scale

        lb_x = np.max([np.floor(x[0] - r), 0]).astype(np.int).item()
        ub_x = np.min([np.ceil(x[0] + r), arr.shape[0]]).astype(np.int).item()
        lb_y = np.max([np.floor(x[1] - r), 0]).astype(np.int).item()
        ub_y = np.min([np.ceil(x[1] + r), arr.shape[1]]).astype(np.int).item()
        view_x = np.arange(lb_x, ub_x)
        view_y = np.arange(lb_y, ub_y)

        tmp_view = arr[lb_x:ub_x, lb_y:ub_y, :]

        d = (view_x[:, np.newaxis] - x[0]) ** 2 + (view_y[np.newaxis, :] - x[1]) ** 2
        tmp_view[d < r * r, :] = self.color


class LineShape(BaseShape):
    """
    LineShapes can be defined one of two ways:
    -A center point (x), angle, and length.
    -Two endpoints x0 and x1

    This shape can be initialized using either of these definitions. If either definition of the line is modified then
    the other definition is updated to reflect this. For example, if the endpoint x0 is modified then new value for
    x, ang, length are calculated to reflect this.
    """

    def __init__(self, x=None, ang=0, length=10, x0=None, x1=None, y0=None, y1=None, color=None):
        # If user defines using two endpoints, convert to (x, ang, L) definition
        # x0, x1 will then be recalculated and assigned below
        if x0 is not None:
            x = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2).item()
            ang = np.arctan2(y1 - y0, x1 - x0).item()

        super(LineShape, self).__init__(x=x, ang=ang, color=color)

        self.length = length
        self.calc_endpoints()
        return

    def __setattr__(self, name, value):
        super(LineShape, self).__setattr__(name, value)
        if name in ["x", "length", "ang"]:  # If one of these values are updated then [x0, x1] need to be recalculated
            self.calc_endpoints()
        elif name in ["x0", "x1"]:  # If an endpoint is updated need to recalculate [x, length, ang]
            self.calc_line_parameters()

    def calc_endpoints(self):
        """
        Calculate endpoints based on [x, length, ang]
        """
        if all(
            hasattr(self, attr) for attr in ["x", "length", "ang"]
        ):  # Can only calculate these if x, length, and ang are set
            # Need to set using self.__dict__ to avoid recursively calling self.__setattr__
            self.__dict__["x0"] = np.array(
                [self.x[0] - self.length / 2 * np.cos(self.ang), self.x[1] - self.length / 2 * np.sin(self.ang)]
            )
            self.__dict__["x1"] = np.array(
                [self.x[0] + self.length / 2 * np.cos(self.ang), self.x[1] + self.length / 2 * np.sin(self.ang)]
            )

    def calc_line_parameters(self):
        """
        Calculate [x, length, ang] based on endpoints
        :return:
        """
        if all(hasattr(self, attr) for attr in ["x0", "x1"]):  # Can only calculate these if x0 and x1 are both set
            x0 = self.x0[0]
            y0 = self.x0[1]
            x1 = self.x1[0]
            y1 = self.x1[1]

            # Need to set using self.__dict__ to avoid recursively calling self.__setattr__
            self.__dict__["x"] = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            self.__dict__["length"] = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2).item()
            self.__dict__["ang"] = np.arctan2(y1 - y0, x1 - x0).item()

    def get_moment_of_inertia(self, m):
        return 1 / 12 * m * self.length ** 2

    def draw(self, arr, scale):
        """
        Render line to provided array
        """
        x0 = self.x0
        x1 = self.x1

        x0 = x0 * scale
        x1 = x1 * scale
        w = 1.414 / 2

        lb_x = np.max([np.min([np.floor(x0[0]), np.floor(x1[0])]), 0]).astype(np.int).item()
        ub_x = np.min([np.max([np.ceil(x0[0]), np.ceil(x1[0])]), arr.shape[0] - 1]).astype(np.int).item()
        lb_y = np.max([np.min([np.floor(x0[1]), np.floor(x1[1])]), 0]).astype(np.int).item()
        ub_y = np.min([np.max([np.ceil(x0[1]), np.ceil(x1[1])]), arr.shape[1] - 1]).astype(np.int).item()

        view_x = np.arange(lb_x, ub_x + 1)
        view_y = np.arange(lb_y, ub_y + 1)

        tmp_view = arr[lb_x : ub_x + 1, lb_y : ub_y + 1, :]

        A = -(x1[1] - x0[1])
        B = x1[0] - x0[0]
        C = -(A * x0[0] + B * x0[1])

        d = (A * view_x[:, np.newaxis] + B * view_y[np.newaxis, :] + C) ** 2 / (A * A + B * B)
        tmp_view[d < w, :] = self.color
