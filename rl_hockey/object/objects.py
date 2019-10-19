import numpy as np


class BaseObject:
    """
    This is the parent class for all objects that will be present in a World.

    Every object includes a shape object (line, circle, rectangle, etc.) from the Shape class.
    In addition to this, every object contains several additional properties describing its state and
    attributes necessary for the physics engine (velocity, force acting on object, mass, etc.)
    """
    def __init__(self, shape, v=None, max_v=400, w=0, max_w=3.1415, mass=1, cor=0.8, tau=1000,
                 ignore_physics=False, **kwargs):
        if v is None:
            v = [0, 0]

        self.shape = shape(**kwargs)     # Shape, also contains current position and orientation
        self.v = np.array(v)             # Velocity
        self.f = np.zeros(self.v.shape)  # Force acting in current time step
        self.w = w                       # Angular velocity
        self.torque = 0                  # Torque acting in current time step
        self.mass = mass                 # Object mass
        self.cor = cor                   # Coefficient of restitution
        self.max_v = max_v               # Maximum allowable speed
        self.max_w = max_w               # Maximum allowable angular speed
        self.ignore_physics = ignore_physics  # Objects that ignore physics will not be considered by the physics engine
        self.tau = tau                        # Used to apply drag on velocity/angular velocity
        return

    def __setattr__(self, name, value):
        if name in ['x', 'ang', 'color', 'x0', 'x1', 'length', 'r']:
            setattr(self.shape, name, value)  # Set values in the contained Shape object
        elif name in ['v', 'f', 'w']:
            super(BaseObject, self).__setattr__(name, np.array(value))  # Ensure these are stored as arrays
        else:
            super(BaseObject, self).__setattr__(name, value)

    def __getattr__(self, name):
        if name in ['x', 'ang', 'color', 'x0', 'x1', 'length', 'r']:
            return getattr(self.shape, name)  # These are properties of shape
        else:
            return self.__dict__[name]

    def limit_speed(self):
        # Limit maximum speed and angular speed
        s = np.sqrt(self.v[0]**2 + self.v[1]**2)
        if s > self.max_v:
            self.v = self.v*self.max_v/s

        if np.abs(self.w) > self.max_w:
            self.w = np.sign(self.w)*self.max_w

    def draw(self, arr, scale):
        # If an object is to be drawn then call the Shape's draw method
        self.shape.draw(arr, scale)
        return

    def get_moment_of_inertia(self):
        return self.shape.get_moment_of_inertia(self.mass)

    def update(self):
        pass


class StaticObject(BaseObject):
    """
    Static objects are considered to be fixed by the physics engine.

    **kwargs: Parameters for the defined shape. See Shapes.py for further information.
    """
    def __init__(self, shape, **kwargs):
        super(StaticObject, self).__init__(shape, v=None, **kwargs)
        pass


class DynamicObject(BaseObject):
    """
    Dynamic objects are subject to changes in position or velocity due to the physics engine.

    There may be external forces acting on these objects (e.g., gravity),
    however they are not controlled.

    **kwargs: Parameters for the defined shape. See Shapes.py for further information.
    """
    def __init__(self, shape, controller=None, v=None, w=0, mass=1, **kwargs):
        self.controller = controller
        super(DynamicObject, self).__init__(shape, v=v, w=w, mass=mass, **kwargs)
        pass

    def apply_action(self, inputs): # No action can be applied to these
        pass

    def apply_pulse(self, dt):
        """
        If there are forces acting on the object in this time step then resolve those forces as changes
        in velocity.
        """
        self.v = self.f*dt/self.mass + self.v
        self.w = self.torque*dt / self.shape.get_moment_of_inertia(self.mass) + self.w

        self.v = self.v * np.exp(-dt/self.tau)

        self.f = self.f*0  # Zero out forces and torques
        self.torque = self.torque*0

    def apply_new_position(self, dt):
        """
        Update position based on velocity
        """
        self.shape.x = self.shape.x + self.v*dt
        self.shape.ang = self.shape.ang + self.w*dt
        return


class ControlledCircle(DynamicObject):
    """
    This is an agent object subject to control. The given inputs are applied as forces to the object.
    """
    def __init__(self, shape, force_mag=1, **kwargs):
        self.force_mag = force_mag
        super(ControlledCircle, self).__init__(shape, **kwargs)
        pass

    def apply_action(self, inputs):
        """
        Apply forces to the object based on the given inputs. Multiple forces may be applied,
        for example the input may be 'UP LEFT'.
        """
        if 'LEFT' in inputs:
            self.f[0] -= self.force_mag
        if 'RIGHT' in inputs:
            self.f[0] += self.force_mag
        if 'UP' in inputs:
            self.f[1] += self.force_mag
        if 'DOWN' in inputs:
            self.f[1] -= self.force_mag
