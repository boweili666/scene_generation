import numpy as np


class VizColor:
    goal = [48 / 255, 245 / 255, 93 / 255, 0.3]
    collision_volume = [0.1, 0.1, 0.1, 0.7]
    collision_volume_ignored = [0.1, 0.1, 0.1, 0.0]
    obstacle_debug = [0.5, 0.5, 0.5, 0.7]
    obstacle_task = [0.5, 0.5, 0.5, 0.7]
    safe = [0, 1, 0, 0.5]
    hold = [245 / 255, 243 / 255, 48 / 255, 0.5]
    unsafe = [0 / 255, 32 / 255, 230 / 255, 0.5]
    safe_zone = [255 / 255, 165 / 255, 0 / 255, 0.5]
    violation = [160 / 255, 32 / 255, 240 / 255, 0.5]
    collision = [255 / 255, 0 / 255, 0 / 255, 0.8]
    not_a_number = [255 / 255, 0 / 255, 0 / 255, 0.4]
    x_axis = [1, 0, 0, 1]
    y_axis = [0, 1, 0, 1]
    z_axis = [0, 0, 1, 1]


class Geometry:
    def __init__(self, type, **kwargs):
        self.type = type
        self.attributes = {}
        self.color = kwargs.get("color", np.array([1, 1, 1, 0.5]))

        if self.type == "sphere":
            self.type_code = 0
            radius = kwargs["radius"]
            self.attributes["radius"] = radius
            self.size = radius * np.ones(3)
        elif self.type == "box":
            self.type_code = 1
            self.attributes["length"] = kwargs["length"]
            self.attributes["width"] = kwargs["width"]
            self.attributes["height"] = kwargs["height"]
            self.size = np.array(
                [kwargs["length"], kwargs["width"], kwargs["height"]],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown geometry type: {self.type}")

    def get_attributes(self):
        return self.attributes
