from typing import List

import numpy as np

from raytracing.common import Settings, Sphere, World


class App:
    def __init__(self, settings: Settings):
        self.scene: List[Sphere] = []
        self.camera: NDArray[np.float32]

        self.settings = settings

        screen_ratio = settings.width / settings.height
        self.screen_coords = np.array([
            -1.0,
            -1.0 / screen_ratio,
            1.0,
            1.0 / screen_ratio,
        ])

        self.image = np.zeros(
            (settings.height, settings.width, 3),
            dtype=np.float32,
        )

        self.world = self.create_world()

    def run(self):
        raise NotImplementedError

    def create_world(self):
        sphere_orange = Sphere(
            center=np.array([-2.75, 0.1, 3.5]),
            radius=0.6,
            color=np.array([1.0, 0.572, 0.184]),
        )
        sphere_pink = Sphere(
            center=np.array([-0.75, 0.1, 2.25]),
            radius=0.6,
            color=np.array([0.5, 0.223, 0.5]),
        )
        sphere_blue = Sphere(
            center=np.array([0.75, 0.1, 1.0]),
            radius=0.6,
            color=np.array([0.0, 0.0, 1.0]),
        )
        sphere_floor = Sphere(
            center=np.array([0.0, -105.0, 0.0]),
            radius=100,
            color=np.array([1.0, 1.0, 1.0]),
        )
        sphere_background = Sphere(
            center=np.array([-5.0, 0.0, 10.0]),
            radius=5,
            color=np.array([56, 255, 18]) / 255.0,
        )
        sphere_behind_camera = Sphere(
            center=np.array([0.0, 0.0, -150.0]),
            radius=80,
            color=np.array([203, 15, 255]) / 255.0,
        )

        w = World(
            objs=[
                sphere_orange,
                sphere_pink,
                sphere_blue,
                sphere_floor,
                sphere_background,
                sphere_behind_camera,
            ]
        )

        w.settings = self.settings
        return w
