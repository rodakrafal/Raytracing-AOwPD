from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Settings:
    width: int = 1280
    height: int = 720
    bounces: int = 0
    camera_position: NDArray[np.float32] = np.array([0.0, 0.0, -1.0], dtype=np.float32)


@dataclass
class Sphere:
    center: NDArray[np.float32]
    radius: float
    color: NDArray[np.float32]


@dataclass
class World:
    objs: list[Sphere]
    light_color: NDArray[np.float32] = np.array([1.0, 1.0, 1.0])
    light_pos: NDArray[np.float32] = np.array([5.0, 5.0, -10.0])
    specular_k: float = 50.0
    settings: Settings = Settings()
