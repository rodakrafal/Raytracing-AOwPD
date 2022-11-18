from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt


@dataclass
class Settings:
    width: int = 1280
    height: int = 720
    bounces: int = 0
    ambient: float = 0.1
    sphere_count: int = 20
    samples_per_ray: int = 50
    camera_position: NDArray[np.float32] = np.array([0.0, 0.35, -1.0])


@dataclass
class Sphere:
    center: NDArray[np.float32]
    radius: float
    color: NDArray[np.float32]


@dataclass
class Hit:
    point: NDArray[np.float32]
    normal: NDArray[np.float32]
    color: NDArray[np.float32]


@dataclass
class World:
    objs: list[Sphere]
    light_color: NDArray[np.float32] = np.array([1.0, 1.0, 1.0])
    light_pos: NDArray[np.float32] = np.array([5.0, 5.0, -10.0])
    specular_k: float = 50.0
    settings: Settings = Settings()


def normalize(v: NDArray[np.float32]) -> NDArray[np.float32]:
    square_sum = 0
    for vi in v:
        square_sum += vi * vi

    return v / np.sqrt(square_sum)


def dot(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    return np.sum(u * v)


def intersect(sphere: Sphere, ray_origin: NDArray[np.float32], ray_dir: NDArray[np.float32]) -> float:
    oc = ray_origin - sphere.center

    a = dot(ray_dir, ray_dir)  # ray_dir is normalized so this is always 1.0
    half_b = dot(oc, ray_dir)
    c = dot(oc, oc) - sphere.radius ** 2

    discriminant = half_b * half_b - a * c

    if discriminant < 0:
        return float("inf")
    else:
        ret = (-half_b - discriminant ** 0.5) / a
        return float("inf") if ret < 0 else ret


def is_in_shadow(world: World, sphere: Sphere, point: NDArray[np.float32], dir_to_light: NDArray[np.float32]) -> bool:
    for obj in world.objs:
        dist = intersect(obj, point, dir_to_light)
        if obj is not sphere and  dist < float("inf"):
            return True

    return False


def shoot_ray(world: World, ray_origin: NDArray[np.float32], ray_dir: NDArray[np.float32]) -> Optional[Hit]:
    # Find the point of intersection with the scene
    dist_to_nearest = float("inf")
    nearest: Optional[Sphere] = None
    for obj in world.objs:
        dist = intersect(obj, ray_origin, ray_dir)
        if dist < dist_to_nearest:
            dist_to_nearest = dist
            nearest = obj

    if nearest is None:
        return None

    # Find the point of intersection on the object and its normal
    point_of_intersection = ray_origin + ray_dir * dist_to_nearest
    normal = normalize(point_of_intersection - nearest.center)

    # Find color of the hit object
    color = nearest.color
    dir_to_light = normalize(world.light_pos - point_of_intersection)
    dir_to_origin = normalize(ray_origin - point_of_intersection)

    ray_color = np.zeros(3, dtype=np.float32) * world.settings.ambient
    if not is_in_shadow(world, nearest, point_of_intersection, dir_to_light):
        ray_color += max(dot(normal, dir_to_light), 0) * color

    ray_color += max(
        dot(
            normal,
            normalize(dir_to_light + dir_to_origin)),
        0,
    ) ** world.specular_k * world.light_color

    return Hit(point_of_intersection, normal, ray_color)


def get_pixel_color(world: World, x: int, y: int) -> NDArray[np.float32]:
    ray_color = np.zeros(3, dtype=np.float32)
    ray_origin = world.settings.camera_position
    ray_dir = normalize(np.array([x, y, 0]) - ray_origin)

    reflection_strength = 1.0
    for _ in range(world.settings.bounces + 1):
        hit = shoot_ray(world, ray_origin, ray_dir)
        if hit is None:
            break

        # create reflected ray
        ray_origin = hit.point + hit.normal * 0.0001
        ray_dir = normalize(ray_dir - 2 * dot(ray_dir, hit.normal) * hit.normal)
        ray_color += reflection_strength * hit.color
        reflection_strength /= 2

    return ray_color


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

    def run(self):
        world = self.create_world()

        for j, y in enumerate(np.linspace(
                self.screen_coords[1],
                self.screen_coords[3],
                self.settings.height,
        )):
            for i, x in enumerate(np.linspace(
                    self.screen_coords[0],
                    self.screen_coords[2],
                    self.settings.width,
            )):
                self.image[self.settings.height - j - 1, i, :] = np.clip(get_pixel_color(world, x, y), 0.0, 1.0)

    def create_world(self):
        w = World(
            objs=[
                sphere_orange,
                sphere_pink,
                sphere_blue,
            ]
        )

        w.settings = self.settings
        return w


settings = Settings(
    width=800,
    height=800,
    bounces=5,
    sphere_count=1,
    samples_per_ray=1,
)

app = App(settings)
app.run()

plt.imshow(app.image)
plt.show(block=True)
