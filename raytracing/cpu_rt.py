from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from raytracing.common import Sphere, World
from raytracing.app import App


@dataclass
class Hit:
    point: NDArray[np.float32]
    normal: NDArray[np.float32]
    color: NDArray[np.float32]


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
    dir_to_origin = -ray_dir

    ray_color = np.zeros(3, dtype=np.float32)
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
    for bounce in range(world.settings.bounces + 1):
        hit = shoot_ray(world, ray_origin, ray_dir)
        if hit is None:
            if bounce == 0:
                ray_color = np.array([(x + 1.0) / 2, (y + 1.0) / 2, 0.0])
            break

        # create reflected ray
        ray_origin = hit.point
        ray_dir = normalize(ray_dir - 2 * dot(ray_dir, hit.normal) * hit.normal)
        ray_color += reflection_strength * hit.color
        reflection_strength /= 2

    return ray_color


class CpuApp(App):
    def run(self):
        for j, y in enumerate(np.linspace(
                -1.0 / self.screen_ratio,
                1.0 / self.screen_ratio,
                self.settings.height,
        )):
            for i, x in enumerate(np.linspace(
                    -1.0,
                    1.0,
                    self.settings.width,
            )):
                self.image[self.settings.height - j - 1, i, :] = np.clip(get_pixel_color(self.world, x, y), 0.0, 1.0)
