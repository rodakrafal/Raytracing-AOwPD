import matplotlib.pyplot as plt
import numba.cuda as cuda
import numpy as np

import cmath

import numba

from raytracing.common import Settings, Sphere
from raytracing.app import App


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True,
)
def add(vec1, vec2):
    return vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2]

@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True,
)
def sub(vec1, vec2):
    return vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]


@cuda.jit(
    func_or_sig='float32[:], float32',
    device=True,
)
def mul_scalar(vec, div):
    return vec[0] * div, vec[1] * div, vec[2] * div


@cuda.jit(
    func_or_sig='float32[:], float32',
    device=True,
)
def div_scalar(vec, div):
    return mul_scalar(vec, 1 / div)


@cuda.jit(
    func_or_sig='float32[:],',
    device=True,
)
def normalize(v):
    square_sum = 0
    for vi in v:
        square_sum += vi * vi

    return div_scalar(v, square_sum ** 0.5)


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True
)
def dot(vec1, vec2):
    output = 0
    for value1, value2 in zip(vec1, vec2):
        output += value1 * value2
    return output


@cuda.jit(
    func_or_sig="float32[:], float32[:]",
    device=True,
)
def set_color(point, color):
    for i in range(3):
        point[i] = color[i]


@cuda.jit(
    func_or_sig="int64, int64, float32",
    device=True,
)
def map_to_screen(coord, range_width, ratio):
    return (coord - range_width) / range_width / ratio

@cuda.jit(
    func_or_sig="float32[:],",
    device=True,
)
def to_tuple(array):
    return array[0], array[1], array[2]

@cuda.jit(
    func_or_sig="float32[:], float32[:], float32[:]",
    device=True,
)
def intersect(sphere_pos, ray_origin, ray_dir) -> float:
    temp = cuda.local.array(shape=(3,), dtype=numba.float32)

    oc = sub(ray_origin, sphere_pos[:3])

    a = dot(ray_dir, ray_dir)  # ray_dir is normalized so this is always 1.0

    temp[:] = oc
    half_b = dot(temp, ray_dir)
    c = dot(temp, temp) - sphere_pos[3] ** 2

    discriminant = half_b * half_b - a * c

    if discriminant < 0:
        return cmath.inf
    else:
        ret = (-half_b - discriminant ** 0.5) / a
        return cmath.inf if ret < 0 else ret


# returns:
#   [
#       point_of_intersection[3]
#       normal[3]
#       color[3]
#       other_info[3]
#   ]
@cuda.jit(
    func_or_sig="float32[:, :, :], float32[:], float32[:]",
    device=True,
)
def shoot_ray(spheres, ray_origin, ray_dir):
    dist_to_nearest = cmath.inf
    nearest = -1
    for i, (sphere_pos, sphere_color) in enumerate(spheres):
        dist = intersect(sphere_pos, ray_origin, ray_dir)
        if dist < dist_to_nearest:
            dist_to_nearest = dist
            nearest = i

    if nearest == -1:
        return (
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (cmath.inf, 0.0, 0.0),
        )

    # Find the point of intersection on the object and its normal
    point_of_intersection = cuda.local.array(shape=(3,), dtype=numba.float32)
    point_of_intersection[:] = mul_scalar(ray_dir, dist_to_nearest)
    point_of_intersection[:] = add(point_of_intersection, ray_origin)

    normal = cuda.local.array(shape=(3,), dtype=numba.float32)
    normal[:] = sub(point_of_intersection, spheres[nearest][0][:3])
    normal[:] = normalize(normal)

    # Find color of the hit object
    color = cuda.local.array(shape=(3,), dtype=numba.float32)
    color[:] = to_tuple(spheres[nearest][1][:3])

    light_pos = cuda.local.array(shape=(3,), dtype=numba.float32)
    light_pos[:] = (5.0, 5.0, -10.0)

    light_color = cuda.local.array(shape=(3,), dtype=numba.float32)
    light_color[:] = (1.0, 1.0, 1.0)

    dir_to_light = cuda.local.array(shape=(3,), dtype=numba.float32)
    dir_to_light[:] = sub(light_pos, point_of_intersection)
    dir_to_light[:] = normalize(dir_to_light)

    dir_to_origin = cuda.local.array(shape=(3,), dtype=numba.float32)
    dir_to_origin[:] = sub(ray_origin, point_of_intersection)
    dir_to_origin[:] = normalize(dir_to_origin)

    ambient = 0.0
    ray_color = cuda.local.array(shape=(3,), dtype=numba.float32)
    ray_color[:] = (ambient, ambient, ambient)

    temp = cuda.local.array(shape=(3,), dtype=numba.float32)
    temp[:] = mul_scalar(color, max(dot(normal, dir_to_light), 0))

    ray_color[:] = add(ray_color, temp)

    temp[:] = add(dir_to_light, dir_to_origin)
    temp[:] = normalize(temp)
    temp2 = dot(normal, temp)

    specular_k = 50.0

    temp[:] = mul_scalar(light_color, max(temp2, 0.0) ** specular_k)
    ray_color[:] = add(ray_color, temp)

    return (
        to_tuple(point_of_intersection),
        to_tuple(normal),
        to_tuple(ray_color),
        (0.0, 0.0, 0.0),
    )


@cuda.jit(func_or_sig="float32[:, :, :], float32[:], int64, float32, float32",
    device=True,
)
def get_pixel_color(spheres, camera_position, bounces, x, y):
    ray_color = cuda.local.array(shape=(3,), dtype=numba.float32)
    ray_color[:] = (0.0, 0.0, 0.0)

    ray_origin = cuda.local.array(shape=(3,), dtype=numba.float32)
    ray_origin[:] = to_tuple(camera_position)

    ray_dir = cuda.local.array(shape=(3,), dtype=numba.float32)
    ray_dir[:2] = (x, y)
    ray_dir[2] = 0.0

    ray_dir[:] = sub(ray_dir, ray_origin)
    ray_dir[:] = normalize(ray_dir)

    reflection_strength = 1.0
    hit_poi = cuda.local.array(shape=(3,), dtype=numba.float32)
    hit_normal = cuda.local.array(shape=(3,), dtype=numba.float32)
    hit_color = cuda.local.array(shape=(3,), dtype=numba.float32)
    temp = cuda.local.array(shape=(3,), dtype=numba.float32)
    for bounce in range(bounces + 1):
        hit = shoot_ray(spheres, ray_origin, ray_dir)
        if hit[3][0] == cmath.inf:
            if bounce == 0:
                ray_color[:] = ((x + 1.0) / 2, (y + 1.0) / 2, 0.0)
            break

        hit_poi[:] = hit[0]
        hit_normal[:] = hit[1]
        hit_color[:] = hit[2]
        # create reflected ray
        temp[:] = mul_scalar(hit_normal, 0.0001)
        ray_origin[:] = add(hit_poi, temp)

        temp[:] = mul_scalar(hit_normal, 2 * dot(ray_dir, hit_normal))

        temp[:] = sub(ray_dir, temp)
        ray_dir[:] = normalize(temp)

        temp[:] = mul_scalar(hit_color, reflection_strength)
        ray_color[:] = add(ray_color, temp)

        reflection_strength /= 2

    return to_tuple(ray_color)


@cuda.jit(
    func_or_sig="float32[:, :, :], float32[:, :, :], float32[:], int64",
    device=False,
)
def fill_image(image, spheres, camera_position, bounces):
    y, x = cuda.grid(2)

    height, width, _ = image.shape

    if y < height and x < width:
        image[height - y - 1, x][:] = get_pixel_color(
            spheres,
            camera_position,
            bounces,
            map_to_screen(x, width // 2, 1.0),
            map_to_screen(y, height // 2, width / height),
        )
    cuda.syncthreads()


class CudaApp(App):
    def __init__(self, threadsperblock: int, settings: Settings):
        super().__init__(settings)
        self.threadsperblock = threadsperblock

    def run(self):
        blockspergrid_x = int(np.ceil(self.image.shape[0] / self.threadsperblock))
        blockspergrid_y = int(np.ceil(self.image.shape[1] / self.threadsperblock))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        d_image = cuda.to_device(self.image)
        fill_image[blockspergrid, (self.threadsperblock, self.threadsperblock)](
            d_image,
            cuda.to_device(self.world_to_cuda()),
            cuda.to_device(self.settings.camera_position),
            self.settings.bounces,
        )
        self.image = d_image.copy_to_host()

    def world_to_cuda(self):
        spheres = []

        for s in self.world.objs:
            spheres.append(
                np.array([
                    np.array((*s.center, s.radius), dtype=np.float32),
                    np.array((*s.color, 0.0), dtype=np.float32),
                ]),
            )

        return np.array(spheres)
