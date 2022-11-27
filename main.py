import argparse

from raytracing.common import Settings

from raytracing.cpu_rt import CpuApp
from raytracing.cuda_rt import CudaApp

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="If provided save to file, otherwise show window",
    )

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--size",
        type=int,
        default=(320, 180),
        nargs=2,
        help="Image size",
    )
    size_group.add_argument(
        "--resolution",
        type=str,
        choices=["SD", "HD", "FHD", "2K", "4K", "8K"],
        help="Image resolution",
    )
    parser.add_argument(
        "--bounces",
        type=int,
        default=5,
        help="Number of times a ray can bounce",
    )

    args = parser.parse_args()

    if args.resolution:
        width, height = {
            "SD": (640, 360),
            "HD": (1280, 720),
            "FHD": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160),
            "8K": (7680, 4320),
        }[args.resolution]
    else:
        width, height = args.size

    settings = Settings(
        width=width,
        height=height,
        bounces=args.bounces,
    )

    if args.cuda:
        app = CudaApp(16, settings)
    else:
        app = CpuApp(settings)

    app.run()

    if args.output is None:
        plt.imshow(app.image)
        plt.show(block=True)
    else:
        try:
            plt.imsave(args.output, app.image)
        except ValueError:
            parser.error("Invalid output path")


if __name__ == "__main__":
    main()
