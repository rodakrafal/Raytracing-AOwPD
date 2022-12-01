import argparse

from raytracing.common import Settings

from raytracing.cpu_rt import CpuApp
from raytracing.cuda_rt import CudaApp

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("--width", type=int, default=320, help="Image width")
    parser.add_argument("--height", type=int, default=180, help="Image height")
    parser.add_argument("--bounces", type=int, default=5, help="Number of times a ray can bounce")

    args = parser.parse_args()

    settings = Settings(
        width=args.width,
        height=args.height,
        bounces=args.bounces,
    )

    if args.cuda:
        app = CudaApp(16, settings)
    else:
        app = CpuApp(settings)

    app.run()

    plt.imshow(app.image)
    plt.show(block=True)


if __name__ == "__main__":
    main()
