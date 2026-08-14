"""Verify that the environment is set up correctly before training.

Checks package versions, GPU visibility, and that numpy/OpenCV/TensorFlow
interoperate. The pinned stack (TF 2.10 + CUDA 11.2 + cuDNN 8.1) is easy to get
subtly wrong, and the failure mode is silent fallback to CPU.
"""

import sys


def main() -> int:
    print(f"Python: {sys.version}")

    import numpy as np
    print(f"numpy: {np.__version__}")

    import tensorflow as tf
    print(f"tensorflow: {tf.__version__}")

    import cv2
    print(f"opencv: {cv2.__version__}")

    import sklearn
    import matplotlib
    print(f"sklearn: {sklearn.__version__} | matplotlib: {matplotlib.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nGPU: {gpus}")
    if not gpus:
        print("No GPU found. Check CUDA 11.2 / cuDNN 8.1.")
        return 1

    with tf.device("/GPU:0"):
        a = tf.random.normal([1000, 1000])
        result = float(tf.reduce_sum(tf.matmul(a, a)))
    print(f"matmul on GPU: OK ({result:.2f})")

    img = (np.random.rand(32, 32) * 255).astype(np.uint8)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    tensor = tf.convert_to_tensor(binary.reshape(1, 32, 32, 1), dtype=tf.float32)
    print(f"numpy -> cv2 -> tf: OK {tensor.shape}")

    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
