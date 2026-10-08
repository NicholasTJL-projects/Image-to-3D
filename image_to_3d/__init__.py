"""image_to_3d: 2D captures -> 3D Gaussian Splatting scene.

Pipeline stages (each one is also a CLI subcommand):

    capture  -> extract sharp frames from a video / webcam into images/
    sfm      -> run COLMAP to recover camera poses and a sparse point cloud
    init     -> turn the sparse point cloud into an initial Gaussian cloud
    train    -> optimise the Gaussians against the captured images (PyTorch)
    render   -> rasterise the Gaussian cloud from any camera (NumPy or torch)
"""

from .camera import Camera
from .gaussians import GaussianCloud
from .scene import Scene

__all__ = ["Camera", "GaussianCloud", "Scene"]
__version__ = "0.1.0"
