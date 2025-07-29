# %% [markdown]
# # Undistort Image
#
# This function undistorts an image using a new camera matrix as done in the
# [workshop](../workshops/03_image_distortion_and_undistortion.ipynb).

import numpy as np

# %%
from nptyping import Float32, NDArray, Shape
from scipy.ndimage._interpolation import map_coordinates

from oaf_vision_3d.lens_model import CameraMatrix, LensModel, normalize_pixels


def undistort_image_with_new_camera_matrix(  # type: ignore
    image: NDArray[Shape["H, W, 3"], Float32],
    lens_model: LensModel,
    new_camera_matrix: CameraMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    y, x = np.indices(image.shape[:2])
    pixels = np.stack([x, y], axis=-1)
    normalized_pixels = normalize_pixels(pixels=pixels, camera_matrix=new_camera_matrix)
    distorted_normalized_pixels = lens_model.distort_pixels(
        normalized_pixels=normalized_pixels
    )
    distorted_pixels = lens_model.denormalize_pixels(pixels=distorted_normalized_pixels)

    undistorted_image = np.stack(
        [
            map_coordinates(
                input=_image,
                coordinates=[distorted_pixels[..., 1], distorted_pixels[..., 0]],
                order=1,
            )
            for _image in image.transpose(2, 0, 1)
        ],
        axis=-1,
    )
    return undistorted_image.astype(np.float32)
