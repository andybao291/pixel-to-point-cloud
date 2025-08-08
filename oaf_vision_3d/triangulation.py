# %% [markdown]
# # Triangulation
#
# This function triangulates 3D points from two sets of two undistorted normalized
# pixels and a [`TransformationMatrix`](transformation_matrix.py) object. The process
# for this was discussed in more detail in the workshop
# [5: Dual Camera Setups](../workshops/05_dual_camera_setups.ipynb).


# %%
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def triangulate_points(  # type: ignore
    undistorted_normalized_pixels_0: NDArray[Shape["H, W, 2"], Float32],
    undistorted_normalized_pixels_1: NDArray[Shape["H, W, 2"], Float32],
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    H, W, _ = undistorted_normalized_pixels_0.shape

    # Flatten to [N, 2] → Add z=1 to make rays [N, 3]
    v0 = undistorted_normalized_pixels_0.reshape(-1, 2)
    v0 = np.concatenate([v0, np.ones((v0.shape[0], 1), dtype=np.float32)], axis=1)

    v1 = undistorted_normalized_pixels_1.reshape(-1, 2)
    v1 = np.concatenate([v1, np.ones((v1.shape[0], 1), dtype=np.float32)], axis=1)

    # Extract rotation matrix and translation vector from the transformation
    R = transformation_matrix.rotation.as_matrix()  # [3, 3]
    t = transformation_matrix.translation  # [3,]

    # Rotate v1 into camera 0's frame
    v1_rotated = v1 @ R.T  # [N, 3]

    # Get P1 (camera 1 origin in camera 0 frame)
    P1 = t.reshape(1, 3)  # [1, 3] will broadcast correctly

    # Compute dot products
    a = np.sum(v0 * v0, axis=1)
    b = np.sum(v0 * v1_rotated, axis=1)
    c = np.sum(v1_rotated * v1_rotated, axis=1)
    d = np.sum(v0 * P1, axis=1)
    e = np.sum(v1_rotated * P1, axis=1)

    denom = b**2 - a * c
    denom = np.where(denom == 0, 1e-6, denom)

    t_val = (b * e - c * d) / denom

    # Get the triangulated point in camera 0's frame
    points_3d = (v0.T * t_val).T  # [N, 3]

    return points_3d.reshape(H, W, 3).astype(np.float32)


def triangulate_disparity(
    disparity: NDArray[Shape["H, W"], Float32],
    lens_model_0: LensModel,
    lens_model_1: LensModel,
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    y, x = np.indices(disparity.shape, dtype=np.float32)
    pixels_0 = np.stack([x, y], axis=-1)
    pixels_1 = np.stack([x - disparity, y], axis=-1)

    undistortied_normalized_pixels_0 = lens_model_0.undistort_pixels(
        normalized_pixels=lens_model_0.normalize_pixels(pixels=pixels_0)
    )
    undistortied_normalized_pixels_1 = lens_model_1.undistort_pixels(
        normalized_pixels=lens_model_1.normalize_pixels(pixels=pixels_1)
    )

    return triangulate_points(
        undistorted_normalized_pixels_0=undistortied_normalized_pixels_0,
        undistorted_normalized_pixels_1=undistortied_normalized_pixels_1,
        transformation_matrix=transformation_matrix,
    )
