# %% [markdown]
# # Plane Sweeping
#
# This function performs plane sweeping to estimate the depth of a pixel in a set of 2D
# images. The process for this was discussed in more detail in the workshop
# [7: Stereo Matching Fundamentals Continues](../workshops/07_stereo_matching_fundamentals_continued.ipynb).


# %%

from typing import Union

import numpy as np
from nptyping import Float32, NDArray, Shape
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.poly_2_subvalue_fit import find_subvalue_poly_2
from oaf_vision_3d.project_points import project_points
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def repeoject_image_at_depth(
    image: NDArray[Shape["H, W, ..."], Float32],
    camera_vectors: NDArray[Shape["H, W, 3"], Float32],
    depth: float,
    lens_model: LensModel,
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, ..."], Float32]:
    xyz = camera_vectors * depth

    projected_points = project_points(
        points=xyz.reshape(-1, 3),
        lens_model=lens_model,
        transformation_matrix=transformation_matrix.inverse(),
    ).reshape(*camera_vectors.shape[:2], 2)

    return np.stack(
        [
            map_coordinates(
                input=_image,
                coordinates=[projected_points[..., 1], projected_points[..., 0]],
                order=1,
                mode="constant",
                cval=np.nan,
            )
            for _image in image.transpose(2, 0, 1)
        ],
        axis=-1,
        dtype=np.float32,
    )


def plane_sweeping(
    image: NDArray[Shape["H, W, ..."], Float32],
    lens_model: LensModel,
    secondary_images: list[NDArray[Shape["H, W, ..."], Float32]],
    secondary_lens_models: list[LensModel],
    secondary_transformation_matrices: list[TransformationMatrix],
    depth_range: NDArray[Shape["2"], Float32],
    step_size: float,
    block_size: Union[int, NDArray[Shape["2"], Float32]],
    subpixel_fit: bool = True,
) -> NDArray[Shape["H, W, 3"], Float32]:

    # Create camera vectors for the main image
    pixels = np.indices(image.shape[:2], dtype=np.float32)[::-1].transpose((1, 2, 0))
    undistorted_normalized_pixels = lens_model.undistort_pixels(
        normalized_pixels=lens_model.normalize_pixels(pixels=pixels)
    )
    camera_vectors = np.pad(
        undistorted_normalized_pixels, ((0, 0), (0, 0), (0, 1)), constant_values=1.0
    )

    # Create depth range
    depths = np.arange(
        start=depth_range[0],
        stop=depth_range[1] + step_size,
        step=step_size,
        dtype=np.float32,
    )

    # Accumulate errors from all secondary images
    total_error = []

    for depth in depths:
        depth_errors = []

        # Compare with each secondary image
        for sec_image, sec_lens_model, sec_transform in zip(
            secondary_images, secondary_lens_models, secondary_transformation_matrices
        ):
            # Reproject secondary image at this depth
            reprojected_image = repeoject_image_at_depth(
                image=sec_image,
                camera_vectors=camera_vectors,
                depth=depth,
                lens_model=sec_lens_model,
                transformation_matrix=sec_transform,
            )

            # Calculate error between main image and reprojected secondary image
            single_pixel_error = np.abs(image - reprojected_image).sum(axis=-1)
            depth_errors.append(single_pixel_error)

        # Average error across all secondary images
        if len(depth_errors) > 1:
            averaged_error = np.stack(depth_errors).mean(axis=0)
        else:
            averaged_error = depth_errors[0]

        # Handle block_size as either int or array
        if isinstance(block_size, (int, np.integer)):
            bx, by = block_size, block_size
        else:
            bx, by = int(block_size[0]), int(block_size[1])

        # Apply block matching (convolution for smoothing)
        convoluted_error = convolve2d(
            convolve2d(averaged_error, np.ones((1, bx)) / bx, mode="same"),
            np.ones((by, 1)) / by,
            mode="same",
        )
        total_error.append(convoluted_error)

    error_array = np.array(total_error, dtype=np.float32)

    # Find best depth for each pixel
    if subpixel_fit:
        best_depth = find_subvalue_poly_2(values=depths, function_value=error_array)
    else:
        best_depth = depths[np.argmin(error_array, axis=0)].astype(np.float32)

    # Set invalid depths to NaN
    best_depth[best_depth >= depths.max()] = np.nan
    best_depth[best_depth <= depths.min()] = np.nan

    # Convert depth to 3D coordinates
    xyz = camera_vectors * best_depth[..., None]

    return xyz