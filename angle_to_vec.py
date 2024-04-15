import numpy as np

DEGREES_TO_RADIANS = np.pi / 180


# def pitchyaw_to_vector(angles, is_degrees=True):
#     r"""Convert given gaze pitch and yaw to vector.
#     Args:
#         angles (:obj:`numpy.array`): gaze pitch (column 0) and yaw (column 1) :math:`(n\times 2)`.
#         is_degrees (bool): specifies whether pitch and yaw are given in degrees. If False, angles are given in radians.
#     Returns:
#         :obj:`numpy.array` of shape :math:`(n\times 3)` specifying (X, Y, Z) vector where positive X direction is to right of head, positive Y is straight up from head, and positive Z is going into the face.
#     """
#     n = angles.shape[0]
#     out = np.empty((n, 3))
#     angles = angles.astype(float) * degrees_to_radians if is_degrees else 1.
#     out[:,0] = np.sin(angles[:,1])
#     out[:,1] = np.tan(angles[:,0])
#     out[:,2] = np.cos(angles[:,1])
#     out = np.apply_along_axis(lambda vec: vec / np.linalg.norm(vec), 1, out)
#     return out
def pitchyaw_to_vector(angles, is_degrees=True):
    r"""Convert given gaze pitch and yaw to vector.
    Args:
        angles (:obj:`numpy.array`): gaze pitch (column 0) and yaw (column 1) :math:`(n\times 2)`.
        is_degrees (bool): specifies whether pitch and yaw are given in degrees. If False, angles are given in radians.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` specifying (X, Y, Z) vector where positive X direction is to right of head, positive Y is straight up from head, and positive Z is going into the face.
    """
    n = angles.shape[0]
    out = np.empty((n, 3))
    if is_degrees:
        angles = angles * DEGREES_TO_RADIANS
    out[:, 0] = np.sin(angles[:, 1])  # Yaw
    out[:, 1] = np.sin(
        angles[:, 0]
    )  # Pitch, assuming pitch is up/down and should be applied to Y
    out[:, 2] = np.sqrt(1 - out[:, 0] * out[:, 0] - out[:, 1] * out[:, 1])
    out[:, 2] = -out[:, 2]  # Negate z direction
    # out = np.apply_along_axis(lambda vec: vec / np.linalg.norm(vec), 1, out)
    return out


if __name__ == "__main__":
    test = np.array(
        [[0.00, 0.09], [-6.13, 4.80], [4.77, 10.84], [-13.43, -5.19], [10.74, 3.40]]
    )
    # gt is not perfect... but close
    gt = np.array(
        [
            [0.0016, 0.0, -1.0],
            [0.0832, -0.1068, -0.9908],
            [0.1874, 0.0831, -0.9788],
            [-0.0880, -0.2322, -0.9687],
            [0.0582, 0.1863, -0.9808],
        ]
    )
    vecs = pitchyaw_to_vector(test)
    print(test)
    print(gt)
    print(vecs)
    print(np.linalg.norm(vecs, axis=1))
    # Should be approx
    # 0.0016, 0.0000, -1.0000
    #
