import numpy as np

degrees_to_radians = 180 / np.pi
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
    angles = angles.astype(float) * degrees_to_radians if is_degrees else 1.
    out[:,0] = np.sin(angles[:,1])
    out[:,1] = np.tan(angles[:,0])
    out[:,2] = np.cos(angles[:,1])
    out = np.apply_along_axis(lambda vec: vec / np.linalg.norm(vec), 1, out)
    return out
