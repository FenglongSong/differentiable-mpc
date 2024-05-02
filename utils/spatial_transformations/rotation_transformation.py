import numpy as np
from scipy.spatial.transform import Rotation


def skew_matrix(vec):
	"""
	Return the skew symmetric of a 3-d vector
	:param vec:
	:return:
	"""
	if np.size(vec) != 3:
		raise ValueError("Input must be a 3 dimension array")
	mat = np.array([
		[0., -vec[2], vec[1]],
		[vec[2], 0., -vec[0]],
		[-vec[1], vec[0], 0.]
	])
	return mat


def rotation_matrix_from_quaternion(quat):
	# rotation = Rotation.from_quat(quat)
	# return rotation.as_matrix()[0]
	w = quat[3]
	xyz = quat[0:3]
	return (2*w**2-1.)*np.eye(3) + 2.*w*skew_matrix(xyz) + 2.*xyz.reshape(-1,1) @ xyz.reshape(-1,1).T



