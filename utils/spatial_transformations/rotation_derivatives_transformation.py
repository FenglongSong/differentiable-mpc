import numpy as np

def global_angular_velocity_to_quaternion_time_derivative(omega, quat):
	"""
	Transforms angular velocity (global, i.e., in world frame) to time derivative of quaternion
	:param omega: [omega_x, omega_y, omega_z]
	:param quat: [qx, qy, qz, qw]
	:return: [qw_dot, qx_dot, qy_dot, qz_dot]

	Reference: Robot Dynamics, (2.94) - (2.99)
	"""
	qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
	H = np.array([
		[-qx, -qy, -qz],
		[qw, qz, -qy],
		[-qz, qw, qx],
		[qy, -qx, qw]])
	return 1/2 * H @ omega

def local_angular_velocity_to_quaternion_time_derivative(omega, quat):
	"""
	Transforms angular velocity (local, i.e., in body frame) to time derivative of quaternion
	:param omega: [omega_x, omega_y, omega_z]
	:param quat: [qx, qy, qz, qw]
	:return: [qx_dot, qy_dot, qz_dot, qw_dot]

	Reference: Robot Dynamics, (2.94) - (2.99)
	"""
	qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
	H = np.array([
		[qw, qz, -qy],
		[-qz, qw, qx],
		[qy, -qx, qw],
		[-qx, -qy, -qz]])
	return 1/2 * H @ omega


def get_mapping_local_angular_velocity_to_euler_angles_xyz_time_derivative(euler_angles):
	sy = np.sin(euler_angles[1])
	cy = np.cos(euler_angles[1])
	sz = np.sin(euler_angles[2])
	cz = np.cos(euler_angles[2])
	cz_cy = cz / cy
	sz_cy = sz / cy

	M = np.array([
		[cz_cy,       -sz_cy,   	0.0],
		[sz,           cz,   		0.0],
		[-sy * cz_cy,  sy * sz_cy,  1.0]
	])
	return M

