"""
Integrators.
"""


def explicit_euler(f, x, u, dt):
	"""
	Integrate a given function with an explicit Euler method.
	:param f: dynamics
	:param x: current states
	:param u: inputs
	:param dt: time step
	:return: next states
	"""
	return x + dt * f(x, u, dt)


def runge_kutta_4(f, x, u, dt):
	"""
	4th order Runge-Kutta method.
	:param f: dynamics
	:param x: current states
	:param u: inputs
	:param dt: time step
	:return: next states
	"""
	k1 = f(x, u)
	k2 = f(x + dt / 2 * k1, u)
	k3 = f(x + dt / 2 * k2, u)
	k4 = f(x + dt * k3, u)
	x_plus = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
	return x_plus
