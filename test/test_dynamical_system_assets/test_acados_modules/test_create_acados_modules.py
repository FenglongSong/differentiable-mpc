import unittest
import numpy as np
import yaml
import sys
sys.path.append('/')
from dynamical_system_assets.acados_modules.create_acados_modules import create_acados_ocp_solver
from dynamical_system_assets.double_integrator import DoubleIntegrator

class TestCreateAcadosModules(unittest.TestCase):
	def test_create_double_integrator_acados_modules(self):
		config_file = '/dynamical_system_assets/config/double_integrator_config.yaml'
		with open(config_file, 'r') as f:
			config = yaml.load(f, Loader=yaml.SafeLoader)

		print(config)
		double_integrator = DoubleIntegrator()
		ocp_solver = create_acados_ocp_solver(double_integrator, config)

		nx = ocp_solver.acados_ocp.model.x.rows()
		nu = ocp_solver.acados_ocp.model.u.rows()
		N = ocp_solver.acados_ocp.dims.N
		Tf = ocp_solver.acados_ocp.solver_options.tf
		simX = np.zeros((N + 1, nx))
		simU = np.zeros((N, nu))

		_ = ocp_solver.solve_for_x0(np.array([0., 1.]))
		ocp_solver.print_statistics()
		status = ocp_solver.get_status()
		if status != 0:
			raise Exception(f'acados returned status {status}.')

		# get solution
		for i in range(N):
			simX[i, :] = ocp_solver.get(i, "x")
			simU[i, :] = ocp_solver.get(i, "u")
		simX[N, :] = ocp_solver.get(N, "x")

		cost_opt = ocp_solver.get_cost()
		print(cost_opt)
		print(simU[0, :])

		sens_x_x0, sens_u_x0 = ocp_solver.eval_solution_sensitivity(0, "initial_state")
		print(sens_u_x0)
		sens_x_p, sens_u_p = ocp_solver.eval_solution_sensitivity(0, "params_global")
		print(sens_u_p)


if __name__ == '__main__':
	unittest.main()
