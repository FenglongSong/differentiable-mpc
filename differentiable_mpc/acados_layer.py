from typing import Any

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from torch import nn
import torch.nn.functional as F
import torch

# Code structure is inspired from here: https://github.com/cvxgrp/cvxpylayers/blob/master/cvxpylayers/torch/cvxpylayer.py

class AcadosOcpLayer(nn.Module):
	def __init__(self, acados_ocp_solver: AcadosOcpSolver) -> None:
		super(AcadosOcpLayer, self).__init__()
		if acados_ocp_solver.acados_ocp.solver_options.hessian_approx != 'EXACT':
			raise ValueError("The exact hessian must be used.")
		self.acados_ocp_solver = acados_ocp_solver

	def forward(self, x0, params):
		# x should be of size (batch_size, nx)
		f = _AcadosOcpLayerFunction(self.acados_ocp_solver)
		sol = f(x0, params)
		return sol


def _AcadosOcpLayerFunction(acados_ocp_solver: AcadosOcpSolver):
	class _AcadosOcpLayer(torch.autograd.Function):
		@staticmethod
		def forward(ctx, x0, params):
			# x0 and params should be in batches
			# Up to now, we only consider setting the same p for all stages
			if not x0.shape[0] == params.shape[0]:
				raise ValueError('Batch size mismatch')
			batch_size = x0.shape[0]

			nu = acados_ocp_solver.acados_ocp.dims.nu
			ctx.nu = nu
			u0 = torch.Tensor(batch_size, ctx.nu)
			for batch in range(batch_size):
				for stage in range(acados_ocp_solver.acados_ocp.dims.N):
					acados_ocp_solver.set(stage, 'p', params[batch, :].cpu().numpy())  # might need to transform params to numpy array
				u_opt = acados_ocp_solver.solve_for_x0(x0[batch, :].cpu().numpy(), fail_on_nonzero_status=False, print_stats_on_failure=False)
				u0[batch, :] = torch.tensor(u_opt)
				if acados_ocp_solver.get_status() not in [0, 2]:
					breakpoint()
			# raise Exception(f'acados returned status {status}.')
			return u0

		@staticmethod
		def backward(ctx, u0):
			# ? which stage should I use when getting param sensitivity?
			sens_x_x0, sens_u_x0 = acados_ocp_solver.eval_solution_sensitivity(0, "initial_state")
			sens_x_p, sens_u_p = acados_ocp_solver.eval_solution_sensitivity(0, "params_global")
			return sens_u_x0, sens_u_p

	return _AcadosOcpLayer.apply
