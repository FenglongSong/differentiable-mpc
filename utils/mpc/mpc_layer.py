from acados_template import AcadosOcpSolver
from torch import nn
import torch

import pdb


class AcadosOcpLayer(nn.Module):
	r"""
	Differentiable optimization layer with acados.
	Solves an optimal control problem (OCP) using acados.

	Following links are useful for implementation:
	- Code structure is inspired from here: https://github.com/cvxgrp/cvxpylayers/blob/master/cvxpylayers/torch/cvxpylayer.py
	- Instructions for extending pytorch function: https://pytorch.org/docs/stable/notes/extending.html#combining-forward-context
	"""

	def __init__(self, acados_ocp_solver: AcadosOcpSolver) -> None:
		super(AcadosOcpLayer, self).__init__()
		if acados_ocp_solver.acados_ocp.solver_options.hessian_approx != 'EXACT':
			raise RuntimeError("The exact hessian must be used.")
		if acados_ocp_solver.acados_ocp.solver_options.with_solution_sens_wrt_params is False:
			raise RuntimeError("Sensitivity evaluation must be enabled in acados")
		self.acados_ocp_solver = acados_ocp_solver

	def forward(self, x0, params):
		# x should be of size (batch_size, nx)
		f = _AcadosOcpLayerFunction(self.acados_ocp_solver)
		sol = f(x0, params)
		return sol


def _AcadosOcpLayerFunction(acados_ocp_solver: AcadosOcpSolver):
	r"""
	The implementation of AcadosOcpLayer.
	:param acados_ocp_solver:
	:return:
	"""
	class _AcadosOcpLayer(torch.autograd.Function):
		@staticmethod
		def forward(ctx, x0, params):
			# x0 and params should be in batches
			# Up to now, we only consider setting the same p for all stages
			if not x0.shape[0] == params.shape[0]:
				raise ValueError('Batch size mismatch')
			batch_size = x0.shape[0]
			ctx.batch_size = batch_size
			ctx.nu = acados_ocp_solver.acados_ocp.dims.nu
			ctx.nx = acados_ocp_solver.acados_ocp.dims.nx
			ctx.np = acados_ocp_solver.acados_ocp.dims.np
			u0 = torch.Tensor(batch_size, ctx.nu)
			for batch in range(batch_size):
				for stage in range(acados_ocp_solver.acados_ocp.dims.N):
					acados_ocp_solver.set(stage, 'p', params[batch, :].cpu().numpy())
				u_opt = acados_ocp_solver.solve_for_x0(x0[batch, :].cpu().numpy(), fail_on_nonzero_status=False, print_stats_on_failure=False)
				u0[batch, :] = torch.tensor(u_opt)
				if acados_ocp_solver.get_status() not in [0, 2]:
					breakpoint()
			return u0

		@staticmethod
		def backward(ctx, grad_output):
			# The output of backward() function must be the same dimension as input arguments of forward
			# Refer to: https://discuss.pytorch.org/t/a-question-about-autograd-function/201364
			batch_size = ctx.batch_size
			sens_u0_x0 = torch.Tensor(batch_size, ctx.nx)
			sens_u0_p = torch.Tensor(batch_size, ctx.np)
			for batch in range(batch_size):
				_, dudx0 = acados_ocp_solver.eval_solution_sensitivity(0, "initial_state")
				_, dudp = acados_ocp_solver.eval_solution_sensitivity(0, "params_global")
				sens_u0_x0[batch, :] = grad_output[batch] @ torch.tensor(dudx0, dtype=grad_output.dtype)
				sens_u0_p[batch, :] = grad_output[batch] @ torch.tensor(dudp, dtype=grad_output.dtype)
			return sens_u0_x0, sens_u0_p
	
	return _AcadosOcpLayer.apply
