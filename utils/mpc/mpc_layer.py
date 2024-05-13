from acados_template import AcadosOcpSolver
from torch import nn
import torch

import pdb
from utils.mpc.mpc_solver import BaseMpcSolver, AcadosMpcSolver


class MpcLayer(nn.Module):
	mpc_solver: BaseMpcSolver

	def __init__(self, mpc_solver: BaseMpcSolver) -> None:
		super(MpcLayer, self).__init__()
		self.mpc_solver = mpc_solver

	def forward(self, x0, params):
		# x should be of size (batch_size, nx)
		f = _AcadosOcpLayerFunction(self.acados_ocp_solver)
		sol = f(x0, params)
		return sol


class AcadosMpcLayer(nn.Module):
	r"""
	Differentiable optimization layer with acados.
	Solves an optimal control problem (OCP) using acados.

	Following links are useful for implementation:
	- Code structure is inspired from here: https://github.com/cvxgrp/cvxpylayers/blob/master/cvxpylayers/torch/cvxpylayer.py
	- Instructions for extending pytorch function: https://pytorch.org/docs/stable/notes/extending.html#combining-forward-context
	"""

	def __init__(self, acados_ocp_solver: AcadosOcpSolver) -> None:
		super(AcadosMpcLayer, self).__init__()
		if acados_ocp_solver.acados_ocp.solver_options.hessian_approx != 'EXACT':
			raise RuntimeError("The exact hessian must be used.")
		if acados_ocp_solver.acados_ocp.solver_options.with_solution_sens_wrt_params is False:
			raise RuntimeError("Sensitivity evaluation must be enabled in acados")
		self.acados_ocp_solver = acados_ocp_solver
		self.nx = acados_ocp_solver.acados_ocp.model.x.shape[0]
		self.nu = acados_ocp_solver.acados_ocp.model.u.shape[0]
		self.np = acados_ocp_solver.acados_ocp.model.p.shape[0]

	def forward(self, x0: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
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
		def forward(x0, params):
			# x0 and params should be in batches
			# Up to now, we only consider setting the same p for all stages
			if not x0.shape[0] == params.shape[0]:
				raise ValueError('Batch size mismatch')

			batch_size = x0.shape[0]
			# TODO: pay attention to the data type
			u0 = torch.Tensor(batch_size, acados_ocp_solver.acados_ocp.dims.nu)
			for batch in range(batch_size):
				for stage in range(acados_ocp_solver.acados_ocp.dims.N):
					acados_ocp_solver.set(stage, 'p', params[batch, :].cpu().numpy())
				u_opt = acados_ocp_solver.solve_for_x0(x0[batch, :].cpu().numpy())
				# u0[batch, :] = torch.tensor(u_opt, dtype=torch.double)
				u0[batch, :] = torch.tensor(u_opt)
				if acados_ocp_solver.get_status() not in [0, 2]:
					breakpoint()
			return u0

		@staticmethod
		def setup_context(ctx, inputs, output):
			x0, _ = inputs
			batch_size = x0.shape[0]
			ctx.batch_size = batch_size
			ctx.nu = acados_ocp_solver.acados_ocp.dims.nu
			ctx.nx = acados_ocp_solver.acados_ocp.dims.nx
			ctx.np = acados_ocp_solver.acados_ocp.dims.np

		@staticmethod
		def backward(ctx, grad_output):
			# The output of backward() function must be the same dimension as input arguments of forward
			# Refer to: https://discuss.pytorch.org/t/a-question-about-autograd-function/201364
			batch_size = ctx.batch_size
			grad_u0_x0 = grad_u0_p = None

			if ctx.needs_input_grad[0]:
				# raise Warning("SHIT")
				grad_u0_x0 = torch.Tensor(batch_size, ctx.nx).type(grad_output.dtype)
				for batch in range(batch_size):
					_, dudx0 = acados_ocp_solver.eval_solution_sensitivity(0, "initial_state")
					grad_u0_x0[batch, :] = grad_output[batch] @ torch.tensor(dudx0, dtype=grad_output.dtype)
			if ctx.needs_input_grad[1]:
				grad_u0_p = torch.Tensor(batch_size, ctx.np).type_as(grad_output)
				for batch in range(batch_size):
					_, dudp = acados_ocp_solver.eval_solution_sensitivity(0, "params_global")
					grad_u0_p[batch, :] = grad_output[batch] @ torch.tensor(dudp, dtype=grad_output.dtype)

			return None, grad_u0_p
	
	return _AcadosOcpLayer.apply
