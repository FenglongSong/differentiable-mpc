import numpy as np
import casadi as ca
import sys
sys.path.append('./')
import torch
import torch.nn as nn
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import pdb

from utils.mpc.mpc_layer import AcadosMpcLayer

dt = 0.01
N = 100
Tf = dt * N
Q_mat = np.diag([1e2, 1e1])
Qf_mat = Q_mat
target_state = np.array([10., 0.])
target_input = np.array([0.])
max_force = 500.
min_force = -max_force


# define a simple double integrator
def create_acados_model() -> AcadosModel:
    # set up states & inputs
    pos = ca.SX.sym('pos', 1)
    vel = ca.SX.sym('vel', 1)
    x = ca.vertcat(pos, vel)
    
    force = ca.SX.sym('force', 1)
    u = ca.vertcat(force)
    
    r_diag = ca.SX.sym('r_diag')
    p = ca.vertcat(r_diag)
    
    # setup parameters
    m = 1.  # kg
    
    pos_dot = x[1]
    vel_dot = u / m  # only considered 1dim
    x_dot = ca.vertcat(pos_dot, vel_dot)
    
    dynamics_fun = ca.Function('f', [x, u, p], [x_dot])

    # dynamics
    f_expl = x_dot
    k1 = dynamics_fun(x, u, p)
    k2 = dynamics_fun(x + dt/2*k1, u, p)
    k3 = dynamics_fun(x + dt/2*k2, u, p)
    k4 = dynamics_fun(x + dt*k3, u, p)
    xplus = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.disc_dyn_expr = xplus
    model.x = x
    model.u = u
    model.p = p
    model.name = 'double_integrator'
    return model


def create_acados_ocp() -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = create_acados_model()
    ocp.model = model

    # set dimensions
    ocp.dims.N = N
    nx = model.x.rows()
    nu = model.u.rows()

    # set cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = 1 / 2 * (ocp.model.x - target_state).T @ Q_mat @ (ocp.model.x - target_state) \
                                   + 1 / 2 * (ocp.model.u - target_input).T @ model.p @ (ocp.model.u - target_input)
    # ocp.model.cost_expr_ext_cost_e = 1 / 2 * (ocp.model.x - target_state).T @ Qf_mat @ (ocp.model.x - target_state)
    ocp.model.cost_expr_ext_cost_e = 0.

    # set constraints
    ocp.constraints.lbu = np.array([min_force])
    ocp.constraints.ubu = np.array([max_force])
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.ones(ocp.model.p.shape[0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.qp_solver_ric_alg = 1  # ? not sure how to set this one
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_value_sens_wrt_params = True

    # set prediction horizon
    ocp.solver_options.tf = Tf

    return ocp


def create_acados_ocp_solver() -> AcadosOcpSolver:
    ocp = create_acados_ocp()
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return ocp_solver


class myNet(nn.Module):
	def __init__(
		self,
		acados_ocp_layer: AcadosMpcLayer,
		input_dim: int,
		hidden_dim: int = 32,
		num_hidden_layers: int = 2,
		act: nn.functional = nn.functional.relu,
	) -> None:

		super().__init__()

		self.input_dim = input_dim
		# Define the input layer
		self.input_layer = nn.Linear(input_dim, hidden_dim)

		# Define the hidden layers
		self.hidden_layers = nn.ModuleList()
		for _ in range(num_hidden_layers):
			self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

		# Define the layer that gives mpc parameters
		self.mpc_param_layer = nn.Linear(hidden_dim, acados_ocp_layer.np)

		# Define the differentiable mpc layer
		self.mpc_solver_layer = acados_ocp_layer

		# Define activation function
		self.activation = act

	def forward(self, y):
		x = y
		x = self.activation(self.input_layer(x))
		for hidden_layer in self.hidden_layers:
			x = self.activation(hidden_layer(x))
		mpc_params = torch.nn.functional.relu(self.mpc_param_layer(x)) + 1e-3
		# print(mpc_params[0])
		action = self.mpc_solver_layer(y, mpc_params)
		return action

batch_size = 1
X = torch.zeros(batch_size, 2)
Y = 80 * torch.ones(batch_size, 1)

ocp_solver = create_acados_ocp_solver()

ocp_solver.constraints_set(0, 'lbx', np.array([0., 0.]))
ocp_solver.constraints_set(0, 'ubx', np.array([0., 0.]))
ocp_solver.solve()
print("Optimal u is:")
print(ocp_solver.get(0, 'u'))

ocp_layer = AcadosMpcLayer(ocp_solver)

hidden_dim = 64
torch.manual_seed(0)
net = myNet(ocp_layer, 2, hidden_dim, 2)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for _ in range(120):
	opt.zero_grad()
	l = torch.nn.MSELoss()(net(X), Y)
	print(l.item(), net(X).mean())
	l.backward()
	opt.step()
