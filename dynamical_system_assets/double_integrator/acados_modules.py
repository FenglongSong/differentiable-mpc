import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver


dt = 0.01
N = 100
Tf = dt * N
Q_mat = np.diag([1e2, 1e1])
Qf_mat = Q_mat
target_state = np.array([0., 0.])
target_input = np.array([0.])
max_force = 1.
min_force = -max_force


# define a simple double integrator
def create_acados_model() -> AcadosModel:
    # set up states & inputs
    pos = ca.SX.sym('pos')
    vel = ca.SX.sym('vel')
    x = ca.vertcat(pos, vel)
    
    force = ca.SX.sym('force')
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
    ocp.parameter_values = 0.001*np.ones(ocp.model.p.shape[0])

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
    ocp_solver = AcadosOcpSolver(ocp)
    return ocp_solver


def create_acados_sim() -> AcadosSim:
    sim = AcadosSim()
    sim.model = create_acados_model()
    sim.solver_options.T = dt
    sim.parameter_values = np.array([1.])
    return sim


def create_acados_sim_solver() -> AcadosSimSolver:
    sim = create_acados_sim()
    return AcadosSimSolver(sim)


import matplotlib.pyplot as plt
def plot_traj(t, u_max, U, X_true, latexify=False, plt_show=True, X_true_label=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = t[N_sim-1]
    Ts = t[1] - t[0]

    plt.subplot(nx+1, 1, 1)
    line, = plt.step(t, np.append([U[0]], U))
    if X_true_label is not None:
        line.set_label(X_true_label)
    else:
        line.set_color('r')

    plt.ylabel('$u$')
    plt.xlabel('$t$')
    plt.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max, 1.2*u_max])
    plt.xlim(t[0], t[-1])
    plt.grid()

    states_lables = [r'$x$', r'$v$']

    for i in range(nx):
        plt.subplot(nx+1, 1, i+2)
        line, = plt.plot(t, X_true[:, i], label='true')
        if X_true_label is not None:
            line.set_label(X_true_label)

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.legend(loc=1)
        plt.xlim(t[0], t[-1])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_show:
        plt.show()


def main():
    ocp_solver = create_acados_ocp_solver()

    nx = ocp_solver.acados_ocp.model.x.rows()
    nu = ocp_solver.acados_ocp.model.u.rows()

    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))

    u0 = ocp_solver.solve_for_x0(np.array([1., 1.]))
    ocp_solver.print_statistics()  # encapsulates: stat = ocp_solver.get_stats("statistics")
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

    plot_traj(np.linspace(0, Tf, N + 1), max_force, simU, simX, latexify=False)

    sens_x_x0, sens_u_x0 = ocp_solver.eval_solution_sensitivity(0, "initial_state")
    print(sens_u_x0)
    sens_x_p, sens_u_p = ocp_solver.eval_solution_sensitivity(0, "params_global")
    print(sens_u_p)

    ocp_solver.set(0, 'p', np.array([2., 2.]))
    print(ocp_solver.acados_ocp.parameter_values)


if __name__ == '__main__':
    main()
