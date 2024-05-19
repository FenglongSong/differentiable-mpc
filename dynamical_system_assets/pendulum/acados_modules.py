import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver


dt = 0.01
N = 100
Tf = dt * N
# Q_mat = np.diag([1e3, 1e2])
R_mat = np.diag([1e-2])
target_state = np.array([0., 0.])
target_input = np.array([0.])
max_torque = 20.
min_torque = -20.

def create_acados_model() -> AcadosModel:
    # set up states and inputs
    theta = ca.SX.sym('theta')
    omega = ca.SX.sym('omega')
    x = ca.vertcat(theta, omega)

    tau = ca.SX.sym('tau')
    u = ca.vertcat(tau)

    q_theta = ca.SX.sym('q_theta')
    q_omega = ca.SX.sym('q_omega')
    p = ca.vertcat(q_theta, q_omega)

    # setup parameters
    l = 1.  # [m]
    m = 1.  # [kg]
    g = 9.81  # [m/s^2]

    # dynamics
    theta_dot = omega
    inertia = 1/3 * m * l * l
    omega_dot = (1/2*l*ca.sin(theta) * m*g + tau) / inertia
    x_dot = ca.vertcat(theta_dot, omega_dot)
    dynamics_fun = ca.Function('f', [x, u, p], [x_dot])

    f_expl = x_dot
    k1 = dynamics_fun(x, u, p)
    k2 = dynamics_fun(x + dt/2*k1, u, p)
    k3 = dynamics_fun(x + dt/2*k2, u, p)
    k4 = dynamics_fun(x + dt*k3, u, p)
    x_plus = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.disc_dyn_expr = x_plus
    model.x = x
    model.u = u
    model.p = p
    model.name = 'pendulum'
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
    Q_mat = ca.diag(model.p)
    ocp.model.cost_expr_ext_cost = 1 / 2 * ocp.model.x.T @ Q_mat @ ocp.model.x \
                                   + 1 / 2 * ocp.model.u.T @ R_mat @ ocp.model.u
    ocp.model.cost_expr_ext_cost_e = 0.

    # set constraints
    ocp.constraints.lbu = np.array([min_torque])
    ocp.constraints.ubu = np.array([max_torque])
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.array([100., 1.])

    ocp.constraints.x0 = np.array([0., 0.])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_value_sens_wrt_params = True

    # set prediction horizon
    ocp.solver_options.tf = Tf

    return ocp


def create_acados_ocp_solver():
    ocp = create_acados_ocp()
    ocp_solver = AcadosOcpSolver(ocp)
    return ocp_solver



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
        line, = plt.plot(t, X_true[:, i])
        if X_true_label is not None:
            line.set_label(X_true_label)

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        plt.grid()
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

    u0 = ocp_solver.solve_for_x0(np.array([np.pi, 1.]))
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

    plot_traj(np.linspace(0, Tf, N + 1), max_torque, simU, simX, latexify=False)


if __name__ == '__main__':
    main()