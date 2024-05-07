import pdb
import numpy as np
import matplotlib.pyplot as plt
from acados_template import latexify_plot
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import yaml
from typing import Dict
import sys
sys.path.append('./')

from dynamical_system_assets.dynamical_system_base import DynamicalSystemBase
from dynamical_system_assets.double_integrator import DoubleIntegrator


def create_acados_model(sys: DynamicalSystemBase, config: Dict) -> AcadosModel:
    # set up states & inputs
    x = sys.x
    u = sys.u
    p = sys.p

    # dynamics
    f_expl = sys.dynamics(x, u, p)
    dt = config['ocp']['solver_options']['tf'] / config['ocp']['solver_options']['N']
    k1 = sys.dynamics(x, u, p)
    k2 = sys.dynamics(x + dt/2*k1, u, p)
    k3 = sys.dynamics(x + dt/2*k2, u, p)
    k4 = sys.dynamics(x + dt*k3, u, p)
    xplus = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.disc_dyn_expr = xplus
    model.x = x
    model.u = u
    model.p = p
    model.name = config['model']['name']
    model.cost_y_expr = sys.nonlinear_reference(x, u, p)
    model.cost_expr_ext_cost = sys.external_cost(x, u, p)
    return model


def create_acados_sim(sys: DynamicalSystemBase, config: Dict) -> AcadosSim:
    sim = AcadosSim()
    sim.model = create_acados_model(sys, config)
    sim.solver_options.T = config['sim']['solver_options']['T']
    sim.parameter_values = np.array(config['sim']['parameter_values'])
    return sim


def create_acados_sim_solver(sys: DynamicalSystemBase, config: Dict) -> AcadosSimSolver:
    sim = create_acados_sim(sys, config)
    json_file = 'acados_ocp.json' if 'ocp_solver' in config and 'json_file' in config['ocp_solver'] else None
    generate = config['ocp_solver']['generate'] if 'ocp_solver' in config and 'generate' in config[
        'ocp_solver'] else True
    build = config['ocp_solver']['build'] if 'ocp_solver' in config and 'build' in config['ocp_solver'] else True
    sim_solver = AcadosSimSolver(sim, json_file=json_file, generate=generate, build=build)
    return sim_solver


def create_acados_ocp(sys: DynamicalSystemBase, config: Dict) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = create_acados_model(sys, config)
    ocp.model = model

    # set dimensions
    ocp.dims.N = config['ocp']['solver_options']['N']
    nx = model.x.rows()
    nu = model.u.rows()

    # set cost
    config_cost = config['ocp']['cost']
    ocp.cost.cost_type = config_cost['cost_type']
    if config_cost['cost_type'] == 'LINEAR_LS':
        ocp.cost.W = np.diag(np.concatenate((config_cost['Q_diag'], config_cost['R_diag'])))
        ocp.cost.Vx = np.eye(nx + nu, nx)
        ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
        ocp.cost.yref = np.concatenate((config_cost['x_ref'], config_cost['u_ref']))
    elif config_cost['cost_type'] == 'NONLINEAR_RLS':
        ocp.cost.W = np.diag(np.concatenate((config_cost['Q_diag'], config_cost['R_diag'])))
        ocp.cost.yref = np.concatenate((config_cost['x_ref'], config_cost['u_ref']))
    elif config_cost['cost_type'] == 'EXTERNAL':
        pass

    try:
        ocp.cost.cost_type_e = config_cost['cost_type_e']
        ocp.cost.W_e = config_cost['Qf_diag']
        ocp.cost.Vx_e = np.eye(nx, nx)
        ocp.cost.yref_e = config_cost['yref_e']
    except:
        pass

    # set constraints
    config_constraints = config['ocp']['constraints']
    try:
        ocp.constraints.lbu = np.array(config_constraints['lbu'], dtype=float)
        ocp.constraints.ubu = np.array(config_constraints['ubu'], dtype=float)
        ocp.constraints.idxbu = np.array(config_constraints['idxbu'], dtype=int)
    except:
        pass

    try:
        ocp.constraints.lbx = np.array(config_constraints['lbx'], dtype=float)
        ocp.constraints.ubx = np.array(config_constraints['ubx'], dtype=float)
        ocp.constraints.idxbx = np.array(config_constraints['idxbx'], dtype=int)
    except:
        pass

    try:
        ocp.constraints.x0 = np.array(config_constraints['x0'])
    except:
        ocp.constraints.x0 = np.zeros(nx)

    # set parameter values
    # pdb.set_trace()
    if ocp.model.p is not None:
        ocp.parameter_values = np.array(config['ocp']['parameter_values'])

    # set solver options
    config_solver_options = config['ocp']['solver_options']
    ocp.solver_options.qp_solver = config_solver_options['qp_solver']
    ocp.solver_options.hessian_approx = config_solver_options['hessian_approx']
    ocp.solver_options.integrator_type = config_solver_options['integrator_type']
    ocp.solver_options.sim_method_num_stages = config_solver_options['sim_method_num_stages']
    ocp.solver_options.nlp_solver_type = config_solver_options['nlp_solver_type']
    ocp.solver_options.nlp_solver_max_iter = config_solver_options['nlp_solver_max_iter']
    # ocp.solver_options.qp_solver_ric_alg = 1  # ? not sure how to set this one


    if config['ocp']['solver_options']['with_solution_sens_wrt_params']:
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        # ocp.model.cost_expr_ext_cost = 1 / 2 * (ocp.model.x - cfg.target_state).T @ cfg.Q_mat @ (
        #             ocp.model.x - cfg.target_state) \
        #                                + 1 / 2 * (ocp.model.u - cfg.target_input).T @ cfg.R_mat @ (
        #                                            ocp.model.u - cfg.target_input)
        # ocp.model.cost_expr_ext_cost_e = 1 / 2 * (ocp.model.x - cfg.target_state).T @ cfg.Qf_mat @ (
        #             ocp.model.x - cfg.target_state)

    # set prediction horizon
    ocp.solver_options.tf = config_solver_options['tf']

    ocp.make_consistent()
    return ocp


def create_acados_ocp_solver(sys: DynamicalSystemBase, config: Dict) -> AcadosOcpSolver:
    ocp = create_acados_ocp(sys, config)
    json_file = 'acados_ocp.json' if 'ocp_solver' in config and 'json_file' in config['ocp_solver'] else None
    generate = config['ocp_solver']['generate'] if 'ocp_solver' in config and 'generate' in config['ocp_solver'] else True
    build = config['ocp_solver']['build'] if 'ocp_solver' in config and 'build' in config['ocp_solver'] else True
    # ocp_solver = AcadosOcpSolver(ocp, json_file=json_file, generate=generate, build=build)
    ocp_solver = AcadosOcpSolver(ocp)
    return ocp_solver


def main():
    config_file = '/Users/fenglongsong/PycharmProjects/differentiable-mpc/dynamical_system_assets/config/double_integrator_config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
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
    main()
