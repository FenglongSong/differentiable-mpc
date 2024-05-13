from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import acados_template


class BaseMpcSolver(ABC):
    """
    Base class for MPC Solvers.
    """

    def __init__(self):
        self.nx = None
        self.nu = None
        self.np = None
        self.horizon = None
        self.status = -1
        pass

    @abstractmethod
    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Solve the MPC problem for the given initial states.
        :param x0: given initial states
        :return: the optimal input at current time step, optimal cost, solver works successfully or not.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_value_gradient(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_policy_gradient(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_value_function(self) -> float:
        pass


class AcadosMpcSolver(BaseMpcSolver):
    """
    Using acados as MPC Solvers.
    """
    def __init__(self, acados_ocp_solver: acados_template.AcadosOcpSolver):
        super().__init__()
        self.acados_ocp_solver = acados_ocp_solver
        self.nx = acados_ocp_solver.acados_ocp.dims.nx
        self.nu = acados_ocp_solver.acados_ocp.dims.nu
        self.np = acados_ocp_solver.acados_ocp.dims.np
        self.horizon = acados_ocp_solver.acados_ocp.dims.N

    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        opt_action = self.acados_ocp_solver.solve_for_x0(x0)
        opt_cost = self.acados_ocp_solver.get_cost()
        status = self.acados_ocp_solver.get_status()
        if status in [0, 2]:
            success = True
        else:
            success = False
        return opt_action, opt_cost, success

    def set_parameters(self, params: np.ndarray) -> None:
        # We only consider setting same parameters for all stages for now.
        for stage in range(self.horizon):
            self.acados_ocp_solver.set(stage, 'p', params)

    def get_policy_gradient(self) -> np.ndarray:
        sens_x_p, sens_u_p = self.acados_ocp_solver.eval_solution_sensitivity(0, "params_global")
        return sens_u_p

    def get_value_gradient(self) -> np.ndarray:
        sens_cost_p = self.acados_ocp_solver.eval_and_get_optimal_value_gradient("params_global")
        return sens_cost_p

    def get_value_function(self) -> float:
        return self.acados_ocp_solver.get_cost()
