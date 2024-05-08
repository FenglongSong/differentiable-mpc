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
        pass

    @abstractmethod
    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Solve the MPC problem for the given initial states.
        :param x0: given initial states
        :return: the optimal input at current time step, optimal cost, solver works successfully or not.
        """
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

    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        opt_action = self.acados_ocp_solver.solve_for_x0(x0)
        opt_cost = self.acados_ocp_solver.get_cost()
        status = self.acados_ocp_solver.get_status()
        if status in [0, 2]:
            success = True
        else:
            success = False
        return opt_action, opt_cost, success

