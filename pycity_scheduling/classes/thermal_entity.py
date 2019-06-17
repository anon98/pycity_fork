import numpy as np
import gurobipy as gurobi

from .optimization_entity import OptimizationEntity


class ThermalEntity(OptimizationEntity):
    """
    Base class for all thermal entities derived from OptimizationEntity.

    This class provides functionalities common to all thermal entities.
    """

    def __init__(self, timer, *args, **kwargs):
        super(ThermalEntity, self).__init__(timer, *args, **kwargs)

        self.P_Th_vars = []
        self.P_Th_Schedule = np.zeros(self.simu_horizon)
        self.P_Th_Ref_Schedule = np.zeros(self.simu_horizon)

    def populate_model(self, model, mode=""):
        """Add variables to Gurobi model.

        Add variables for the thermal demand of the entity to the optimization
        model.

        Parameters
        ----------
        model : gurobi.Model
        mode : str, optional
        """
        self.P_Th_vars = []
        for t in self.op_time_vec:
            self.P_Th_vars.append(
                model.addVar(
                    name="%s_P_Th_at_t=%i" % (self._long_ID, t+1)
                )
            )
        model.update()

    def update_schedule(self):
        """Update the schedule with the scheduling model solution."""
        t1 = self.timer.currentTimestep
        t2 = t1 + self.op_horizon
        self.P_Th_Schedule[t1:t2] = [var.x for var in self.P_Th_vars]

    def save_ref_schedule(self):
        """Save the schedule of the current reference scheduling."""
        np.copyto(
            self.P_Th_Ref_Schedule,
            self.P_Th_Schedule
        )

    def reset(self, schedule=True, reference=False):
        """Reset entity for new simulation.

        Parameters
        ----------
        schedule : bool, optional
            Specify if to reset schedule.
        reference : bool, optional
            Specify if to reset reference schedule.
        """
        if schedule:
            self.P_Th_Schedule.fill(0)
        if reference:
            self.P_Th_Ref_Schedule.fill(0)
