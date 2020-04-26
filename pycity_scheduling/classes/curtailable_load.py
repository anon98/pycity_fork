import warnings
import gurobipy as gurobi
import pycity_base.classes.demand.ElectricalDemand as ed
import numpy as np

from .electrical_entity import ElectricalEntity


class CurtailableLoad(ElectricalEntity, ed.ElectricalDemand):
    """
    Extension of pyCity_base class ElectricalDemand for scheduling purposes.
    """

    def __init__(self, environment, P_El_Nom, max_curtailment,
                 max_low=None, min_full=None):
        """Initialize a curtailable load.

        Parameters
        ----------
        environment : Environment
            Common Environment instance.
        P_El_Nom : float
            Nominal electric power in [kW].
        max_curtailment : float
            Maximal Curtailment of the load
        max_low : int, optional
            Maximum number of timesteps the curtailable load can stay under
            nominal load
        min_full : int, optional
            Minimum number of timesteps the curtailable load has to stay at
            nominal operation level when switching to the nominal operation
            level
        """
        shape = environment.timer.timestepsTotal
        super().__init__(environment, 0, np.zeros(shape))
        self._long_ID = "CUL_" + self._ID_string

        self.P_El_Nom = P_El_Nom
        self.max_curt = max_curtailment
        if max_low is not None or min_full is not None:
            assert max_low is not None
            assert min_full is not None
            assert min_full >= 1
            assert max_low >= 0
        self.max_low = max_low
        self.min_full = min_full
        self.P_El_Curt = self.P_El_Nom * self.max_curt
        self.new_var("P_State", dtype=np.bool, func=lambda t: self.P_El_vars[t].x > 0.99*P_El_Nom)
        self.constr_previous_state = []
        self.constr_previous = []
        self.constr_previous_start = None

    def populate_model(self, model, mode="convex"):
        """Add variables to Gurobi model

        Call parent's `populate_model` method and set variables upper bounds to
        the loadcurve and lower bounds to s`elf.P_El_Min`.

        Parameters
        ----------
        model : gurobi.Model
        mode : str, optional
            Specifies which set of constraints to use
            - `convex`  : Use linear constraints
            - `integer`  : Uses integer variables for max_low and min_full constraints if necessary
        """
        super(CurtailableLoad, self).populate_model(model, mode)

        if mode in ["convex", "integer"]:
            for t in self.op_time_vec:
                self.P_El_vars[t].lb = self.P_El_Curt
                self.P_El_vars[t].ub = self.P_El_Nom

            if self.max_low is None:
                # if max_low is not set the entity can choose P_State freely.
                # as a result no constraints are required
                pass
            elif self.max_low == 0:
                # if max_low is zero the P_State_vars would have to  always be one
                # this results in operation at always 100%.
                # the following bound is enough to represent this behaviour
                for t in self.op_time_vec:
                    self.P_El_vars[t].lb = self.P_El_Nom
            elif mode == "integer":
                # generate integer constraints for max_low min_full values

                # create binary variables representing the state if the device is operating at full level
                for t in self.op_time_vec:
                    self.P_State_vars.append(model.addVar(
                        vtype=gurobi.GRB.BINARY,
                        name="%s_Mode_at_t=%i"
                             % (self._long_ID, t + 1)
                    ))
                model.update()

                # coupling the state variable to the electrical variable
                # since operation infinitly close to 100% can be chosen by the entity to create a state
                # of zero, coupling in one direction is sufficient.
                for t in self.op_time_vec:
                    model.addConstr(
                        self.P_State_vars[t] * self.P_El_Nom <= self.P_El_vars[t]
                    )

                # creat constraints which can be used by update_model to take previous states into account.
                # update_schedule only needs to modify RHS which should be faster than deleting and creating
                # new constraints
                max_overlap = max(self.max_low, self.min_full - 1)
                max_overlap = min(max_overlap, self.op_horizon)  # cap overlap constraints at op_horizon
                for t in range(1, max_overlap + 1):
                    self.constr_previous_state.append(model.addConstr(
                        gurobi.quicksum(self.P_State_vars[:t]) >= -gurobi.GRB.INFINITY
                    ))

                # add constraints forcing the entity to operate at least once at 100% between every range
                # of max_low + 1 in the op_horizon
                for t in self.op_time_vec:
                    next_states = self.P_State_vars[t:t + self.max_low + 1]
                    assert 1 <= len(next_states) <= self.max_low + 1
                    if len(next_states) == self.max_low + 1:
                        model.addConstr(
                            gurobi.quicksum(next_states) >= 1
                        )

                # add constraints to operate at a minimum of min_full timestaps at 100% when switching
                # from the state 0 to the state 1
                if self.min_full > 1:
                    next_states = self.P_State_vars[1:self.min_full]
                    self.constr_previous_start = model.addConstr(
                        self.P_State_vars[0] * len(next_states)
                        - gurobi.quicksum(next_states) <=
                        gurobi.GRB.INFINITY  # self.P_State_vars[t-1] set via update_model
                    )
                    for t in self.op_time_vec[:-2]:
                        next_states = self.P_State_vars[t + 2: t + self.min_full + 1]
                        assert 1 <= len(next_states) <= self.min_full - 1
                        model.addConstr(
                            (self.P_State_vars[t + 1] - self.P_State_vars[t]) * len(next_states) <=
                            gurobi.quicksum(next_states)
                        )
            else:
                # generate relaxed constraints with max_low min_full values
                width = self.min_full + self.max_low
                for t in self.op_time_vec[:-width + 1]:
                    next_vars = self.P_El_vars[t:t + width]
                    assert len(next_vars) == width
                    model.addConstr(
                        gurobi.quicksum(next_vars) >=
                        self.P_El_Nom * self.min_full + self.P_El_Curt * self.max_low
                    )

                # creat constraints which can be used by update_model to take previous P_El values into
                # account. update_schedule only needs to modify RHS which should be faster than deleting
                # and creating new constraints
                max_overlap = min(self.max_low + self.min_full - 1, self.op_horizon)
                for overlap in range(0, max_overlap):
                    self.constr_previous.append(model.addConstr(
                        gurobi.quicksum(self.P_El_vars[:overlap + 1]) >= -gurobi.GRB.INFINITY
                    ))
        else:
            raise ValueError(
                "Mode %s is not implemented by CHP." % str(mode)
            )


    def update_model(self, model, mode="convex"):
        super(CurtailableLoad, self).update_model(model, mode)
        timestep = self.timer.currentTimestep

        # if the timestep is zero a perfect initial constraint is assumed.
        # this results in no constraints in integer mode
        if timestep != 0 and len(self.constr_previous_state) > 0:
            # reset all constraints which could have been previously been modified.
            for constr in self.constr_previous_state:
                constr.RHS = -gurobi.GRB.INFINITY
            if self.constr_previous_start is not None:
                if self.P_State_Schedule[timestep - 1]:
                    self.constr_previous_start.RHS = gurobi.GRB.INFINITY #len(next_states)
                else:
                    self.constr_previous_start.RHS = 0
            # if the device was operating at 100% in the previous timestep
            if self.P_State_Schedule[timestep - 1]:
                # count the last timesteps it was operating at 100%
                full_ts = 1
                while (timestep - full_ts - 1) >= 0 and self.P_State_Schedule[timestep - full_ts - 1]:
                    full_ts += 1
                if timestep - full_ts - 1 < 0:
                    # if the device was operating at 100% back until timestep 0,
                    # perfect initial state is assumed resulting in no constraints
                    pass
                else:
                    # calculate the remaining timesteps the device needs to operate
                    # at 100%
                    remaining_fulls = self.min_full - full_ts
                    if remaining_fulls <= 0:
                        # if device was operating longer than min_full at 100%,
                        # no constraints need to be created
                        pass
                    elif remaining_fulls > self.op_horizon:
                        # if remaining timesteps to operate at 100% are more than
                        # op_horizon, CL has to operate at 100% in entire op_horizon
                        self.constr_previous_state[-1].RHS = self.op_horizon
                    else:
                        # create constraints by modifying RHS
                        self.constr_previous_state[remaining_fulls - 1].RHS = remaining_fulls
            # if the device was not operating at 100% in the previous timestep
            else:
                # count the last timesteps it was operating under 100%
                low_ts = 1
                while (timestep - low_ts - 1) >= 0 and not self.P_State_Schedule[timestep - low_ts - 1]:
                    assert low_ts <= self.max_low
                    low_ts += 1
                # calculate the timesteps in which the device has to operate at 100% in
                overlap = self.max_low - low_ts + 1
                # create constraints by modifying RHS
                # if CL can remain under 100% for entire op_horizon, no constraint is created
                if self.op_horizon >= overlap:
                    self.constr_previous_state[overlap - 1].RHS = 1


        if len(self.constr_previous) > 0:
            # no resets are required, because previously modified RHSs will be modified
            # again
            width = self.min_full + self.max_low
            for overlap, constr in enumerate(self.constr_previous, start=1):
                # calculate the minimum required power in one width window
                required = [self.P_El_Curt] * self.max_low + [self.P_El_Nom] * self.min_full
                # calculate already consumed power
                start_t = timestep - (width - overlap)
                if start_t < 0:
                    # if window goes back after simu_horizon, P_El_Nom
                    # is assumed for timesteps before simu_horizon
                    required = required[:start_t]
                    start_t = 0
                already_done = self.P_El_Schedule[start_t:timestep]

                assert len(already_done) + overlap == len(required)
                required = sum(required)
                already_done = sum(already_done)
                # create constraints by modifying RHS
                constr.RHS = required - already_done
