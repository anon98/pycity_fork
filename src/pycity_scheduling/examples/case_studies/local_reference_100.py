from pycity_scheduling.algorithms import *
import pycity_scheduling.util.factory as factory
from pycity_scheduling.util.district_analyzer import DistrictAnalyzer
import numpy as np


def main():
    print("\n\n------ Evaluate Local N=100 ------\n\n")
    # First, create an environment using the factory's "generate_standard_environment" method. The environment
    # automatically encapsulates time, weather, and price data/information.
    env = factory.generate_standard_environment(initial_date=(2018, 1, 10), step_size=900, op_horizon=96)

    # Create N=100 single-family houses:
    num_sfh = 100

    # TABULA building stock distribution:
    sfh_distribution = {
        'SFH.1200': 0.033,
        'SFH.1860': 0.096,
        'SFH.1919': 0.112,
        'SFH.1949': 0.085,
        'SFH.1958': 0.150,
        'SFH.1969': 0.145,
        'SFH.1979': 0.070,
        'SFH.1984': 0.11,
        'SFH.1995': 0.102,
        'SFH.2002': 0.077,
        'SFH.2010': 0.01,
        'SFH.2016': 0.01,
    }

    # All households are equipped with electro-thermal heat pump units:
    sfh_heating_distribution = {
        'HP': 1.0,
        'BL': 0.0,
        'EH': 0.0,
        'CHP': 0.0,
    }

    # All households are equipped with a fixed load and PV unit.
    # Moreover, 50% of all households have a stationary battery storage unit and 30% have an electric vehicle.
    # The values are rounded in case they cannot be perfectly matched to the given number of buildings.
    sfh_device_probs = {
        'FL': 1.0,
        'DL': 0.0,
        'EV': 0.3,
        'BAT': 0.5,
        'PV': 1.0,
    }

    # Finally, create the desired city district using the factory's "generate_tabula_district" method. The district
    # operator's objective is defined as "peak-shaving" and the buildings' objectives are defined as
    # "self-consumption". Both objectives are quadratic and therefore the optimization problem becomes an MIQP.
    district = factory.generate_tabula_district(environment=env,
                                                number_sfh=num_sfh,
                                                number_mfh=0,
                                                sfh_building_distribution=sfh_distribution,
                                                sfh_heating_distribution=sfh_heating_distribution,
                                                sfh_device_probabilities=sfh_device_probs,
                                                district_objective='peak-shaving',
                                                building_objective='self-consumption',
                                                seed=0)

    # # Set the parameters for the local reference algorithm:
    mode = 'integer'

    # Perform the city district scheduling using the local reference algorithm:
    opt = LocalOptimization(city_district=district, mode=mode)
    results = opt.solve()
    print("Day-ahead local reference optimization completed.")

    # Print schedules:
    district_net_schedule = np.zeros(len(district.p_el_schedule))
    for bd in district.get_lower_entities():
        district_net_schedule = np.add(district_net_schedule, bd.p_el_schedule)
    print("System level schedule:")
    print(list(district_net_schedule))

    # Verify the feasibility of the solution:
    district.copy_schedule("district_schedule")
    d_a = DistrictAnalyzer(city_district=district)
    d_a.save_schedules()
    d_a.count_violated_constraints()

    # Calculate and print the objective value:
    obj_new = 0.0
    for t in range(district.op_horizon):
        obj_new += district_net_schedule[t] * district_net_schedule[t]  # peak-shaving
    for bd in district.get_lower_entities():
        for t in range(district.op_horizon):
            if bd.p_el_schedule[t] < 0.0:
                obj_new += bd.p_el_schedule[t] * bd.p_el_schedule[t]  # self-consumption
    print("Objective value: ", obj_new)
    return


if __name__ == '__main__':
    # Run the case study:
    main()
