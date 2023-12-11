# Class allowing us to easily modify and pass parameters to our controller, changing it's behavior.

class SimulationConfig:
    # Simulation parameters
    simulation_duration_days = 300

    # Learning parameters
    epsilon = 0.1  # this should be adjustable
    learning_rate = 0.07  # this should also do something.
    discount_factor = 0.8  # This is probably too high

    # Controller Parameters
    discrete_water_levels = 7

    # Visualization parameters
    plot_events = True
    performance_map = False
    saveGraphs = True

    def __init__(self):
        pass
