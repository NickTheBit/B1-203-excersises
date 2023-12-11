# Packages loading
import numpy as np
import matplotlib.pyplot as plt
import WaterNetwork as WN
import random as rd
from SupportFunctions import SupportFunctions
from SimulationConfig import SimulationConfig

# Visualization packages
from rich.progress import track


### Class defintions ####################################################################################################

class PumpingStationControl:
    sumAvg = 0
    sumCount = 0

    def __init__(self, total_iterations, conf):
        # Parameters
        self.hmin = 2.4  # [m] min level in the tank
        self.hmax = 3.2  # [m] max level in the tank
        self.num_of_pumps = 3  # Number of pumps in the pumping station (WHY WAS THIS A FLOAT???? DO YOU HAVE HALF PUMPS IN YOUR VILLAGE???)
        # State initialization
        self.num_running_pumps = 0  # Number of running pumps
        self.speed = 1.0  # [0-1] Speed of active pumps, I'll politely ignore that for now.
        self.discreteStates = conf.discrete_water_levels

        # todo: Investigate this + 1
        self.total_iterations = total_iterations + 1

        # For this iteration the controller makes a decision once per hour.
        self.Qtable = np.ones((24 + 1, self.discreteStates + 1, self.num_of_pumps + 1))
        # epsilon parameter
        self.epsilon = conf.epsilon  # this should be adjustable
        self.learning_rate = conf.learning_rate  # this should also do something.
        self.discount_factor = conf.discount_factor  # This is probably too high

        # Starting action
        self.currentAction = 1
        self.last_action = 1
        self.tank_last_state = 0
        self.last_time = 0

        # testing shit.
        self.states = []

        # Initializing Q-learning
        self.ql = SupportFunctions(self.hmax, self.hmin, self.discreteStates, self.learning_rate,
                                   self.discount_factor)

    def sim_step(self, water_level, current_time, step, total_steps, power_consumed):
        # We are updating the Q-values, unless the step = 0

        # Updating controller data
        tank_current_state = self.ql.get_tank_level_discrete(water_level)
        self.states.append(tank_current_state)

        # Take action a and observe s',r'
        self.num_running_pumps = self.currentAction

        # We penalize more if we get out of bounds
        cost = self.ql.cost(water_level, power_consumed)

        current_q_value = self.Qtable[self.last_time][self.tank_last_state][self.last_action]

        # Updating the Q - Value
        self.Qtable[self.last_time][self.tank_last_state][self.last_action] = self.ql.compute_q(current_q_value, cost,
                                                                                              np.argmin(self.Qtable[
                                                                                                            current_time][
                                                                                                            tank_current_state]))

        # Selecting next step
        optimal_action = np.argmin(self.Qtable[current_time][tank_current_state])

        # Setting epsilon depending on simulation progress
        # If statement is to avoid division by zero
        if step != 0:
            # self.epsilon = (step / total_steps)
            self.epsilon = np.log10(10 - (step / total_steps) * 20)

        # This is the epsilon-greedy algo.
        if tank_current_state <=0:
            self.currentAction = 3
        elif tank_current_state >=7:
            self.currentAction = 0 
        elif rd.random() > self.epsilon:
            # Optimal selection based on Q
            self.currentAction = optimal_action
        else:
            self.currentAction = rd.randint(0, self.num_of_pumps)
        
        self.tank_last_state =  tank_current_state
        self.last_action = self.currentAction
        self.last_time= current_time

        # Evaluation section
        if self.epsilon < 0.10:
            self.sumAvg += self.ql.cost(water_level, 0)
            self.sumCount += 1

    def get_outputs(self):
        return self.speed, self.num_running_pumps

    def get_performance(self):
        return self.sumAvg / self.sumCount

    def vis_q_table(self, index, simConfig):
        fig, ax = plt.subplots()

        averaged_data = np.mean(self.Qtable, axis=2)
        averaged_data = np.rot90(averaged_data)

        im = ax.imshow(averaged_data, cmap="viridis")

        if simConfig.saveGraphs:
            path = "figures/qTable{}.png".format(index)
            plt.savefig(path)
        else:
            plt.show()


def run_simulation(sim_config, sim_index=0):
    ### Simulation parameters ###
    print("Initializing the simulation.")
    # Simulation settings
    sim_time = sim_config.simulation_duration_days * 24.0 * 3600.0
    # Setup instance of the simulation
    water_network = WN.waterSupplyNetworkObject()
    sim_steps = int(sim_time / water_network.getSampTime())
    controller = PumpingStationControl(sim_steps, sim_config)

    ### Simulation ###
    print("Setting up memory.")
    time = np.zeros(sim_steps)
    level = np.zeros(sim_steps)
    demand1 = np.zeros(sim_steps)
    demand2 = np.zeros(sim_steps)
    pump_station_flow = np.zeros(sim_steps)
    pump_station_pressure = np.zeros(sim_steps)
    pump_station_power = np.zeros(sim_steps)
    pump_speed = np.zeros(sim_steps)
    num_of_running_pumps = np.zeros(sim_steps)

    # Each step has a duration of 15 minutes.
    for k in track(range(sim_steps), description="Simulation running..."):
        # Water supply network
        water_network.simStep(pump_speed[k], num_of_running_pumps[k])
        time[k], level[k], pump_station_flow[k], pump_station_pressure[k], pump_station_power[k], demand1[k], demand2[
            k] = water_network.getMeasurements()

        # Controller is only engaged every 60 minutes
        if k % 4 == 0 :
            controller.sim_step(level[k], int(k / 4) % 24, k, sim_steps, pump_station_power[k])

        if k < sim_steps - 1:
            pump_speed[k + 1], num_of_running_pumps[k + 1] = controller.get_outputs()

    if sim_config.performance_map:
        controller.vis_q_table(sim_index, sim_config)

    if sim_config.plot_events:
        ### Plot results ###
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(time, level, label='Tank Level')
        axs[0].legend()
        axs[0].grid()
        axs[0].set_ylabel("level [m]")
        # Plotting safe limits
        axs[0].axhline(y=controller.hmax, color='r', linestyle='-')
        axs[0].axhline(y=controller.hmin, color='r', linestyle='-')

        axs[1].plot(time, pump_station_flow, label='pump flow')
        axs[1].plot(time, demand1, label='demand flow 1')
        axs[1].plot(time, demand2, label='demand flow 2')
        axs[1].legend()
        axs[1].set_ylabel("flow [m3/h]")
        axs[2].plot(time, pump_station_power, label='pump speed')
        axs[2].legend()
        axs[2].set_ylabel("power [KW]")
        axs[2].set_xlabel("time [sec]")

        if sim_config.saveGraphs:
            path = "figures/plot{}.png".format(sim_index)
            plt.savefig(path)
        else:
            plt.show()

    return controller.get_performance()


if __name__ == "__main__":
    simulation_configuration = SimulationConfig()
    sim_results = []
    for sim in range(10):
        sim_results.append(run_simulation(simulation_configuration, sim))

    for result in range(len(sim_results)):
        print("Simulation: {}\t Performance: {}".format(result, sim_results[result]))