# Here we define all the functions needed for q-learning to function

class SupportFunctions:
    def __init__(self, new_tank_max_level, new_tank_min_level, new_tank_discrete_levels, new_learning_rate, new_discount_factor) -> None:
        self.tankMaxLevel = new_tank_max_level
        self.tankMinLevel = new_tank_min_level
        self.tankDiscreteLevels = new_tank_discrete_levels
        self.singleLevelQuantity = self.tankMaxLevel / self.tankDiscreteLevels
        self.state_upper_limit = new_tank_discrete_levels
        self.learning_rate = new_learning_rate
        self.discount_factor = new_discount_factor
    
	

    def get_tank_level_discrete(self, new_current_tank_level):
        # Compute and map out tank level to discrete states.
        return_value = int(new_current_tank_level / self.singleLevelQuantity)

        # Since no soft limit, the tank will go negative, this patches an underlying problemo
        if return_value < 0:
            return_value = 0

        if return_value > 7:
            return_value = self.state_upper_limit

        return return_value

    def cost(self, current_tank_level, power_consumed):
        # Extreme punishment for exceeding legal limits
        level_cost = 0
        if current_tank_level > self.tankMaxLevel:
            level_cost = (current_tank_level - self.tankMaxLevel) ** 2+1
        elif current_tank_level < self.tankMinLevel:
            level_cost = (current_tank_level - self.tankMinLevel) ** 2+1

        return power_consumed + level_cost

    # This needs to account for cost of operation, but I don't give a shit right now

    def compute_q(self, current_value, cost, estimate_optimal_future_value):
        reward = cost
        return (1 - self.learning_rate) * current_value + self.learning_rate * (
                    reward + self.discount_factor * estimate_optimal_future_value-current_value)
