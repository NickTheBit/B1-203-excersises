### Packages loading
import waterSupplyNetwork


### Class defintions ####################################################################################################

# System object
class waterSupplyNetworkObject:

    def __init__(self):
        self.handle = waterSupplyNetwork.initialize()
        Qdesign = 80
        number_of_pumps = 3
        self.networkParam = {
            'area'            : 240.0,                   # [m^2] tank area
            'Dz'              :  30.0,                   # [m] elevation of the tank
            'r1'              : 5.0/(Qdesign*Qdesign),   # [m/(m^3/h)] network resistance
            'r2'              : 5.0/(Qdesign*Qdesign),   # [m/(m^3/h)] network resistance
            'number_of_pumps' : number_of_pumps,         # [-] number of pumpes at the pumping station
            'pumppar_a' : 0.0352,                        # [-] pump pressure parameter 1
            'pumppar_b' : 75.0,                          # [-] pump pressure parameter 2
            'pumppowerpar_a' : -0.0021,                  # [-] pump power parameter 1
            'pumppowerpar_b' :  0.2222,                  # [-] pump power parameter 2
            'pumppowerpar_c' :  4.4444                   # [-] pump power parameter 3
        }
        ctrlSettings = {
            'hmin' : 2.4,           # [m] min level in the tank
            'hmax' : 3.2            # [m] max level in the tank
        }
        self.dt   = 15.0*60.0       # [sec] sample time for the simulation
        self.time = 0.0             # [sec] start time for the simulation
        self.networkStates = self.handle.init_network(self.networkParam,ctrlSettings)
        
                        
    def getSampTime(self):
        return self.dt


    def simStep(self,pump_speed,num_of_operating_pumps):
        ctrl = {
            'speed'             : pump_speed,             # [0-1] speed of the pumps in operation
            'num_running_pumps' : num_of_operating_pumps  # [-] Number of running pumps
        }
        self.networkStates = self.handle.run_network(self.networkStates,ctrl,self.networkParam,self.time,self.dt)
        self.time = self.time + self.dt
        
    def getMeasurements(self):
        return self.time, self.networkStates['level'], self.networkStates['pump_flow'], self.networkStates['pump_pressure'], self.networkStates['pump_power'], self.networkStates['demand1'], self.networkStates['demand2']    
        
        