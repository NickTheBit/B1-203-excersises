# This file includes all the necessary file definitions to establish a test environment.
# For preliminary agent testing

import numpy as np
import scipy.optimize as optimize

class Enviro:
	tankMaxLevel = 8.0
	tankMinLevel = 0.0
	costOfOperationPerHour = 1
	tankDircreteLevels = 8
	pumpFlowRate=6
	noiseStatus=0

	consumption_record=[]

	# Safe zone
	tankLowerLimit = tankMaxLevel / tankDircreteLevels
	tankUpperLimit = (tankMaxLevel / tankDircreteLevels) * \
		(tankDircreteLevels - 1)
	
	demandAtenuation=10
	
	# Enviroment status
	currentTankLevel = 100
	pumpStatus = 0

	# Import expected demand and expected std
	mean_h= np.array([[21.33726572],
       [18.3268918 ],
       [16.93748845],
       [18.88926934],
       [17.03673154],
       [32.61273064],
       [51.23640651],
       [63.56530467],
       [65.2972982 ],
       [60.03052152],
       [53.76088541],
       [53.55298658],
       [52.24595511],
       [51.16605613],
       [54.02167404],
       [55.18792935],
       [53.62179446],
       [49.39319854],
       [56.00209146],
       [48.59388232],
       [39.80549087],
       [37.34640026],
       [30.83563069],
       [25.74452874]])
	
	
	Stdd_h=np.array([[ 7.30575868],
       [ 7.60336177],
       [ 9.39440481],
       [ 8.28560809],
       [ 7.48416466],
       [17.79298767],
       [15.24356728],
       [12.89008299],
       [11.65732454],
       [10.97107084],
       [ 5.59203111],
       [ 8.85522173],
       [ 5.09258188],
       [ 9.4017683 ],
       [ 8.81812826],
       [20.74735851],
       [11.44272346],
       [19.62373481],
       [13.33540488],
       [17.47234512],
       [ 8.00550895],
       [12.81140972],
       [ 7.46738026],
       [ 9.12657279]])


	
	
	def __init__(self,tankLevel, pumpStatus,noiseStatus):
		self.currentTankLevel = tankLevel
		self.pumpStatus = pumpStatus
		self.noiseStatus= noiseStatus

	"""
	def inst_consumption(self, Time):
		if Time < 6:
			inst_consumption = 5/3*Time+2
		elif Time < 12:
			inst_consumption = -5/6*Time+17
		elif Time < 18:
			inst_consumption = 13/6*Time-19
		elif Time < 24:
			inst_consumption = -3*Time+70
		
		noise_samples=np.random.normal(0,self.stnd)
		inst_consumption=inst_consumption+noise_samples
		if inst_consumption <0:
			inst_consumption =0
	
		return inst_consumption
	"""
	def inst_consumption(self, Time):
		inst_consumption=np.random.normal(self.mean_h[Time,0],self.noiseStatus*self.Stdd_h[Time,0])
		# To avoid negative values
		if inst_consumption <0:
			inst_consumption=0
		
		self.consumption_record.append(inst_consumption/self.demandAtenuation)
		
		return inst_consumption


	def consumptionFunc(self, Time):
		deltaT=1
		sum = deltaT*self.inst_consumption(Time)

		return sum/self.demandAtenuation

	def updateWaterLevel(self, Time, flow):
		A = 20
		deltaT=1

		# Flow introduced after endTime-startTime 
		flow_=(deltaT)*flow #Rectangular rule
		update = self.currentTankLevel +(flow_-self.consumptionFunc(Time))/A
		if update <=0:
			self.currentTankLevel = 0
		elif update >= self.tankMaxLevel:
			self.currentTankLevel = self.tankMaxLevel
		else:
			self.currentTankLevel = update

			

	
	def cost(self, flow):
		# Discrete cost
		lowerLimit=2
		upperLimit=4
		barrierCost = 0
		if self.currentTankLevel >= upperLimit:
			barrierCost =(self.currentTankLevel-upperLimit)**2
		elif self.currentTankLevel <= lowerLimit:
			barrierCost = (lowerLimit-self.currentTankLevel)**2
		else:
			barrierCost = 0

		return flow*self.costOfOperationPerHour+self.pumpFlowRate*2*barrierCost
		
	def getTankLevelDiscreteVariable(self):
		
		steepnes=4 # Steepnes of the non-linear region (steepness of the sigmoid function)
		N_fine=7 # Number of discrete states in the region of fine discretization
		spread=1

		#Boundaries
		lowerBoundary=2
		higherBoundary=4

		# This is a pice wise continuous function from x=[0,Max_h], and consists of two linear regions 
		# and two sigmoid functions near the boundaries
		x=self.currentTankLevel

		# This is a pice wise continuous function from x=[0,Max_h], and consists of two linear regions 
		# and two sigmoid functions near the boundaries
		Z=2*N_fine+spread

		if x <=lowerBoundary-spread:
			y=x
		elif x <=lowerBoundary+spread:
			z = 1/(1 + np.exp(-steepnes*(x-lowerBoundary))) 
			y=N_fine*z+lowerBoundary-spread
        
		elif x <=higherBoundary+spread:
			z = 1/(1 + np.exp(-steepnes*(x-higherBoundary))) 
			y=N_fine*z+N_fine+spread
		else:
			y=x-(higherBoundary+spread)+Z
		return int(y)

	def getTankLevelDiscrete(self):
		# Compute and map out tank level to discrete states.
		singleLevelQuantiny = self.tankMaxLevel / self.tankDircreteLevels
		return int(self.currentTankLevel / singleLevelQuantiny)-1
	def RBF(self,center, variable, sigma):
		return np.exp((-(np.abs(center-variable))**2)/2*sigma)
	def RBFapprox(self, w, height):
		qestimate = []
		for h in range(len(height)):
			qsum = 0
			for j in range(19):
				qsum += w[j]*self.RBF((j*0.42)/2, height[h], 20)
			qestimate.append(qsum)
		return qestimate	
	def RBFerr(self, qval, w, time):
		hqestimate = 0
		tqestimate = 0
		# Jw = []
		for k in range(19):
			hqestimate += np.exp((-(np.abs((k*0.42)/2-self.currentTankLevel))**2)/2*20)
		for i in range(24):
			tqestimate += np.exp((-(np.abs((i)-time))**2)/2*25)
		err = qval - w*(hqestimate)
		#for i in range(100):
		#	Jw.append((qval - 0.01*i*qestimate)**2)
		# minJw = np.argmin(Jw)
		# the arrays are not the same size need to find a way to fix
		return err
	def sgd_update(self, w, learning_rate, error, s):
		return w + learning_rate * error * s  # Use element-wise multiplication for the error term


