# This file includes all the necessary file definitions to establish a test environment.
# For preliminary agent testing

class Enviro:
	tankMaxLevel = 200.0
	tankMinLevel = 0.0
	costOfOperationPerHour = 1
	tankDircreteLevels = 4

	# Safe zone
	tankLowerLimit = tankMaxLevel / tankDircreteLevels
	tankUpperLimit = (tankMaxLevel / tankDircreteLevels) * \
		(tankDircreteLevels - 1)
	
	# Enviroment status
	currentTankLevel = 0
	pumpStatus = 0
	
	def __init__(self,tankLevel, pumpStatus):
		self.currentTankLevel = tankLevel
		self.pumpStatus = pumpStatus

	def inst_consumption(self, Time):
		if Time < 6:
			inst_consumption = 5/3*Time+2
		elif Time < 12:
			inst_consumption = -5/6*Time+17
		elif Time < 18:
			inst_consumption = 13/6*Time-19
		elif Time < 24:
			inst_consumption = -18/6*Time+54

		return inst_consumption

	def consumptionFunc(self, startTime, endTime):
		sum = 0.5*(endTime-startTime) * \
			(self.inst_consumption(startTime)+self.inst_consumption(endTime))

		return sum

	def updateWaterLevel(self, startTime, endTime, tankLevel, flow):
		A = 10
		self.tankLevel = tankLevel + \
			(flow-self.consumptionFunc(startTime, endTime))/A
	
	def cost(self, currentTankLevel, flow):
		# Discrete cost
		barrierCost = 0
		if currentTankLevel >= self.tankDircreteLevels:
			barrierCost = 1
		elif currentTankLevel <= 0:
			barrierCost = 1
		else:
			barrierCost = 0

		return self.costOfOperationPerHour * flow + barrierCost
	
	def getTankLevelDiscrete(self):
		# Compute and map out tank level to discrete states.
		singleLevelQuantiny = self.tankMaxLevel / self.tankDircreteLevels
		return int(self.currentTankLevel / singleLevelQuantiny)
	

# ----------------- Section of shame ------------------------

	# def cost(self, currentTankLevel,flow):

	# 	# Cost associated to the tank level
	# 	if currentTankLevel < self.tankLowerLimit:
	# 		barrierCost = (currentTankLevel-self.tankLowerLimit)**2
	# 	elif currentTankLevel > self.tankUpperLimit:
	# 		barrierCost = (currentTankLevel-self.tankUpperLimit)**2
	# 	else:
	# 		barrierCost = 0
	# 	return self.costOfOperationPerHour * flow + barrierCost