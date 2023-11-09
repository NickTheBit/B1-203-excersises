# This file includes all the necessary file definitions to establish a test environment.
# For preliminary agent testing
import numpy as np

class Enviro:
	tankMaxLevel = 8.0
	tankMinLevel = 0.0
	costOfOperationPerHour = 1
	tankDircreteLevels = 8
	pumpFlowRate=10

	# Safe zone
	tankLowerLimit = tankMaxLevel / tankDircreteLevels
	tankUpperLimit = (tankMaxLevel / tankDircreteLevels) * \
		(tankDircreteLevels - 1)
	
	# Enviroment status
	currentTankLevel = 100
	pumpStatus = 0

	# Noise model
	stnd=10


	
	
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
			inst_consumption = -3*Time+70
		
		noise_samples=np.random.normal(0,self.stnd)
		inst_consumption=inst_consumption+noise_samples
		if inst_consumption <0:
			inst_consumption =0




		return inst_consumption

	def consumptionFunc(self, startTime, endTime):
		sum = 0.5*(endTime-startTime) * \
			(self.inst_consumption(startTime)+self.inst_consumption(endTime))

		return sum/5

	def updateWaterLevel(self, startTime, endTime, flow):
		A = 20

		# Flow introduced after endTime-startTime 
		flow=(endTime-startTime)*flow #Rectangular rule
		update = self.currentTankLevel +(flow-self.consumptionFunc(startTime, endTime))/A
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

		return flow*self.costOfOperationPerHour+barrierCost
	
	def getTankLevelDiscrete(self):
		# Compute and map out tank level to discrete states.
		singleLevelQuantiny = self.tankMaxLevel / self.tankDircreteLevels
		return int(self.currentTankLevel / singleLevelQuantiny)-1
	

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
