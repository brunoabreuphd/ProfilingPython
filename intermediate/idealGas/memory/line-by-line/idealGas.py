import SimulationCell as SC
import ConfigurationSpace as CS
import Simulator as S

nParticles = 100
density = 1.0
nPartitions = 10
nSteps = 1000000
displacement = 2.0
checkPointInterval = 0.2

cell = SC.SimulationCell(nParticles, density, nPartitions)
space = CS.ConfigurationSpace(cell)
simulator = S.Simulator(cell, space)
simulator.run(nSteps, displacement, checkPointInterval)
