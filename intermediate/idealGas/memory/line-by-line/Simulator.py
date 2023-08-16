import numpy as np

class Simulator:
    def __init__(self, simulationCell, configurationSpace):
        self.simulationCell = simulationCell
        self.configurationSpace = configurationSpace
        self.lastAcceptanceRatio = 0.0

    @profile	
    def run(self, nSteps, trialDisplacement=1.0, checkPointInterval=0.25):
        """
        Proposes a number of displacements, accepting it if particles don't leave the cell. 

        Input:
            - nSteps: number of steps to attempt (int)
            - trialDisplacement: amplitude of displacement (float)
            - checkPointInterval: how often occupancy will be checked (float)
        """
        acc = 0
        for i in range(nSteps):
            luckyParticle = np.random.randint(0, self.simulationCell.nParticles)
            randomNumber = np.random.rand()
            newPosition = (2.0*randomNumber - 1)*trialDisplacement + self.configurationSpace.positions[luckyParticle]
            if (newPosition > 0.0 and newPosition < self.simulationCell.cellSize):
                self.configurationSpace.positions[luckyParticle] = newPosition
                acc = acc + 1

            if (i % int(nSteps*checkPointInterval) == 0):
                print(f"Step {i}")
                self.configurationSpace.check_occupancy()
        print(f"Step {i}")
        self.configurationSpace.check_occupancy()
        self.lastAcceptanceRatio = acc / nSteps
        
        return
