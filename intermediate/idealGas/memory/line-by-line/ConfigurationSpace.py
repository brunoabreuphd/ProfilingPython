import numpy as np

class ConfigurationSpace:
    """
    Contains the positions of all particles in a simulation cell. At start, it distributes all the particles into the first partition of the cell.

    Input:
        - simulationCell: A SmulationCell instance
    
    Attributes:
        - positions: numpy.array with the positions of each particle

    Methods:
        - check_occupancy(): verifies the occupancy of each partition
    """

    def __init__(self, simulationCell):
        self.simulationCell = simulationCell

        positions = np.random.rand(simulationCell.nParticles)
        positions = positions * simulationCell.partitionSize
        self.positions = positions

    def check_occupancy(self):
        """
        Verify the particle occupancy of each partition in the simulation cell.
        """
        occupancy = np.zeros(self.simulationCell.nPartitions)
        for i in range(self.simulationCell.nPartitions):
            for j in range(self.simulationCell.nParticles):
                if (self.positions[j] > self.simulationCell.pttLeft[i] and self.positions[j] <= self.simulationCell.pttRight[i]):
                    occupancy[i] = occupancy[i] + 1
        print("Check occupancy:")
        for i in range(self.simulationCell.nPartitions):
            print(f"Partition {i} has {occupancy[i]} particles")

        return