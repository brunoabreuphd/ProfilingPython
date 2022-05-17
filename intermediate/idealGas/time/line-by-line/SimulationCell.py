import numpy as np

class SimulationCell:
    """
    A class that holds a 1d, partitioned simulation cell.

    Input:
        - nParticles: the number of particles in the cell
        - density: the density of the cell
        - nPartitions: number of partitions in the cell

    Attributes:
        - nParticles
        - density
        - nPartitions
        - size: the size of the simulation cell
    """
    def __init__(self, nParticles, density, nPartitions):
        self.nParticles = nParticles
        self.density = density
        self.nPartitions = nPartitions

        cellSize = nParticles / density
        self.cellSize = cellSize
        
        partitionSize = cellSize / nPartitions 
        self.partitionSize = partitionSize
        self.pttLeft = np.linspace(0.0, cellSize, nPartitions, endpoint=False)
        self.pttRight = np.linspace(partitionSize, cellSize, nPartitions)