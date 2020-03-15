import numpy as np

class DHGrid:
    @staticmethod
    def createGrid(bw):
        n_grid = 2 * bw - 1
        k = 0;
        points = np.empty([n_grid * n_grid, 2])
        for i in range(n_grid):
            for j in range(n_grid):
                points[k, 0] = (np.pi*(2*i+1))/(4*bw)
                points[k, 1] = (2*np.pi*j)/(2*bw);
                k = k + 1;
        return points
if __name__ == "__main__":
    grid = DHGrid.createGrid(50)
    print("DH grid: ", grid.shape)
