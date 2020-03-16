import numpy as np

class DHGrid:
    @staticmethod
    def CreateGrid(bw):
        n_grid = 2 * bw
        k = 0;
        points = np.empty([2, n_grid, n_grid])
        for i in range(n_grid):
            for j in range(n_grid):
                points[0, i, j] = (np.pi*(2*i+1))/(4*bw)
                points[1, i, j] = (2*np.pi*j)/(2*bw);
                k = k + 1;
        return points

    @staticmethod
    def ConvertGridToEuclidean(grid):
        cart_grid = np.zeros([3, grid.shape[1], grid.shape[2]])
        cart_grid[0,:,:] = np.multiply(np.sin(grid[0, :,:]), np.cos(grid[1,:,:]))
        cart_grid[1,:,:] = np.multiply(np.sin(grid[0, :, :]), np.sin(grid[1, :, :]))
        cart_grid[2,:,:] = np.cos(grid[0, :, :])
        return cart_grid

if __name__ == "__main__":
    grid = DHGrid.CreateGrid(50)
    print("DH grid: ", grid.shape)
