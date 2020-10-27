import numpy as np
from PCL.utils import *

class RealtimeCorrelativeScanMatching(object):
    """
    Real-time correlative scan matching algorithm class.

    Arg:
        source: source coordinates, numpy.array
        target: target coordinates, numpy.array
        resolution: integer
        search_win: search window, (meter, meter, degree)
        gird_size: length of each grid in high resolution lookup table, meter
        sensor_p: confidence of sensor, float, (0, 1)
        lidar_range: valid range of lidar data, meter
        factor: scale factor between adjacent maps

    Attributes:
         generate_maps: Rasterize the coordinates to highest map, and pooling it to different resolution of maps
         calc_log_likelihood: Calculate the log likelihood of the input coordinates on target map
         get_max_likelihood_in_win: Calculate the max log likelihood in given search window
         two_resolution_csm: Implementation of 2-resolution real-time correlative scan matching
         multi_resolution_csm: Implementation of multi-resolution real-time correlative scan matching
    """

    def __init__(self, source, target, grid_size, search_win, sensor_p,resolution=2, lidar_range=10, factor=10):
        self.source = source
        self.grid_size = grid_size
        self.p = sensor_p
        self.lidar_range = lidar_range
        self.factor = factor
        self.resolution = resolution
        self.search_win = search_win
        self.maps = []
        self.generate_maps(target)


    def generate_maps(self, coordinates):
        grid_coor = (coordinates / self.grid_size).astype(np.int)
        add = np.log(self.p / (1 - self.p))
        size = int(self.lidar_range / self.grid_size)
        map = np.zeros((size, size))
        for i in range(grid_coor.shape[0]):
            # lidar coordinate -> image coordinate
            x = size // 2 - grid_coor[i][1]
            y = size // 2 + grid_coor[i][0]
            if 0 <= x < size and 0 <= y < size:
                map[x][y] += add
        map = np.exp(map) / (1 + np.exp(map))
        self.maps.append(map)
        for _ in range(self.resolution-1):
            lower_map = []
            for i in range(0, map.shape[0], self.factor):
                line = []
                for j in range(0, map.shape[1], self.factor):
                    line.append(np.max(map[i:i + self.factor, j:j + self.factor]))
                lower_map.append(line)
            lower_map = np.array(lower_map)
            self.maps.append(lower_map)
            map = lower_map
        # inverse
        self.maps = self.maps[::-1]
        # for m in self.maps:
        #     print(m.shape)

    def calc_log_likelihood(self, approx, grid_map, grid_size):
        size = grid_map.shape[0]
        # if abs(grid_size*size - self.lidar_range) > 1e-5:
        #     raise RuntimeError("grid size and map size miss matching: ", grid_size, size, self.lidar_range)
        approx_coors = (approx//grid_size).astype(np.int)
        log_likelihood = 0
        for i in range(approx_coors.shape[0]):
            # lidar coordinate -> image coordinate
            x = size // 2 - approx_coors[i][1]
            y = size // 2 + approx_coors[i][0]
            if 0 <= x < size and 0 <= y < size:
                log_likelihood += grid_map[x][y]
        return log_likelihood

    def get_max_likelihood_in_win(self, coordinates, grid_map, search_win, grid_size):
        # if abs(grid_size*grid_map.shape[0] - self.lidar_range) > 1e-5:
        #     raise RuntimeError("grid size and map size miss matching: ", grid_size, grid_map.shape[0], self.lidar_range)
        delta_x = search_win[0] / 2
        delta_y = search_win[1] / 2
        delta_angle = search_win[2] / 2
        max_log_likelihood = 0
        best_trans = []
        best_coors = []
        for angle in np.arange(-1 * delta_angle, delta_angle, search_win[2]/self.factor, dtype=np.float):
            theta = angle / 180 * np.pi
            rotate_mat = np.mat([[np.cos(theta), np.sin(theta), 0],
                                  [-np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])
            rotated_coors = transform(coordinates, rotate_mat)
            for x in np.arange(-delta_x, delta_x+1e-3, grid_size, dtype=np.float):
                for y in np.arange(-delta_y, delta_y+1e-3, grid_size, dtype=np.float):
                    translate_mat = np.mat([[1, 0, x],
                                            [0, 1, y],
                                            [0, 0, 1]])
                    translated_coors = transform(rotated_coors, translate_mat)
                    log_likelihood = self.calc_log_likelihood(translated_coors, grid_map, grid_size)
                    # print(x, y, angle, log_likelihood)
                    if log_likelihood > max_log_likelihood:
                        max_log_likelihood = log_likelihood
                        best_trans = np.array([x,y,angle])
                        best_coors = translated_coors
                        # print(best_trans, log_likelihood)

        return max_log_likelihood, best_trans, best_coors


    def two_resolution_csm(self,lower_map, higher_map, search_win, factor):
        delta_x = search_win[0]/2
        delta_y = search_win[1]/2
        delta_angle = search_win[2]/2
        xy_step = self.grid_size*factor
        theta_step = search_win[2]/self.factor
        H_best = 0
        best_trans = []
        best_coors = []
        for angle in np.arange(-1*delta_angle, delta_angle, theta_step, dtype=np.float):
            theta = angle/180*np.pi
            rotate_mat = np.mat([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            rotate_coors = transform(self.source, rotate_mat)
            for x in np.arange(-1*delta_x, delta_x+1e-3, xy_step, dtype=np.float):
                for y in np.arange(-1*delta_y, delta_y+1e-3, xy_step, dtype=np.float):
                    translate_mat = np.mat([[1, 0, x],
                                            [0, 1, y],
                                            [0, 0, 1]])
                    translated_coors = transform(rotate_coors, translate_mat)
                    # in low resolution, when rasterizing the coordinates, the grid size should be scaled following the param 'res'
                    L_i = self.calc_log_likelihood(translated_coors, lower_map, factor*self.grid_size)
                    if L_i < H_best:
                        # print('L_i < H_best: ', x, y, angle)
                        continue
                    H_i, H_trans, H_coors = self.get_max_likelihood_in_win(translated_coors,
                                                                           higher_map,
                                                                           (factor*self.grid_size*2,
                                                                            factor*self.grid_size*2,
                                                                            theta_step*2),
                                                                           factor/self.factor*self.grid_size
                                                                           )
                    if H_i > H_best:
                        H_best = H_i
                        print(np.array([x, y, angle]), H_trans, H_best)
                        best_trans = H_trans + np.array([x, y, angle])
                        best_coors = H_coors
        return best_trans, best_coors

    def multi_resolution_csm(self):
        factor = self.factor**(self.resolution-1)
        search_win = self.search_win
        best_coors = self.source
        best_trans = np.zeros((3,))
        for i in range(0, self.resolution, 2):
            # brute force
            if i == self.resolution-1:
                higher_map = self.maps[i]
                T, best_coors = self.get_max_likelihood_in_win(best_coors, higher_map, search_win, self.grid_size)[1:]
            # multi-level resolution
            else:
                lower_map = self.maps[i]
                higher_map = self.maps[i+1]
                print(lower_map.shape, higher_map.shape)
                T, best_coors = self.two_resolution_csm(lower_map, higher_map, search_win, factor)

            best_trans += T
            factor/=self.factor
            search_win = search_win * (2**2) / (self.factor**2)
        return best_trans, best_coors



if __name__ == '__main__':
    laser = []
    with open('laser_data.txt', 'r') as f:
        laser = map(float, f.read().split(','))
    params = {}
    with open('param.txt', 'r') as f:
        for row in f.readlines():
            params[row.split(':')[0]] = float(row.split(':')[1])

    # row laser data to (x,y)
    coordinates = scan2coor(laser, params['angle_min'], params['angle_increment'])
    # generate target
    # test1 = np.array([[1,1] for _ in range(100)])
    # test2 = np.array([[3,-1.5] for _ in range(100)])
    # test = np.concatenate([test1, test2])
    target = generate_target_coordinates(coordinates, x=0.75,y=-0.83,angle=2)

    realtime_csm = RealtimeCorrelativeScanMatching(coordinates,
                                                   target,
                                                   resolution=1,
                                                   grid_size=0.05,
                                                   search_win=np.array([2,2,10], dtype=np.float),
                                                   factor=4, sensor_p=0.99, lidar_range=10)

    # visualize_map(map)
    target_map = realtime_csm.maps[-1]
    visualize_map(target_map)


    best_trans, best_coors = realtime_csm.multi_resolution_csm()

    visualize_map(rasterization(best_coors, grid_size=realtime_csm.grid_size, lidar_range=realtime_csm.lidar_range))
    print('end: ',best_trans)
