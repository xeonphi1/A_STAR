import cv2
import numpy as np
import os
import math
import heapq
import sys
import timeit

from scripts import plotting


class AStar:
    """AStar set the cost + heuristics as the priority
    """

    def __init__(self, grid_map, s_start, s_goal, heuristic_type, draw=False):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.grid_map = grid_map
        self.move_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                         (1, 0), (1, -1), (0, -1), (-1, -1)]  # feasible input set

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come # prediction

        # For debugging and visualization
        self.canvas = self.grid_map.copy()
        cv2.circle(self.canvas, s_goal, 3, 150)
        self.draw = draw

    @staticmethod
    def show_im(im, name):
        im = cv2.resize(im, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(name, im)
        key = cv2.waitKey(1)
        if key == ord('s'):
            pass

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = np.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            if self.draw:
                cv2.circle(self.canvas, s, 1, 150, -1)
                self.show_im(self.canvas, "canvas")
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = np.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
        return self.extract_path(self.PARENT), self.CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.move_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return np.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if self.grid_map[s_start[1], s_start[0]] > 0 or self.grid_map[s_end[1], s_end[0]] > 0:
            return True

        # Check if it is not a horizontal or vertical line
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:

            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            # Check if any of the corners are solid, EXAMPLE:
            ###########
            # SS # S1 #
            ###########
            # S2 # SE #
            ###########
            if self.grid_map[s1[1], s1[0]] > 0 or self.grid_map[s2[1], s2[0]] > 0:
                return True
        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    # The map resolution [m/cell]
    res = 0.05
    robot_size = 1. / res  # pixels
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    im_path = ROOT_DIR + "/../im/map.png"
    # We load map and remove uncertainty
    grid_map = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    grid_map[np.where(grid_map <= 1)] = 0.
    grid_map[np.where(grid_map > 0)] = 255

    # since the robot will be seen as one pixel we need to inflate all the pixels to compensate for the robot size:
    # 1. Find all solid pixels:
    indices = np.where(grid_map == 255)
    solid_pixels = zip(indices[1], indices[0])  # X,Y
    # 2. inflate using circles:
    inflated_map = grid_map.copy()
    for pixel in solid_pixels:
        inflated_map = cv2.circle(inflated_map, pixel, np.ceil(robot_size / 2).astype(int), 255, -1)
    # grid_map_resized = cv2.resize(grid_map_resized, (0, 0), fx=res, fy=res)
    scale = 1
    s_start = (int(0* scale), int(0 * scale))
    s_goal = (int(100 * scale), int(100 * scale))

    # Check if our points are impossible:
    if inflated_map[s_start[1], s_start[0]] > 0:
        print("Start point is a collion")
    if inflated_map[s_goal[1], s_goal[0]] > 0:
        print("End point is a collion")

    tmp = inflated_map.copy()

    tmp = cv2.circle(tmp, s_start, 3, 150, -1)
    tmp = cv2.circle(tmp, s_goal, 3, 150)

    astar = AStar(inflated_map, s_start, s_goal, "euclidean")
    plot = plotting.Plotting(inflated_map, s_start, s_goal)
    start_time = timeit.default_timer()
    path, visited = astar.searching()
    print("found in:", timeit.default_timer() - start_time)
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
