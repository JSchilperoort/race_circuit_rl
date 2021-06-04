import gym
from gym.envs.classic_control import rendering
import numpy as np
import time

import parameters


class RacingEnv(gym.Env):
    def __init__(self, n_steps=parameters.N_STEPS, max_speed=parameters.MAX_SPEED, max_ray_dist=parameters.MAX_RAY_DIST,
                 checkpoints_in_view=parameters.CHECKPOINTS_IN_VIEW, dead_reward=parameters.DEAD_REWARD):

        self.viewer = None

        # continuous action- and state-space normalized to [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.n_steps = n_steps
        self.maxSpeed = max_speed
        self.max_ray_dist = max_ray_dist
        self.checkpoints_in_view = checkpoints_in_view
        self.dead_reward = dead_reward

        self.accel = 0.05
        self.turnRate = 2.0
        self.minTurnRadius = 75

        self.make_track()

        self.printtime = False

        # store previous state checkpoint/ray intersections to speed up state retrieval
        self.prev_state_cp = [-1] * 5
        self.prev_state_lr = [''] * 5

        self.reset()

    def reset_prev_state(self):
        self.prev_state_cp = [-1] * 5
        self.prev_state_lr = [''] * 5

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps

    def get_action_space(self):
        return self.action_space

    def translate(self, v, angle, dist):
        """polar to cartesian offset"""

        angle = angle / 180 * np.pi
        x, y = v
        x += np.cos(angle) * dist
        y += np.sin(angle) * dist
        return x, y

    def make_track(self):
        """construct predefined racetrack from segments"""

        v = (0, 0)
        self.lwall = [(0, 100)]
        self.rwall = [(0, -100)]

        # angles to create an oval-shaped circuit
        angles = [0, 0, 0, 0, 0, 0, 0, 0,

                  10, 20, 30, 40, 50, 60, 70, 80, 90, 90,
                  90, 90, 100, 110, 120, 130, 140, 150, 160, 170,

                  180, 180, 180, 180, 180, 180, 180, 180, 180, 180,

                  165, 150, 135, 120, 105, 90,
                  105, 120, 135, 150, 165, 180,
                  195, 210, 225, 240, 255, 270,
                  255, 240, 225, 210, 195, 180,

                  180, 180, 180, 180, 180, 180, 180, 180, 180, 180,

                  190, 200, 210, 220, 230, 240, 250, 260, 270, 270,
                  270, 270, 280, 290, 300, 310, 320, 330, 340, 350,

                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                  15, 30, 45, 60, 75, 90,
                  75, 60, 45, 30, 15, 0,
                  345, 330, 315, 300, 285, 270,
                  285, 300, 315, 330, 345, 0]
        for angle in angles:
            v = self.translate(v, angle, 200)
            self.lwall.append(self.translate(v, angle + 90, 100))
            self.rwall.append(self.translate(v, angle - 90, 100))

    def magnitude(self, vector):
        return np.sqrt(np.dot(np.array(vector), np.array(vector)))

    def norm(self, vector):
        return np.array(vector) / self.magnitude(np.array(vector))

    def lineRayIntersectionDist(self, origin, direction, dist, point1, point2):
        """return intersection between wall and sensor from https://gist.github.com/danieljfarrell/faf7c4cafd683db13cbc"""

        rayOrigin = np.array(origin, dtype=np.float)
        rayDirection = np.array(self.norm(self.translate((0, 0), direction, 1)), dtype=np.float)
        point1 = np.array(point1, dtype=np.float)
        point2 = np.array(point2, dtype=np.float)
        v1 = rayOrigin - point1
        v2 = point2 - point1
        v3 = np.array([-rayDirection[1], rayDirection[0]])

        dotv2v3 = np.dot(v2, v3) if np.dot(v2, v3) else -0.00000001
        t1 = np.cross(v2, v1) / dotv2v3
        t2 = np.dot(v1, v3) / dotv2v3

        if t1 < 0.0 or t1 > dist or t2 < 0.0 or t2 > 1.0:
            return dist
        return t1

    def rayDist(self, direction, prev_cp_idx):
        """get distance of intersection of ray emitted from car with racetrack"""

        dist = self.max_ray_dist
        d = dist
        i_range = np.arange(self.checkpoint, self.checkpoint + self.checkpoints_in_view)

        if self.prev_state_lr[prev_cp_idx] == 'r' or self.prev_state_lr[prev_cp_idx] == '':
            if self.prev_state_cp[prev_cp_idx] in i_range:
                prev_i = np.where(i_range == self.prev_state_cp[prev_cp_idx])[0][0]
                i_range = i_range[parameters.SORTKEYSRIGHT[prev_i]]

            for i in range(self.checkpoints_in_view):
                i1 = i_range[i]
                i2 = i1 + 1
                if i1 >= len(self.lwall):
                    i1 -= len(self.lwall)
                if i2 >= len(self.lwall):
                    i2 -= len(self.lwall)

                v0 = self.rwall[i1]
                v1 = self.rwall[i2]
                d = self.lineRayIntersectionDist([self.x, self.y], direction, dist, v0, v1)
                if d < dist:
                    self.prev_state_cp[prev_cp_idx] = i1
                    self.prev_state_lr[prev_cp_idx] = 'r'
                    return d

        if self.prev_state_lr[prev_cp_idx] == 'l' or self.prev_state_lr[prev_cp_idx] == '':
            if self.prev_state_cp[prev_cp_idx] in i_range:
                prev_i = np.where(i_range == self.prev_state_cp[prev_cp_idx])[0][0]
                i_range = i_range[parameters.SORTKEYSLEFT[prev_i]]
            for i in range(self.checkpoints_in_view):
                i1 = i_range[i]
                i2 = i1 + 1
                if i1 >= len(self.lwall):
                    i1 -= len(self.lwall)
                if i2 >= len(self.lwall):
                    i2 -= len(self.lwall)

                v0 = self.lwall[i1]
                v1 = self.lwall[i2]
                d = self.lineRayIntersectionDist([self.x, self.y], direction, dist, v0, v1)
                if d < dist:
                    self.prev_state_cp[prev_cp_idx] = i1
                    self.prev_state_lr[prev_cp_idx] = 'l'
                    return d
        self.prev_state_cp[prev_cp_idx] = -1
        self.prev_state_lr[prev_cp_idx] = ''
        return d

    def getCheckpoint(self, i):
        """return one of the polygons that make up the racetrack"""

        if i < 0: i = 0

        if i >= len(self.lwall):
            i -= len(self.lwall)
        i2 = i + 1
        if i2 >= len(self.lwall):
            i2 -= len(self.lwall)

        return [self.lwall[i], self.lwall[i2], self.rwall[i2], self.rwall[i]]

    def pointInsidePolygon(self, x, y, poly):
        """from http://www.ariel.com.au/a/python-point-int-poly.html"""

        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def midpoint(self, p):
        """center of checkpoint polygon"""

        x = (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4
        y = (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4
        return x, y

    def checkpointAngle(self, i):
        """angle from a checkpoint to the next one"""

        c0 = self.midpoint(self.getCheckpoint(i))
        c1 = self.midpoint(self.getCheckpoint(i + 1))
        xdiff = c1[0] - c0[0]
        ydiff = c1[1] - c0[1]
        return np.arctan2(ydiff, xdiff)

    def step(self, action):
        """perform a single step in the environment"""

        assert self.action_space.contains(action)

        self.forward, self.steer = action

        # simplistic acceleration
        targetSpeed = (self.forward / 2 + 0.5) * self.maxSpeed
        self.speed += (targetSpeed - self.speed) * self.accel

        # steering dependent on speed
        self.angle -= self.steer * self.speed / (
                max(self.speed * self.speed / self.turnRate, self.minTurnRadius) * 2 * np.pi) * 2 * np.pi

        # update car position
        self.x += np.sin(self.angle) * self.speed
        self.y += np.cos(self.angle) * self.speed

        # reward is proportional to speed
        reward = self.speed / self.maxSpeed
        # reward = self.speed / 50

        # checkpoint and collision detection
        insideCurrent = self.pointInsidePolygon(self.x, self.y, self.getCheckpoint(self.checkpoint))
        dead = False
        if not insideCurrent:
            insideNext = self.pointInsidePolygon(self.x, self.y, self.getCheckpoint(self.checkpoint + 1))
            if insideNext:
                # car has entered new checkpoint
                self.checkpoint = self.checkpoint + 1
                if self.checkpoint >= (len(self.lwall)):
                    self.checkpoint = 0
            else:
                # car has left the racetrack
                # respawn car at last checkpoint and subtract reward
                if self.checkpoint > 0:
                    self.checkpoint -= 1
                self.x, self.y = self.midpoint(self.getCheckpoint(self.checkpoint))
                self.angle = -self.checkpointAngle(self.checkpoint) + np.pi / 2
                self.speed = 0
                reward += self.dead_reward
                self.reset_prev_state()
                dead = True

        # end episode after n steps
        self.t += 1
        done = False
        if self.t >= self.n_steps:
            done = True
        return np.array(self.get_state()), reward, done, dead

    def get_state(self):
        """state consists of distance sensor readings and current speed"""

        s = list()
        # 5 distance readings are taken from a range of -45 to 45 degrees in front of the car
        for i in range(-2, 3):
            direction = i * 45 / 2 + 90 - self.angle * 180 / np.pi
            s.append(self.rayDist(direction, i + 2) / self.max_ray_dist)
        s.append(self.speed / self.maxSpeed)
        return s

    def reset(self):
        """give car a random initial checkpoint with some noise in its angle"""

        self.checkpoint = np.random.randint(0, len(self.lwall) - 1)
        self.x, self.y = self.midpoint(self.getCheckpoint(self.checkpoint))
        angle_noise = ((np.random.rand() - 0.5) * 10) / 180 * np.pi
        self.angle = -self.checkpointAngle(self.checkpoint) + np.pi / 2 + angle_noise
        self.speed = 0
        self.t = 0
        self.reset_prev_state()
        return self.get_state()

    def rotate(self, xy, theta):
        """https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions"""

        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        return xy[0] * cos_theta - xy[1] * sin_theta,xy[0] * sin_theta + xy[1] * cos_theta

    def get_polygon(self):
        """return car rectangle"""

        points = [
            (-20, -30),
            (-20, 30),
            (20, 30),
            (20, -30)
        ]
        points = [self.rotate(xy, -self.angle) for xy in points]
        polygon = [
            (points[0][0] + 640 + self.x, points[0][1] + 100 + self.y),
            (points[1][0] + 640 + self.x, points[1][1] + 100 + self.y),
            (points[2][0] + 640 + self.x, points[2][1] + 100 + self.y),
            (points[3][0] + 640 + self.x, points[3][1] + 100 + self.y)
        ]
        return polygon


    def get_bounds(self):
        all_hor = [x[0] for x in self.lwall] + [x[0] for x in self.rwall]
        all_vert = [x[1] for x in self.lwall] + [x[1] for x in self.rwall]

        left_bound = min(all_hor) - 100
        right_bound = max(all_hor) + 1000
        lower_bound = min(all_vert) - 100
        upper_bound = max(all_vert) + 400

        width_scale = (right_bound - left_bound) / 1280
        height_scale = (upper_bound - lower_bound) / 720

        if width_scale > height_scale:
            middle_point = (upper_bound + lower_bound) / 2
            lower_bound = middle_point - (360 * width_scale)
            upper_bound = middle_point + (360 * width_scale)
        elif height_scale > width_scale:
            middle_point = (right_bound + left_bound) / 2
            left_bound = middle_point - (640 * height_scale)
            right_bound = middle_point + (640 * height_scale)
        return left_bound, right_bound, lower_bound, upper_bound

    def render(self):
        """display environment"""

        time.sleep(1.0 / 60)

        # initialize renderer
        if self.viewer == None:
            self.viewer = rendering.Viewer(1280, 720)
            self.transform_translation = rendering.Transform()
            self.transform_rotation = rendering.Transform()
            self.transform_rotation.set_translation(640, 100)

        # define racetrack as black lines
        self.transform_translation.set_translation(640, 100)

        lwall = rendering.PolyLine(self.lwall, False)
        lwall.add_attr(self.transform_translation)
        self.viewer.add_onetime(lwall)

        l_piece = rendering.PolyLine([self.lwall[-1], self.lwall[0]], False)
        l_piece.set_color(0.8, 0.8, 0.8)
        l_piece.add_attr(self.transform_translation)
        self.viewer.add_onetime(l_piece)

        rwall = rendering.PolyLine(self.rwall + [self.rwall[0]], False)
        rwall.add_attr(self.transform_translation)
        self.viewer.add_onetime(rwall)

        r_piece = rendering.PolyLine([self.rwall[-1], self.rwall[0]], False)
        r_piece.set_color(0.8, 0.8, 0.8)
        r_piece.add_attr(self.transform_translation)
        self.viewer.add_onetime(r_piece)

        self.viewer.draw_polygon(self.get_polygon(), color=(250, 0, 0))

        # draw rays
        for i in range(-2, 3):
            direction = i * 45 / 2 + 90 - self.angle * 180 / np.pi
            d = self.rayDist(direction, i+2)
            p = self.translate((640 + self.x, 100 + self.y), direction, d)
            self.viewer.draw_line([640 + self.x, 100 + self.y], p)

        left_bound, right_bound, lower_bound, upper_bound = self.get_bounds()

        self.viewer.set_bounds(left_bound, right_bound, lower_bound, upper_bound)
        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
