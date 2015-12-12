__author__ = 'julia'

import tf_rl.utils.svg as svg

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import h5py
from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2, Vector3, Point3

import tf_rl.utils.svg as svg

class GameObject(object):
    def __init__(self, position, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
        self.radius = self.settings["object_radius"]

        self.obj_type = obj_type
        self.position = position
        # self.speed    = speed
        self.bounciness = 1.0


    def wall_collisions(self):
        """Update speed upon collision with the wall."""
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and self.speed[dim] < 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and self.speed[dim] > 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness

    def move(self, direction):
        """Move as if dt seconds passed"""
        # self.position += dt * self.speed
        # avoid going out of the wall borders
        world_size = self.settings["world_size"]
        check_outside_world = False
        for dim in range(3):
            if self.position[dim] < 0:
                check_outside_world = True
            elif self.position[dim] >= world_size[dim]-2:
                check_outside_world = True
        if check_outside_world:
            self.jump()
            print "outside world resetting"
        else:
            self.position += direction

    def jump(self, new_position=None):
        if new_position == None:
            new_position = []
            for dim in range(3):
                new_position.append(np.random.randint(30, self.settings["world_size"][dim]-30))

        self.position = Point3(new_position[0], new_position[1], new_position[2])

    # def step(self, dt):
    #     """Move and bounce of walls."""
    #     # self.wall_collisions()
    #     self.move(dt)

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self, scale_up=Point2(1, 1)):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        position_2d = Point2(self.position.x, self.position.y)
        position_2d = Point2(position_2d.x*scale_up.x, position_2d.y*scale_up.y)
        return svg.Circle(position_2d, self.radius, color=color)

class NeuronMaze(object):
    def __init__(self, settings):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.size  = self.settings["world_size"]
        # self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
        #               LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
        #               LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
        #               LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]

        self.hero = GameObject(Point3(*self.settings["hero_initial_position"]),
                               # Vector2(*self.settings["hero_initial_speed"]),
                               "hero",
                               self.settings)
        if not self.settings["hero_bounces_off_walls"]:
            self.hero.bounciness = 0.0

        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            for _ in range(number):
                self.spawn_object(obj_type)



        self.object_reward = 0
        self.collected_rewards = []

        # every observation_line sees one of objects or wall and
        # two numbers representing speed of the object (if applicable)
        # self.eye_observation_size = len(self.settings["objects"]) + 3
        # additionally there are two numbers representing agents own speed.


        self.directions = [Vector3(*d) for d in [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
        self.num_actions = len(self.directions)
        self.observation_lines = [Vector3(*d) for d in [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
        self.observation_size = len(self.observation_lines)
        self.objects_eaten = defaultdict(lambda: 0)
        self.image_name = settings['image_name']
        # Load reward matrix
        try:
            reward_filename = settings['reward_matrix_filename']
            f = h5py.File(reward_filename, 'r')
            reward_matrix = f['distance'].value
            f.close()
            self.reward_matrix = reward_matrix
        except KeyError:
            print "%s does not exist" %reward_filename

        try:
            observation_filename = settings['observation_matrix_filename']
            f = h5py.File(observation_filename, 'r')
            observation_matrix = f['distance'].value
            f.close()
            self.observation_matrix = observation_matrix
        except KeyError:
            print "%s does not exist" %observation_filename


    def perform_action(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions
        # self.hero.position += self.directions[action_id]
        self.hero.move(self.directions[action_id])
        # self.hero.speed += self.directions[action_id] * self.settings["delta_v"]

    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        position = np.random.uniform([radius, radius, radius], np.array(self.size) - radius)
        position = Point3(float(position[0]), float(position[1]), float(position[2]))
        # max_speed = np.array(self.settings["maximum_speed"])
        # speed    = np.random.uniform(-max_speed, max_speed).astype(float)
        # speed = Vector2(float(speed[0]), float(speed[1]))

        self.objects.append(GameObject(position, obj_type, self.settings))

    def step(self, dt):
        """Simulate all the objects for a given ammount of time.

        Also resolve collisions with the hero"""
        # for obj in self.objects + [self.hero] :
        #     obj.step(dt)
        # self.resolve_collisions()
        dummy = 3

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def resolve_collisions(self):
        """If hero touches, hero eats. Also reward gets updated."""
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        to_remove = []
        for obj in self.objects:
            if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                to_remove.append(obj)
        for obj in to_remove:
            self.objects.remove(obj)
            self.objects_eaten[obj.obj_type] += 1
            self.object_reward += self.settings["object_reward"][obj.obj_type]
            self.spawn_object(obj.obj_type)

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        # num_obj_types = len(self.settings["objects"]) + 1 # and wall
        # max_speed_x, max_speed_y = self.settings["maximum_speed"]

        # observable_distance = self.settings["observation_line_length"]

        # relevant_objects = [obj for obj in self.objects
        #                     if obj.position.distance(self.hero.position) < observable_distance]
        # objects sorted from closest to furthest
        # relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))
        current_position = self.hero.position
        observation_list = []
        for direction in self.observation_lines:
            measurement_location = current_position + direction
            observation = self.observation_matrix[measurement_location.x, measurement_location.y, measurement_location.z]
            observation_list.append(observation)
        observation = np.array(observation_list)
        # observation = self.observation_matrix
        # observation = np.zeros(self.observation_size)
        # observation_offset = 0
        # for i, observation_line in enumerate(self.observation_lines):
        #     # shift to hero position
        #     observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
        #                                     self.hero.position + Vector2(*observation_line.p2))
        #
        #     observed_object = None
        #     # if end of observation line is outside of walls, we see the wall.
        #     if not self.inside_walls(observation_line.p2):
        #         observed_object = "**wall**"
        #     for obj in relevant_objects:
        #         if observation_line.distance(obj.position) < self.settings["object_radius"]:
        #             observed_object = obj
        #             break
        #     object_type_id = None
        #     speed_x, speed_y = 0, 0
        #     proximity = 0
        #     if observed_object == "**wall**": # wall seen
        #         object_type_id = num_obj_types - 1
        #         # a wall has fairly low speed...
        #         speed_x, speed_y = 0, 0
        #         # best candidate is intersection between
        #         # observation_line and a wall, that's
        #         # closest to the hero
        #         best_candidate = None
        #         for wall in self.walls:
        #             candidate = observation_line.intersect(wall)
        #             if candidate is not None:
        #                 if (best_candidate is None or
        #                         best_candidate.distance(self.hero.position) >
        #                         candidate.distance(self.hero.position)):
        #                     best_candidate = candidate
        #         if best_candidate is None:
        #             # assume it is due to rounding errors
        #             # and wall is barely touching observation line
        #             proximity = observable_distance
        #         else:
        #             proximity = best_candidate.distance(self.hero.position)
        #     elif observed_object is not None: # agent seen
        #         object_type_id = self.settings["objects"].index(observed_object.obj_type)
        #         speed_x, speed_y = tuple(observed_object.speed)
        #         intersection_segment = obj.as_circle().intersect(observation_line)
        #         assert intersection_segment is not None
        #         try:
        #             proximity = min(intersection_segment.p1.distance(self.hero.position),
        #                             intersection_segment.p2.distance(self.hero.position))
        #         except AttributeError:
        #             proximity = observable_distance
        #     for object_type_idx_loop in range(num_obj_types):
        #         observation[observation_offset + object_type_idx_loop] = 1.0
        #     if object_type_id is not None:
        #         observation[observation_offset + object_type_id] = proximity / observable_distance
        #     observation[observation_offset + num_obj_types] =     speed_x   / max_speed_x
        #     observation[observation_offset + num_obj_types + 1] = speed_y   / max_speed_y
        #     assert num_obj_types + 2 == self.eye_observation_size
        #     observation_offset += self.eye_observation_size
        #
        # observation[observation_offset]     = self.hero.speed[0] / max_speed_x
        # observation[observation_offset + 1] = self.hero.speed[1] / max_speed_y
        # assert observation_offset + 2 == self.observation_size

        return observation

    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        return res - self.settings["object_radius"]

    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        # wall_reward =  self.settings["wall_distance_penalty"] * \
        #                np.exp(-self.distance_to_walls() / self.settings["tolerable_distance_to_wall"])
        # assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
        # Get reward from reward matrix
        current_position = self.hero.position
        reward = self.reward_matrix[int(current_position.x), int(current_position.y), int(current_position.z)]
        # total_reward = self.object_reward
        # self.object_reward = 0
        self.collected_rewards.append(reward)
        return reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])
        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, scale_up=[2, 2]):
        """Return svg representation of the simulator"""
        scale_up = Point2(scale_up[0], scale_up[1])
        size_value = self.size
        # Update image_name
        section_id = self.hero.position.z
        section_id_str = '%04i' %section_id
        image_name = self.image_name + section_id_str + ".png"
        scene = svg.Scene((size_value[0]*scale_up.x, size_value[1]*scale_up.y), image_name=image_name)


        # for line in self.observation_lines:
        #     scene.add(svg.Line(line.p1 + self.hero.position*scale_up,
        #                        line.p2 + self.hero.position*scale_up))

        for obj in self.objects + [self.hero] :
            scene.add(obj.draw(scale_up=scale_up))

        return scene

