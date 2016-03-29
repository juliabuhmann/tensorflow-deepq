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
import networkx as nx
from skeleton import networkx_utils
from random import choice
import math3d as m3d
import helpers


import tf_rl.utils.svg as svg

# Default settings
cube_number = 26

default_settings = {
    'colors': {
        'hero':   'red',
        'friend': 'yellow',
        'skeleton':   'yellow',
        'endnode': 'pink'
    },
    'rewards': {
        'skeleton': 0.8,
        'endnode': 2,
        # Reward settings
        'visited_location_punishement': -0.4,
        'distance_reward_factor': (1/40.),
        # 'distance_reward_factor': (0.),
        'wall_reward': 0.8,
        'outside_neuron_punishement': -3,
            },
    'world_size': (256, 256, 128),
    "object_radius": 4.0,
    #File settings
    "cube_number": cube_number,
    "knossos_offset": {},
    "image_name": 'data/cube%i_tifstack/' %cube_number,

    # Settings for how large the memory of the agent should be
    "len_of_memory": 4,
    "store_observation": True,

    # Settings for what the agent sees /observes and where it can go
    "number_of_actions": 5,
    "relative_coordinate_system": True,
    "observation_options": 'global',

    "observation_matrix_datasetname": 'distance',
    "reward_matrix_filename": '/raid/julia/projects/LSTM/data/distance_matrices/cube%i.h5' %cube_number,
    "volumetric_objects_filename":'/raid/julia/projects/LSTM/data/ground_truth_old_numbering/cube%i_neuron.h5' %cube_number,
    "observation_matrix_filename": '/raid/julia/projects/LSTM/data/distance_matrices/cube%i.h5'%cube_number,
    "nx_skeleton_filename": '/raid/julia/projects/LSTM/data/nx_skeletons/cube%i.gpickle' %cube_number,
}



class GameObject(object):
    def __init__(self, position, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""


        self.settings = settings
        self.radius = self.settings["object_radius"]

        self.obj_type = obj_type
        self.position = position

        self.number_of_steps = 0
        self.is_outside_world = False

        self.collected_eaten_obj = defaultdict(lambda: [])
        self.visited_positions = []
        self.taken_actions = []
        self.observed_observations = []
        self.reached_goal = False
        self.stepped_outside = False
        self.game_ends = False


    def reset(self):
        self.is_outside_world = False
        self.visited_positions = []
        self.reached_goal = False
        self.stepped_outside = False
        self.game_ends = False
        self.number_of_steps = 0
        self.taken_actions = []
        self.observed_observations = []
        self.collected_eaten_obj = defaultdict(lambda: [])

    def move(self, direction):
        """Move one step"""
        # avoid going out of the wall borders
        world_size = self.settings["world_size"]
        check_outside_world = False
        self.position += direction
        for dim in range(3):
            if self.position[dim] < 10:
                check_outside_world = True
            elif self.position[dim] >= world_size[dim]-10:
                check_outside_world = True

        self.number_of_steps += 1
        if check_outside_world:
            self.is_outside_world = True
        else:
            self.is_outside_world = False

    def jump(self, new_position=None):
        if new_position is None:
            new_position = []
            for dim in range(3):
                new_position.append(np.random.randint(30, self.settings["world_size"][dim]-30))
        self.position = Point3(new_position[0], new_position[1], new_position[2])

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self, scale_up=Point2(1, 1), color=None):
        """Return svg object for this item."""
        if color is None:
            color = self.settings["colors"][self.obj_type]
        position_2d = Point2(self.position.x, self.position.y)
        position_2d = Point2(position_2d.x*scale_up.x, position_2d.y*scale_up.y)
        return svg.Circle(position_2d, self.radius, color=color)

    def outside_world(self, pos=None, margin=30):
        world_size = self.settings["world_size"]
        check_outside_world = False
        if pos is None:
            pos = self.position
        for dim in range(3):
            if pos[dim] < margin:
                check_outside_world = True
            elif pos[dim] >= world_size[dim]-margin:
                check_outside_world = True
        return check_outside_world

    def get_current_direction(self):
        if len(self.visited_positions) > 1:
            direction_of_agent = np.array(self.visited_positions[-1]) - np.array(self.visited_positions[-2])
        else:
            # direction_of_agent = np.array([0, 1, 0])
            direction_of_agent = None
        return direction_of_agent



class NeuronMaze(object):
    def __init__(self, settings={}):
        """Initiallize game simulator with settings"""
        if not settings:
            settings = default_settings
        self.settings = settings
        self.size = self.settings["world_size"]

        voxel_factors = settings['voxel_factors']
        self.voxel_factors = Point3(*voxel_factors)

        self.hero = GameObject(Point3(self.size[0]//2, self.size[1]//2, self.size[2]//2),
                               "hero",
                               self.settings)
        self.objects = []
        self.object_reward = 0
        self.collected_rewards = []

        # every observation_line sees the values given by the observation matrix
        if settings["number_of_actions"] == 6:
            # This is the original setting
            self.directions = [Vector3(*d) for d in [[1, 0, 0], [0, 1, 0],
                                                     [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
            self.observation_lines = [Vector3(*d) for d in [[1, 0, 0],
                                                            [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
            self.observation_lines.extend([Vector3(*d)*3 for d in [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]])
            self.observation_lines.extend([Vector3(*d)*5 for d in [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]])
        elif settings["number_of_actions"] == 9:
            # The agent can only move within a forward directed cone
            self.directions = [Vector3(*d) for d in [[0, 1, 0], [-1, 1, 0], [1, 1, 0], [-1, 1, 1],
                                                     [-1, 1, -1], [0, 1, -1], [0, 1, 1], [1, 1, -1], [1, 1, 1]]]

            # The agent should still be able to look backwards

            self.observation_lines = [Vector3(*d) for d in [[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
            # But of course also where it can potentially step in
            self.observation_lines.extend(self.directions)
            self.observation_lines.extend([vector*3 for vector in self.directions])
            self.observation_lines.extend([vector*5 for vector in self.directions])
            if settings["observation_options"] == 'global':
                # Add some more to the receptive field of the agent
                orthogonal_lines = [Vector3(*d) for d in [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]]
                self.observation_lines.extend([vector*1 for vector in orthogonal_lines])
                self.observation_lines.extend([vector*3 for vector in orthogonal_lines])
                self.observation_lines.extend([vector*5 for vector in orthogonal_lines])


        elif settings["number_of_actions"] == 5:
            # The agent can only move within a forward directed cone, but not diagonal
            self.directions = [Vector3(*d) for d in [[0, 1, 0], [-1, 1, 0], [1, 1, 0], [0, 1, -1], [0, 1, 1]]]

            # The agent should still be able to look backwards
            self.observation_lines = [Vector3(*d) for d in [[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
            # But of course also where it can potentially step in
            self.observation_lines.extend(self.directions)
            self.observation_lines.extend([vector*3 for vector in self.directions])
            self.observation_lines.extend([vector*5 for vector in self.directions])


            if settings["observation_options"] == 'global':
                # Add some more to the receptive field of the agent
                orthogonal_lines = [Vector3(*d) for d in [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]]
                self.observation_lines.extend([vector*1 for vector in orthogonal_lines])
                self.observation_lines.extend([vector*3 for vector in orthogonal_lines])
                self.observation_lines.extend([vector*5 for vector in orthogonal_lines])

        # self.directions = [Vector3(*d) for d in [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
        self.num_actions = len(self.directions)

        # Adds one because it is also observed on what the agent is currently standing on
        len_of_memory = settings['len_of_memory']
        self.len_of_memory = len_of_memory
        store_observations_as_memory = settings['store_observation']
        if store_observations_as_memory:
            len_of_memory *= 2
        self.observation_size = len(self.observation_lines)+1+len_of_memory

        # Set the reward
        self.visited_location_punishement = settings['rewards']['visited_location_punishement']
        self.distance_reward_factor = settings['rewards']['distance_reward_factor']

        self.objects_eaten = defaultdict(lambda: 0)
        self.proximity_dic = defaultdict(lambda: 0)
        for object_type, max_distance in settings['rewards']['proximity_dic'].iteritems():
            self.proximity_dic[object_type] = max_distance
        self.image_name = settings['image_name']
        self.removed_objects = []
        self.margin = [30, 30, 10]
        self.games_played = 0
        self.dic_objects = {}
        self.knossos_offset = settings["knossos_offset"]
        self.cube_number = settings["cube_number"]
        self.relative_coordinate_system = settings["relative_coordinate_system"]

        # Load reward matrix
        try:
            reward_filename = settings['reward_matrix_filename']
            print reward_filename
            f = h5py.File(reward_filename, 'r')
            reward_matrix = f['distance'].value
            f.close()
            self.reward_matrix = reward_matrix
            self.reward_matrix_ori = reward_matrix.copy()
        except KeyError:
            print "%s does not exist" %reward_filename

        try:
            observation_filename = settings['observation_matrix_filename']
            dataset_name = settings['observation_matrix_datasetname']
            f = h5py.File(observation_filename, 'r')
            observation_matrix = f[dataset_name].value
            f.close()
            self.observation_matrix = observation_matrix
        except KeyError:
            print "%s does not exist" %observation_filename

        if 'nx_skeleton_filename' in settings:
            nx_skeleton_filename = settings['nx_skeleton_filename']
            try:
                loaded_skeleton = nx.read_gpickle(nx_skeleton_filename)
                print('loaded skeleton file', nx_skeleton_filename)
            except KeyError:
                print "%s does not exist" %nx_skeleton_filename
            margin = self.margin
            world_size = self.settings['world_size']
            lower_bound = margin
            upper_bound = np.array(world_size) - np.array(lower_bound)
            networkx_utils.crop_nx_graph(loaded_skeleton, lower_bound, upper_bound)
            self.nx_graph = loaded_skeleton
            nx_skeletons = networkx_utils.NxSkeletons(nx_skeleton=networkx_utils.NxSkeleton(loaded_skeleton))
            self.nx_skeletons = nx_skeletons

        if 'volumetric_objects_filename' in settings:
            volumetric_objects_filename = settings['volumetric_objects_filename']
            try:
                f = h5py.File(volumetric_objects_filename, 'r')
                data = f['seg'].value
                f.close()
                self.volumetric_object_matrix = data
            except KeyError:
                print "%s does not exist" %volumetric_objects_filename
        self.reinitialize_world()

    def skeletons_to_objects(self, seg_id=None, blocked_objects=[]):
        # If seg_id is given, only nodes are added as objects that have the specific seg_id
        # blocked objects --> in the case that an object is the source object,
        # this should not be added to the pool of objects, otherwise the agent
        # starts at the end and the game is over immediately
        current_nx_graph = self.nx_skeletons.nx_graph_dic[seg_id]
        degree_of_current_graph = nx.degree(current_nx_graph)
        for node_id, node_attr in current_nx_graph.nodes_iter(data=True):
            pos = node_attr['position']
            pos = Point3(pos[0], pos[1], pos[2])
            current_seg_id = node_attr['seg_id']
            degree_of_current_node = degree_of_current_graph[node_id]
            if degree_of_current_node == 1:
                if not pos in blocked_objects:
                    self.spawn_object('endnode', pos)
            else:
                self.spawn_object('skeleton', pos)
            if 'volumetric_objects_filename' in self.settings:
                seg_id_vol = self.volumetric_object_matrix[pos.x, pos.y, pos.z]
                assert current_seg_id == seg_id_vol

    def conv_dir_to_action_id(self, dir):
        dir = helpers.np_array_to_vector3(dir)
        if dir in self.directions:
            ind = self.directions.index(dir)
            return ind
        else:
            # print 'direction not in the set of actions'
            # print dir
            # print 'available directions'
            # print self.directions
            return 0


    def abs_dir_to_rel_dir(self, direction, direction_of_agent=None):
        if direction_of_agent is None:
            direction_of_agent = self.hero.get_current_direction()
            if direction_of_agent is None:
                direction_of_agent = direction.copy()

        rotation_matrix = helpers.get_rot_matrix_from_vec1_to_vec2(direction_of_agent, [0, 1, 0])
        relative_direction = np.matmul(rotation_matrix, direction)
        relative_direction = helpers.discretize_vector_to_voxel(relative_direction)
        return relative_direction

    def relative_dir_to_abs_dir(self, direction):
        direction = np.array(direction)
        direction_of_agent = self.hero.get_current_direction()
        if direction_of_agent is None:
            direction_of_agent = direction.copy()
            direction_of_agent = helpers.np_array_to_vector3(direction_of_agent)
            direction_of_agent.normalize()
            direction_of_agent = helpers.discretize_vector_to_voxel(direction_of_agent)
        rotation_matrix = helpers.get_rot_matrix_from_vec1_to_vec2([0, 1, 0], direction_of_agent)

        absolute_direction = np.matmul(rotation_matrix, direction)
        absolute_direction = helpers.discretize_vector_to_voxel(absolute_direction)
        check_direction = np.matmul(rotation_matrix, np.array([0, 1, 0]))
        check_direction = helpers.discretize_vector_to_voxel(check_direction)
        assert ((check_direction == direction_of_agent).all()), "rotating the system to agent's perspective is broken"
        return absolute_direction

    def perform_action(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions, 'action id %iid not in' %action_id

        # action_id = choice(range(self.num_actions))
        # Add sort of memory to the agent (by changing the observation for already visited pixels
        # self.observation_matrix[self.hero.position.x, self.hero.position.y, self.hero.position.z] = -1
        if self.relative_coordinate_system:
            # Get old direction vector
            absolute_direction = self.relative_dir_to_abs_dir(self.directions[action_id])
        else:
            absolute_direction = self.directions[action_id]

        # Assert that the agent moves only one step
        for dim in range(3):
            assert absolute_direction[dim] < 2
        self.hero.move(absolute_direction)
        self.hero.taken_actions.append(action_id)
        # This is really conservative; as soon as it steps into a membrane, the game is over
        agent_sits_on = self.observation_matrix[self.hero.position.x, self.hero.position.y, self.hero.position.z]
        outside_world = self.hero.is_outside_world
        # Two cases terminates the game: stepping outside of the neuron or reaching an endpoint of a skeleton
        self.hero.visited_positions.append(tuple([self.hero.position.x, self.hero.position.y, self.hero.position.z]))
        if outside_world:
            # Punish the bouncing in the wall
            self.object_reward = self.settings['rewards']['wall_reward']
            self.hero.game_ends = True
        elif -1 < agent_sits_on < 2:
            self.object_reward = self.settings['rewards']['outside_neuron_punishement']
            self.objects_eaten['outside_neuron'] += 1
            # self.hero.is_outside_world = True
            self.hero.reached_goal = False
            self.hero.stepped_outside = True
            self.hero.game_ends = True
        else:
            self.resolve_collisions()

    def reinitialize_world(self, nx_graph_id=None, center_node=False, start_node=False, number_of_initial_steps=0):
        # assert nx_graph_id is None and node_id is not None,'node id can only be specified when also nx_graph id is specified'
        self.hero.reset()
        blocked_objects = []
        if 'nx_skeleton_filename' in self.settings:
            # Select random skeleton
            node_to_jump = None
            if nx_graph_id is None:
                nx_graph_id = choice(self.nx_skeletons.nx_graph_dic.keys())
            current_graph = self.nx_skeletons.nx_graph_dic[nx_graph_id]
            if center_node:
                node_to_jump = networkx_utils.get_center_node_from_nx_graph(current_graph)
                if not len(node_to_jump) == 1:
                    node_to_jump = choice(current_graph.nodes())
                else:
                    node_to_jump = node_to_jump[0][0]
            if start_node:
                assert center_node is False, 'either start node or center node have to be set to False'
                node_to_jump = networkx_utils.get_nodes_with_a_specific_degree(current_graph)
                if len(node_to_jump) == 2:
                    node_to_jump = node_to_jump[0]
            if node_to_jump is None:
                node_to_jump = choice(current_graph.nodes())

            if number_of_initial_steps > 0 and current_graph.number_of_nodes()-2 > number_of_initial_steps:
                successor_dic = nx.dfs_successors(current_graph, source=node_to_jump)
                source = [node_to_jump]
                source_node = node_to_jump
                # Save the starting position for later to make sure that it is
                # not added as an endpoint in the game
                source_pos = current_graph.node[node_to_jump]['position']
                source_pos = Point3(source_pos[0], source_pos[1], source_pos[2])
                blocked_objects.append(source_pos)
                for steps in range(number_of_initial_steps):
                    source = successor_dic[source[0]]
                target_node = source[0]
                self.make_agent_walk_along_nx_graph(current_graph, source_node, target_node)
                # Remove potential endnode objects from the source node


                # print 'hero walked already a bit', self.hero.visited_positions
                # print 'hero walked already a bit', self.hero.taken_actions
                # print 'hero walked already a bit', self.hero.observed_observations
            else:
                pos = self.nx_graph.node[node_to_jump]['position']

                self.hero.jump(pos)
        else:
            self.jump()

        #Get id for position
        pos = self.hero.position
        if 'volumetric_objects_filename' in self.settings:
            seg_id = self.volumetric_object_matrix[pos.x, pos.y, pos.z]
            # new_reward_matrix = self.reward_matrix_ori.copy()
            # new_observation_matrix = self.reward_matrix_ori.copy()
            # new_reward_matrix[self.volumetric_object_matrix != seg_id] = 0
            # self.reward_matrix = new_reward_matrix
            # self.observation_matrix = new_observation_matrix
            # self.observation_matrix = new_reward_matrix
        else:
            seg_id= None
        # #Reinitialize the skeletons
        # for obj in self.removed_objects:
        #     self.objects.append(obj)
        self.objects = []
        self.removed_objects= []
        self.objects_eaten['skeleton'] = 0
        self.skeletons_to_objects(seg_id=seg_id, blocked_objects=blocked_objects)
        self.games_played += 1

        # new_observation_matrix = self.observation_matrix_ori.copy()
        # self.observation_matrix = new_observation_matrix

    def spawn_object(self, obj_type, pos=None):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]

        # max_speed = np.array(self.settings["maximum_speed"])
        # speed    = np.random.uniform(-max_speed, max_speed).astype(float)
        # speed = Vector2(float(speed[0]), float(speed[1]))
        if pos is not None:
            self.objects.append(GameObject(pos, obj_type, self.settings))
        else:
            pos = np.random.uniform([radius, radius, radius], np.array(self.size) - radius)
            pos = Point3(float(pos[0]), float(pos[1]), float(pos[2]))
            self.objects.append(GameObject(pos, obj_type, self.settings))

    def squared_distance(self, p1, p2):
        # return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        voxel_factors = self.voxel_factors
        return ((p1-p2)*voxel_factors).magnitude_squared()

    def distance(self, p1, p2):
        # Accounts for possible anisotropy
        voxel_factors = self.voxel_factors
        return ((p1-p2)*voxel_factors).magnitude()

    def resolve_collisions(self):
        """If hero is on position, hero eats. Also reward gets updated."""
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        to_remove = []
        distance_list = []
        object_list = []
        proximity_distance_list = []
        proximity_object_list = []
        proximity_to_reward = []
        for obj in self.objects:
            if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                distance_list.append(self.squared_distance(self.hero.position, obj.position))
                object_list.append(obj)
            # Sense the proximity but do not eat the object,
            # but still get either punishement or reward for proximity
            for obj_type, max_distance in self.proximity_dic.iteritems():
                if self.squared_distance(self.hero.position, obj.position) < (2*max_distance)**2:
                    proximity_distance_list.append(self.distance(self.hero.position, obj.position))
                    proximity_object_list.append(obj)

        assert len(self.proximity_dic) <= 1

        if distance_list:
            # Here, only the closest object gets eaten. This assures that several 'skeleton' objects do
            # not get eaten at once
            min_index = np.argmin(np.array(distance_list))
            to_remove.append(object_list[min_index])

        if proximity_distance_list:

            max_distance = self.proximity_dic['skeleton']
            min_index = np.argmin(np.array(proximity_distance_list))
            distance = proximity_distance_list[min_index]
            proximity_to_reward.append(proximity_object_list[min_index])
            prox_reward = ((distance-max_distance)**2)
            if (max_distance - distance) < 0:
                # prox_reward = (max_distance- distance)
                prox_reward = 0
            # Normalize number by highest possible score
            prox_reward /= max_distance**2
            self.object_reward += prox_reward
            # print prox_reward

        # If there is an object in proximity, get reward accordingly

        check_for_goal = False
        assert len(to_remove) <= 1
        for obj in to_remove:
            self.objects.remove(obj)
            self.objects_eaten[obj.obj_type] += 1
            self.removed_objects.append(obj)
            self.object_reward += self.settings["rewards"][obj.obj_type]
            if 'endnode' in obj.obj_type:
                check_for_goal = True
                self.objects_eaten['goal'] += 1
                self.hero.reached_goal = True
            current_position = self.hero.position.copy()
            self.hero.collected_eaten_obj[str(obj.obj_type + '_hero_pos')].append(current_position)
            self.hero.collected_eaten_obj[obj.obj_type].append(obj.position)
        if check_for_goal:
            self.hero.game_ends = True

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
            # print "---"
            # print direction
            # print self.hero.position
            # print self.hero.visited_positions[-2:]
            if self.relative_coordinate_system:
                direction = self.relative_dir_to_abs_dir(direction)
            measurement_location = current_position + direction
            observation = self.observation_matrix[measurement_location.x, measurement_location.y, measurement_location.z]

            # Check wether th agent already visited this position and change observation accordingly
            # obs_position = tuple([measurement_location.x, measurement_location.y, measurement_location.z])
            # if obs_position in self.hero.visited_positions:
            #     observation = -1

            observation_list.append(observation)
        # The agent also sense where it is sitting right now
        current_observation = self.observation_matrix[current_position.x, current_position.y, current_position.z]
        observation_list.append(current_observation)
        assert len(self.hero.taken_actions) == len(self.hero.observed_observations), \
            '% iactions  %i observation ' %(len(self.hero.taken_actions), len(self.hero.observed_observations))
        for memory_event in range(self.len_of_memory):
            if len(self.hero.taken_actions) > self.len_of_memory:
                action_made = self.hero.taken_actions[-memory_event]
                observed_distance = self.hero.observed_observations[-memory_event]
            elif len(self.hero.taken_actions) > 0:
                # Repeat the last value
                action_made = self.hero.taken_actions[-1]
                observed_distance = self.hero.observed_observations[-1]
            else:
                # Use dummy value
                action_made = 0
                observed_distance = 0
            observation_list.extend([action_made, observed_distance])
        self.hero.observed_observations.append(current_observation)
        observation = np.array(observation_list)
        # print observation, 'observation'
        return observation

    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        return res - self.settings["object_radius"]

    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""

        # Get reward from reward matrix
        current_position = self.hero.position
        # max_value = np.amax(self.reward_matrix)
        reward = self.reward_matrix[int(current_position.x),
                                    int(current_position.y), int(current_position.z)]*self.distance_reward_factor

        # If position was already visited, the agent should not get any reward
        if self.visited_location_punishement != 0:
            if tuple(current_position) in self.hero.visited_positions:
                reward = self.visited_location_punishement

        # reward = 0
            # print current_position
            # print self.hero.visited_positions
            # print 'already visited'

        # reward = 0
        # if reward == 0:
        #     # This means being outside of the neuron
        #     # reward = -10
        #     self.objects_eaten['outside_neuron'] += 1
        # else:
        #     self.objects_eaten['inside_neuron'] += 1
        reward += self.object_reward
        self.collected_rewards.append(reward)
        self.object_reward = 0
        return reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        print np.max(np.array(plottable))
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

    def to_html(self, scale_up=[2, 2], extra_image=False):
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

        for obj in self.objects + [self.hero]:
            z_pos = obj.position.z
            if z_pos == section_id:
                scene.add(obj.draw(scale_up=scale_up))

        for obj in self.removed_objects:
            z_pos = obj.position.z
            if z_pos == section_id:
                scene.add(obj.draw(scale_up=scale_up, color='green'))
        if extra_image:
            if self.hero.stepped_outside:
                scene = svg.Scene((size_value[0]*scale_up.x, size_value[1]*scale_up.y), image_name='data/looser.png')
            if self.hero.reached_goal:
                scene = svg.Scene((size_value[0]*scale_up.x, size_value[1]*scale_up.y), image_name='data/winner.png')
        return scene

    def make_agent_walk_along_nx_graph(self, nx_skeleton, source_node, target_node):
        if isinstance(nx_skeleton, nx.Graph):
            nx_skeleton = networkx_utils.NxSkeleton(nx_graph=nx_skeleton)
        self.relative_coordinate_system = True

        nx_skeleton.add_direction_vector_to_nodes(check_for_circles_branches=False)
        nx_graph = nx_skeleton.nx_graph
        walk = nx.shortest_path(nx_graph, source_node, target_node)

        # Let the hero walk along the given graph
        for ii in walk:
            first_pos = nx_graph.node[ii]['position']
            dir_vector = nx_graph.node[ii]['dir_vector']
            # print dir_vector
            dir_vector = helpers.discretize_vector_to_voxel(dir_vector)
            rel_vector = self.abs_dir_to_rel_dir(dir_vector)
            rel_vector = helpers.discretize_vector_to_voxel(rel_vector)
            action_id = self.conv_dir_to_action_id(rel_vector)
            # print "----"
            # print 'dir vector', dir_vector
            # print 'rel_vector', rel_vector
            # print 'direction vector of agent', self.hero.get_current_direction()
            # print 'action id', action_id
            # print self.directions[action_id]

            self.hero.jump(first_pos)
            self.hero.observed_observations.append(self.observation_matrix[first_pos[0], first_pos[1], first_pos[2]])
            self.hero.visited_positions.append(tuple(nx_graph.node[ii]['position']))
            self.hero.taken_actions.append(action_id)


            # node_id1 = walk[ii]
            # node_id2 = walk[ii+1]
            # pos1 = nx_graph.node[node_id1]['position']
            # pos2 = nx_graph.node[node_id2]['position']




            # self.observe()



    # def make_agent_walk_along_nx_graph(self, nx_graph, source_node, target_node):
    #     self.relative_coordinate_system = True
    #     walk = nx.shortest_path(nx_graph, source_node, target_node)
    #
    #     first_pos = nx_graph.node[source_node]['position']
    #     print 'first pos', first_pos
    #     self.hero.jump(first_pos)
    #     self.hero.observed_observations.append(self.observation_matrix[first_pos[0], first_pos[1], first_pos[2]])
    #     self.hero.visited_positions.append(tuple(nx_graph.node[source_node]['position']))
    #     self.hero.taken_actions.append(0)
    #     sec_pos = nx_graph.node[walk[1]]['position']
    #     print 'sec_pos pos', sec_pos
    #     self.hero.jump(sec_pos)
    #     self.hero.observed_observations.append(self.observation_matrix[sec_pos[0], sec_pos[1], sec_pos[2]])
    #     self.hero.visited_positions.append(tuple(sec_pos))
    #     self.hero.taken_actions.append(0)
    #
    #     # Let the hero walk along the given graph
    #     print "------------"
    #     for ii in walk:
    #         print nx_graph.node[ii]['position']
    #     for ii in range(1, len(walk)-1):
    #         print "new poistio ------"
    #         node_id1 = walk[ii]
    #         node_id2 = walk[ii+1]
    #
    #         pos1 = nx_graph.node[node_id1]['position']
    #         pos2 = nx_graph.node[node_id2]['position']
    #         print 'pos1', pos1
    #         print 'pos2', pos2
    #         new_positions = []
    #         for dim in range(3):
    #             step = np.abs(pos2[dim] - pos1[dim])
    #             print step
    #             if step > 1:
    #                 print "inconsistencies in the graph, filling up"
    #
    #
    #         print pos2
    #         dir = pos2 - pos1
    #         if ii == -10:
    #             direction_of_agent = pos1 -pos2
    #         else:
    #             direction_of_agent = None
    #         print 'dir of agent', direction_of_agent
    #         rel_dir = self.abs_dir_to_rel_dir(dir, direction_of_agent=direction_of_agent)
    #
    #
    #         print 'dir of agent ', self.hero.get_current_direction()
    #         print 'rel dir', rel_dir
    #         print 'abs dir', dir
    #         action_id = self.conv_dir_to_action_id(rel_dir)
    #         self.observe()
    #         self.perform_action(action_id)
    #         print pos1
    #         print self.hero.position, helpers.np_array_to_vector3(pos2)
    #         assert self.hero.position == helpers.np_array_to_vector3(pos2)




