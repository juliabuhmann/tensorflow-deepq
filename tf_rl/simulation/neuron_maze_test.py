import unittest
import neuron_maze
import numpy as np
from skeleton import networkx_utils
from euclid import Vector3
import networkx as nx

cube_number = 26
settings = {
    'objects': [
        'skeleton'
    ],
    'colors': {
        'hero':   'red',
        'friend': 'yellow',

        'skeleton':   'yellow',
        'endnode': 'pink'
    },
    'object_reward': {
        'skeleton': 0.8,
        'endnode': 2,
    },
    'hero_bounces_off_walls': False,
    # 'wall_reward':-400,
    'wall_reward': 0.8,
    'outside_neuron_punishement': -3,
    'world_size': (256, 256, 128),
    'hero_initial_position': [100, 100, 50],
    "object_radius": 4.0,
    "num_objects": {
    },
    "num_observation_lines" : 32,
    "observation_line_length": 120.,
    "tolerable_distance_to_wall": 50,
    "wall_distance_penalty": -0.0,
    "delta_v": 50,
    "cube_number": 26,
    "knossos_offset": [1, 1, 1],
    "image_name": 'data/cube%i_tifstack/' %26,
    "observation_matrix_datasetname": 'distance',
    "reward_matrix_filename": '/raid/julia/projects/LSTM/data/distance_matrices/cube%i.h5' %cube_number,
    "volumetric_objects_filename":'/raid/julia/projects/LSTM/data/ground_truth_old_numbering/cube%i_neuron.h5' %cube_number,
    "observation_matrix_filename": '/raid/julia/projects/LSTM/data/distance_matrices/cube%i.h5'%cube_number,
    "nx_skeleton_filename": '/raid/julia/projects/LSTM/data/nx_skeletons/cube%i.gpickle' %cube_number,
    # Settings for how large the memory of the agent should be
    "len_of_memory": 5,
    "store_observation": True,
    # 6 encodes for
    "number_of_actions": 6,
    "relative_coordinate_system": True,
    'visited_location_punishement': -0.4,
    'distance_reward_factor': (1/40.),


}

class TestKnossosUtils(unittest.TestCase):
    def test_direction_vector_to_rotation_matrix(self):
        nm = neuron_maze.NeuronMaze()
        ori_dir_vector = np.array([0, 1, 0])

        # test_direction = np.array([0, 1, 0])
        # rotation_matrix = nm.direction_vector_to_rotation_matrix(test_direction)
        # to_test_direction = np.matmul(rotation_matrix, ori_dir_vector)
        # self.assertTrue((test_direction ==to_test_direction).all())
        #
        # test_direction = np.array([1, 0, 0])
        # rotation_matrix = nm.direction_vector_to_rotation_matrix(test_direction)
        # to_test_direction = np.matmul(rotation_matrix, ori_dir_vector)
        # self.assertTrue((test_direction ==to_test_direction).all())
        #
        # # test_direction = np.array([0, 0, -1])
        # # rotation_matrix = nm.direction_vector_to_rotation_matrix(test_direction)
        # # to_test_direction = np.matmul(rotation_matrix, ori_dir_vector)
        # # self.assertTrue((test_direction ==to_test_direction).all())
        # #
        # # test_direction = np.array([0, 0, 1])
        # # rotation_matrix = nm.direction_vector_to_rotation_matrix(test_direction)
        # # to_test_direction = np.matmul(rotation_matrix, ori_dir_vector)
        # # self.assertTrue((test_direction ==to_test_direction).all())
        # #
        # # test_direction = np.array([0, -1, 0])
        # # rotation_matrix = nm.direction_vector_to_rotation_matrix(test_direction)
        # # to_test_direction = np.matmul(rotation_matrix, ori_dir_vector)
        # # self.assertTrue((test_direction ==to_test_direction).all())
        # #
        # # test_direction = np.array([-1, 0, 0])
        # # rotation_matrix = nm.direction_vector_to_rotation_matrix(test_direction)
        # # to_test_direction = np.matmul(rotation_matrix, ori_dir_vector)
        # # self.assertTrue((test_direction ==to_test_direction).all())

    def test_hero_move(self):
        nm = neuron_maze.NeuronMaze()
        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.move([0, 1, 0])
        nm.hero.move([0, 1, 0])
        new_position = nm.hero.position
        self.assertEqual(new_position.y, 102)

        nm.hero.visited_positions.extend([(100, 101, 100), (100, 102, 100)])

        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        # Since the agent moves currently in the
        # direction of the absolute direction, it should not change anything
        self.assertTrue((direction_to_move == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(101, 100, 100), (102, 100, 100)])

        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        # The agent moves in the right direction, the direction to move indicates
        # that it should keep gooing in the right direction
        self.assertTrue((np.array([1, 0, 0]) == new_direction).all())

        direction_to_move = np.array([1, 0, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([0, -1, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (101, 101, 100)])

        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        # A direction_to_move vector of 0, 1, 0 indicates
        # that the agent does not change its current direction
        self.assertTrue((np.array([1, 1, 0]) == new_direction).all())
        #
        direction_to_move = np.array([-1, 0, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([-1, 1, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (100, 101, 100)])

        direction_to_move = np.array([1, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        # A direction_to_move vector of 0, 1, 0 indicates
        # that the agent does not change its current direction
        self.assertTrue((direction_to_move == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (100, 101, 101)])
        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([0, 1, 1]) == new_direction).all())


        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (100, 101, 99)])
        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([0, 1, -1]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (99, 101, 100)])
        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([-1, 1, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (101, 101, 100)])
        direction_to_move = np.array([1, 0, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([1, -1, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (101, 101, 100)])
        direction_to_move = np.array([1, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([1, 0, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (99, 101, 100)])
        direction_to_move = np.array([1, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([0, 1, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (101, 101, 101)])
        direction_to_move = np.array([0, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([1, 1, 1]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (101, 101, 101)])
        direction_to_move = np.array([1, 1, 0])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)
        self.assertTrue((np.array([1, 0, 0]) == new_direction).all())

        nm.hero.jump(new_position=[100, 100, 100])
        nm.hero.visited_positions.extend([(100, 100, 100), (101, 101, 101)])
        direction_to_move = np.array([1, 1, 1])
        new_direction = nm.relative_dir_to_abs_dir(direction_to_move)

        # self.assertTrue((np.array([1, 1, -1]) == new_direction).all())

    def test_make_agent_walk_along_nx_graph(self):
        nx_skeleton = networkx_utils.NxSkeleton()
        coords = np.array([[100, 100, 100], [100, 101, 100],
                           [100, 102, 101], [100, 103, 102], [100, 104, 103], [100, 105, 104]])
        edgelist = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        nx_skeleton.initialize_from_edgelist(coords, edgelist)

        nm = neuron_maze.NeuronMaze()
        nm.make_agent_walk_along_nx_graph(nx_skeleton, 0, 5)
        self.assertEqual(nm.hero.visited_positions[1], tuple(coords[1]))
        first_direction = Vector3(0, 1, 0)
        self.assertEqual(nm.directions.index(first_direction), nm.hero.taken_actions[0])


        current_nx_graph = nm.nx_skeletons.nx_graph_dic[1]
        nx_skeleton = networkx_utils.NxSkeleton(nx_graph=current_nx_graph)

        source = networkx_utils.get_nodes_with_a_specific_degree(current_nx_graph,
                                                                  degree_value=1)
        source_node = source[0]
        number_of_steps = 5
        successor_dic = nx.dfs_successors(current_nx_graph, source=source[0])
        for steps in range(number_of_steps):
            source = successor_dic[source[0]]
        target_node = source[0]
        print 'source node', source_node
        print 'target node', target_node
        nm = neuron_maze.NeuronMaze()
        nm.make_agent_walk_along_nx_graph(nx_skeleton, source_node, target_node)



        # nx_skeleton = networkx_utils.NxSkeleton()
        # coords = np.array([[100, 100, 100], [100, 99, 100],
        #                    [100, 98, 100], [100, 97, 100], [100, 96, 100],
        #                    [100, 95, 100]])
        # edgelist = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        # nx_skeleton.initialize_from_edgelist(coords, edgelist)
        #
        # nm = neuron_maze.NeuronMaze()
        # nm.make_agent_walk_along_nx_graph(nx_skeleton, 0, 5)
        # self.assertEqual(nm.hero.visited_positions[1], tuple(coords[1]))
        # first_direction = Vector3(0, 1, 0)
        # self.assertEqual(nm.directions.index(first_direction), nm.hero.taken_actions[0])
        #



