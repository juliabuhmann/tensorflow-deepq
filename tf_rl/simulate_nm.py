import time

from IPython.display import clear_output, display, HTML
from os.path import join, exists
from os import makedirs
import tensorflow as tf
import random
import numpy as np
import h5py
import time
from skeleton import networkx_utils
from skeleton import knossos_utils


def visualize(simulation):
    clear_output(wait=True)
    svg_html = simulation.to_html()
    display(svg_html)
    time.sleep(0.05)
    if simulation.hero.game_ends:
        print 'take extra image'
        clear_output(wait=True)
        svg_html = simulation.to_html(extra_image=True)
        display(svg_html)
        time.sleep(0.1)


class GameWatcher(object):
    def __init__(self):
        self.number_of_games = 0
        self.number_of_steps = 0
        self.actions = []
        self.collected_reward = []
        self.number_of_reached_goals = 0
        self.number_of_lost_games = 0
        self.number_of_outside_steps = 0
        self.collected_skeleton_ratio = []


def evaluation_to_tf(tf_ops, session, controller, game_watcher, step,
                     type_of_data='test', average_reward=None, verbose=True):
    sk_to_stepratio = game_watcher.number_of_reached_goals/float(game_watcher.number_of_games)
    inside_neuron_to_stepratio = game_watcher.number_of_outside_steps/float(game_watcher.number_of_games)
    perc_of_skeleton_steps = np.mean(game_watcher.collected_skeleton_ratio)
    if average_reward is not None:
        summary_str1 = session.run(tf_ops['reward_%s'%type_of_data], {tf_ops['fl_placeholder']: average_reward})
        controller.summary_writer.add_summary(summary_str1, step)

    summary_str2 = session.run(tf_ops['skstepratio_%s'%type_of_data], {tf_ops['fl_placeholder']: sk_to_stepratio})
    summary_str3 = session.run(tf_ops['instepratio_%s'%type_of_data], {tf_ops['fl_placeholder']: inside_neuron_to_stepratio})
    summary_str4 = session.run(tf_ops['perc_skeletonsteps_%s'%type_of_data], {tf_ops['fl_placeholder']: perc_of_skeleton_steps})
    controller.summary_writer.add_summary(summary_str2, step)
    controller.summary_writer.add_summary(summary_str3, step)
    controller.summary_writer.add_summary(summary_str4, step)
    if verbose:
        print 'reached goals percentage', sk_to_stepratio
        print 'outside stepped ratio', inside_neuron_to_stepratio
        print 'number of skeleton eaten objects ratio', perc_of_skeleton_steps


def control(simulation, controller, ctrl_s,
            disable_training=True, img_save_path=None,
            summary_writer=None, session=None, tf_ops={},
            global_step=None, game_watcher=None, evaluate=False):
    # sense
    stop_sign = False
    reward = simulation.collect_reward()
    new_observation = simulation.observe()
    if simulation.hero.game_ends:
        if game_watcher is not None:
            if simulation.hero.reached_goal is True:
                game_watcher.number_of_reached_goals += 1
            if simulation.hero.stepped_outside is True:
                game_watcher.number_of_outside_steps += 1
        stop_sign = True
        if ctrl_s['last_observation'] is not None:
            controller.store(ctrl_s['last_observation'], ctrl_s['last_action'], reward, new_observation)
        return ctrl_s, stop_sign

    if global_step is not None:
        step = global_step
    else:
        step = simulation.hero.number_of_steps

    # if 'reward_test' in tf_ops and (step+1) % 1000==0 or evaluate:
    #     if disable_training:
    #         type_of_data = 'test'
    #     else:
    #         type_of_data = 'train'
    #
    #     if global_step is not None:
    #         step = global_step
    #     else:
    #         step = simulation.hero.number_of_steps
    #     average_reward = np.mean(simulation.collected_rewards)
    #     # print 'step %i; reward: %f'%(simulation.hero.number_of_steps, average_reward)
    #     print game_watcher.number_of_games
    #     # sk_to_stepratio = simulation.objects_eaten['skeleton']/float(simulation.hero.number_of_steps)
    #     if game_watcher is not None:
    #         sk_to_stepratio = game_watcher.number_of_reached_goals/float(game_watcher.number_of_games)
    #         inside_neuron_to_stepratio = game_watcher.number_of_outside_steps/float(game_watcher.number_of_games)
    #     else:
    #         sk_to_stepratio = simulation.objects_eaten['goal']/float(simulation.games_played)
    #         inside_neuron_to_stepratio = simulation.objects_eaten['outside_neuron']/float(simulation.games_played)
    #
    #     print step, reward, 'reward'
    #     print 'percentage of reached goals', sk_to_stepratio
    #     print 'percentage of outside steps', inside_neuron_to_stepratio
    #     summary_str1 = session.run(tf_ops['reward_%s'%type_of_data], {tf_ops['fl_placeholder']: average_reward})
    #     summary_str2 = session.run(tf_ops['skstepratio_%s'%type_of_data], {tf_ops['fl_placeholder']: sk_to_stepratio})
    #     summary_str3 = session.run(tf_ops['instepratio_%s'%type_of_data], {tf_ops['fl_placeholder']: inside_neuron_to_stepratio})
    #     controller.summary_writer.add_summary(summary_str1, step)
    #     controller.summary_writer.add_summary(summary_str2, step)
    #     controller.summary_writer.add_summary(summary_str3, step)

    # store last transition
    if ctrl_s['last_observation'] is not None:
        controller.store(ctrl_s['last_observation'], ctrl_s['last_action'], reward, new_observation)

    # act
    new_action = controller.action(new_observation)
    simulation.perform_action(new_action)
    if game_watcher is not None:
        game_watcher.actions.append(new_action)

    step_limit = 200
    # if simulation.hero.number_of_steps%step_limit == 0 and not disable_training:
    if simulation.hero.number_of_steps%step_limit == 0:
        # simulation.reinitialize_world()
        stop_sign = True
        ctrl_s['last_observation'] = None
        print "reinitialize world because step %ihas reached" %step_limit
        return ctrl_s, stop_sign

    #train
    if not disable_training:
        controller.training_step()

    # update current state as last state.
    ctrl_s['last_action'] = new_action
    ctrl_s['last_observation'] = new_observation
    return ctrl_s, stop_sign




def do_evaluation(simulation, session, controller, var_dic, global_step,
                  type_of_data='test', inference_outputpath=None, experience_collection={}):
    # Evaluate model on testdata
    trajectory_collection = []

    print 'running whole cube: ', type_of_data
    test_game_watcher = GameWatcher()
    # Go through all neurons once
    for nx_graph_id in simulation.nx_skeletons.nx_graph_dic.iterkeys():
        # print 'nx_graph_id', nx_graph_id
        stop_sign = False
        simulation.reinitialize_world(nx_graph_id=nx_graph_id, start_node=True,
                                      center_node=False,
                                      number_of_initial_steps=simulation.len_of_memory)

        nx_graph = simulation.nx_skeletons.nx_graph_dic[nx_graph_id]
        number_of_gt_skeleton = nx_graph.number_of_nodes()
        # while stop_sign is False and simulation.hero.number_of_steps < 200:
        while stop_sign is False and simulation.hero.number_of_steps < number_of_gt_skeleton + 20:
            experience_collection, stop_sign = control(simulation, controller,
                                  experience_collection,
                                  disable_training=True,
                                  tf_ops=var_dic, session=session,
                                             game_watcher=test_game_watcher)
        # print 'len of visit positions', len(simulation.hero.visited_positions)
        if len(simulation.hero.visited_positions) > 1:
            test_game_watcher.number_of_games += 1

            skeleton_ratio = simulation.objects_eaten['skeleton']/float(number_of_gt_skeleton)
            test_game_watcher.collected_skeleton_ratio.append(skeleton_ratio)
            trajectory = simulation.hero.visited_positions
            number_of_edges = len(trajectory)
            edgelist = zip(range(number_of_edges-1), range(1, number_of_edges))
            nx_skeleton = networkx_utils.NxSkeleton()
            nx_skeleton.initialize_from_edgelist(np.array(trajectory), edgelist)


            # Add a comment to those nodes with a specific attribute
            pos_to_node_id_dic = nx_skeleton.position_to_node_dic()
            for pos in simulation.hero.collected_eaten_obj['skeleton_hero_pos']:
                node_id_of_graph = pos_to_node_id_dic[tuple([pos.x, pos.y, pos.z])]
                nx_skeleton.nx_graph.node[node_id_of_graph]['comment'] = 'skeleton'


            # Mark start and endpoint
            first_pos = trajectory[0]
            last_pos = trajectory[-1]
            node_id_of_graph = pos_to_node_id_dic[first_pos]
            nx_skeleton.nx_graph.node[node_id_of_graph]['comment'] = 'start'
            node_id_of_graph = pos_to_node_id_dic[last_pos]
            nx_skeleton.nx_graph.node[node_id_of_graph]['comment'] = 'end'
            if simulation.hero.is_outside_world:
                print 'stepped outside world'
                # Check whether agent left world within the correct neuron
                seg_id_of_gt_neuron = simulation.volumetric_object_matrix[first_pos[0],
                                                                          first_pos[1], first_pos[2]]
                seg_id_of_agent = simulation.volumetric_object_matrix[last_pos[0],
                                                                          last_pos[1], last_pos[2]]
                if seg_id_of_gt_neuron == seg_id_of_agent:
                    last_neuron_comment = 'within_neuron_end'
                else:
                    last_neuron_comment = 'looser'

                nx_skeleton.nx_graph.node[node_id_of_graph]['comment'] = last_neuron_comment

            for pos in simulation.hero.collected_eaten_obj['endnode_hero_pos']:
                node_id_of_graph = pos_to_node_id_dic[tuple([pos.x, pos.y, pos.z])]
                nx_skeleton.nx_graph.node[node_id_of_graph]['comment'] = 'winner'

            nx_skeleton.scale_positions([1, 1, 1], additive=simulation.knossos_offset)

            if simulation.hero.reached_goal:
                nx_skeleton.nx_graph.name = 'winner'
            if simulation.hero.stepped_outside:
                nx_skeleton.nx_graph.name = 'looser'
            trajectory_collection.append(nx_skeleton)

    # print 'testing end'
    # print 'number of total games', test_game_watcher.number_of_games
    # print 'number of outside stepped games', test_game_watcher.number_of_outside_steps
    # print 'number of reached goals', test_game_watcher.number_of_reached_goals
    average_reward = np.mean(simulation.collected_rewards)
    evaluation_to_tf(var_dic, session, controller, test_game_watcher, step=global_step,
                     type_of_data=type_of_data, average_reward=average_reward)
    if inference_outputpath is not None:
        knossos_utils.from_nx_graphs_to_knossos([nx_skeleton.nx_graph
                                                 for nx_skeleton in trajectory_collection],
                                                inference_outputpath + '/cube_%i_knossos_%i.nml' %(simulation.cube_number,
                                                                                                   global_step))
    simulation.reinitialize_world()


def simulate_neuron_maze(session, simulation,
             controller = None,
             speed=1.0,
             disable_training=False,
             img_save_path=None, saver=None, store_path=None,
             test_simulation=None,
             simulation_training_dic=None,
             display_var=True, inference_outputpath=None):
    """Start the simulation. Performs three tasks

        - visualizes simulation in iPython notebook
        - advances simulator state
        - reports state to controller and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated ;-)
    controller: tr_lr.controller
        controller used
    fps: int
        frames per seconds to display;
        decrease for faster training.
    actions_per_simulation_second: int
        how many times perform_action is called per
        one second of simulation time
    speed: float
        executed <speed> seconds of simulation time
        per every second of real time
    disable_training: bool
        if true training_step is never called.
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """
    ctrl_s = {
        'last_observation': None,
        'last_action':      None,
    }
    ctrl_s_test = ctrl_s.copy()
    print ctrl_s, ctrl_s_test
    # Tensorflow values for monitoring game (that are not related to controller)
    fl_pl = tf.placeholder(tf.float32, name='fl_pl')
    var_dic = {}
    var_dic['fl_placeholder'] = fl_pl
    for type_of_data in ['train', 'test']:
        sum_reward = tf.scalar_summary('reward_%s'%type_of_data, fl_pl)
        sum_1 = tf.scalar_summary('ratioreachedgoalstototal_%s'%type_of_data, fl_pl)
        sum_2= tf.scalar_summary('ratiosteppedoutsideneurontototal_%s'%type_of_data, fl_pl)
        sum_3= tf.scalar_summary('perc_skeletonsteps_%s'%type_of_data, fl_pl)
        var_dic['reward_%s'%type_of_data] = sum_reward
        var_dic['skstepratio_%s'%type_of_data] = sum_1
        var_dic['instepratio_%s'%type_of_data] = sum_2
        var_dic['perc_skeletonsteps_%s'%type_of_data] = sum_3

    train_game_watcher = GameWatcher()
    print 'starting playing the game'
    global_step = 0
    while global_step < 1000000000:
        ctrl_s, stop_sign = control(simulation, controller, ctrl_s,
            disable_training=False, img_save_path=None,
            session=session, tf_ops=var_dic,
            global_step=global_step, game_watcher=train_game_watcher)
        if display_var:
            visualize(simulation)
        if stop_sign:
            # This means that the agent left the world or world was
            # reinitialized so that one can exchange simulations without problem
            # print 'taken actions', simulation.hero.taken_actions
            if simulation_training_dic is not None:
                simulation_key = random.choice(simulation_training_dic.keys())
                simulation = simulation_training_dic[simulation_key]
            simulation.reinitialize_world()
            train_game_watcher.number_of_games += 1

        if (global_step + 1) % 1000 == 0:
            # Store model
            print 'global step %i' %global_step
            if saver is not None and store_path is not None:
                saver.save(session, store_path, global_step=global_step)
            do_evaluation(test_simulation, session, controller, var_dic,
                          global_step, type_of_data='test',
                          inference_outputpath=inference_outputpath,
                          experience_collection=ctrl_s_test)
            for train_simulation in simulation_training_dic.values()[:1]:
                do_evaluation(train_simulation, session, controller, var_dic,
                  global_step, type_of_data='train', inference_outputpath=inference_outputpath,
                  experience_collection=ctrl_s)



        global_step += 1



def simulate_neuron_maze_inference(session, simulation,
             controller = None,
             speed=1.0,
             disable_training=False,
             img_save_path=None, saver=None, store_path=None,
             test_simulation=None,
             simulation_training_dic=None,
             display_var=True, inference_outputpath=None):
    """Start the simulation. Performs three tasks

        - visualizes simulation in iPython notebook
        - advances simulator state
        - reports state to controller and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated ;-)
    controller: tr_lr.controller
        controller used
    fps: int
        frames per seconds to display;
        decrease for faster training.
    actions_per_simulation_second: int
        how many times perform_action is called per
        one second of simulation time
    speed: float
        executed <speed> seconds of simulation time
        per every second of real time
    disable_training: bool
        if true training_step is never called.
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """
    ctrl_s = {
        'last_observation': None,
        'last_action':      None,
    }
    ctrl_s_test = ctrl_s.copy()
    print ctrl_s, ctrl_s_test
    # Tensorflow values for monitoring game (that are not related to controller)
    fl_pl = tf.placeholder(tf.float32, name='fl_pl')
    var_dic = {}
    var_dic['fl_placeholder'] = fl_pl
    for type_of_data in ['train', 'test']:
        sum_reward = tf.scalar_summary('reward_%s'%type_of_data, fl_pl)
        sum_1 = tf.scalar_summary('ratioreachedgoalstototal_%s'%type_of_data, fl_pl)
        sum_2= tf.scalar_summary('ratiosteppedoutsideneurontototal_%s'%type_of_data, fl_pl)
        sum_3= tf.scalar_summary('perc_skeletonsteps_%s'%type_of_data, fl_pl)
        var_dic['reward_%s'%type_of_data] = sum_reward
        var_dic['skstepratio_%s'%type_of_data] = sum_1
        var_dic['instepratio_%s'%type_of_data] = sum_2
        var_dic['perc_skeletonsteps_%s'%type_of_data] = sum_3

    train_game_watcher = GameWatcher()
    print 'starting playing the game'
    global_step = 0
    while global_step < 1000000000:
        ctrl_s, stop_sign = control(simulation, controller, ctrl_s,
            disable_training=True, img_save_path=None,
            session=session, tf_ops=var_dic,
            global_step=global_step, game_watcher=train_game_watcher)
        if display_var:
            visualize(simulation)
        if stop_sign:
            # This means that the agent left the world or world was
            # reinitialized so that one can exchange simulations without problem
            # print 'taken actions', simulation.hero.taken_actions
            if simulation_training_dic is not None:
                simulation_key = random.choice(simulation_training_dic.keys())
                simulation = simulation_training_dic[simulation_key]
            simulation.reinitialize_world()
            train_game_watcher.number_of_games += 1

        if (global_step + 1) % 1 == 0:
            # Store model
            print 'global step %i' %global_step
            if saver is not None and store_path is not None:
                saver.save(session, store_path, global_step=global_step)
            do_evaluation(test_simulation, session, controller, var_dic,
                          global_step, type_of_data='test',
                          inference_outputpath=inference_outputpath,
                          experience_collection=ctrl_s_test)
            for train_simulation in simulation_training_dic.values()[:1]:
                do_evaluation(train_simulation, session, controller, var_dic,
                  global_step, type_of_data='train', inference_outputpath=inference_outputpath,
                  experience_collection=ctrl_s)



        global_step += 1
