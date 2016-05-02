import numpy as np
import random
import tensorflow as tf

from collections import deque

class DiscreteDeepQ(object):
    #TODO: Batch training and inference not efficiently implemented
    def __init__(self, observation_size,
                       num_actions,
                       observation_to_actions,
                       optimizer,
                       session,
                       random_action_probability=0.05,
                       exploration_period=1000,
                       store_every_nth=5,
                       train_every_nth=5,
                       minibatch_size=32,
                       discount_rate=0.95,
                       max_experience=30000,
                       target_network_update_rate=0.01,
                       clip_loss_function = False,
                       clip_reward = False,
                       replay_start_size= 1000,
                       summary_writer=None,
                       game_watcher=None):
        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_size : int
            length of the vector passed as observation
        num_actions : int
            number of actions that the model can execute
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, num_actions]
        optimizer: tf.solver.*
            optimizer for prediction error
        session: tf.Session
            session on which to execute the computation
        random_action_probability: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to random_action_probability over
            exploration_period
        store_every_nth: int
            to further decorrelate samples do not all
            transitions, but rather every nth transition.
            For example if store_every_nth is 5, then
            only 20% of all the transitions is stored.
        train_every_nth: int
            normally training_step is invoked every
            time action is executed. Depending on the
            setup that might be too often. When this
            variable is set set to n, then only every
            n-th time training_step is called will
            the training procedure actually be executed.
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        target_network_update_rate: float
            how much to update target network after each
            iteration. Let's call target_network_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        # memorize arguments
        self.observation_size          = observation_size
        self.num_actions               = num_actions

        self.q_network                 = observation_to_actions
        self.optimizer                 = optimizer
        self.s                         = session

        self.random_action_probability = random_action_probability
        self.exploration_period        = exploration_period
        self.store_every_nth           = store_every_nth
        self.train_every_nth           = train_every_nth
        self.minibatch_size            = minibatch_size
        self.discount_rate             = tf.constant(discount_rate)
        self.max_experience            = max_experience
        self.target_network_update_rate = \
                tf.constant(target_network_update_rate)
        self.replay_start_size = replay_start_size

        # deepq state
        self.actions_executed_so_far = 0
        self.experience = deque()

        self.iteration = 0
        self.summary_writer = summary_writer

        self.number_of_times_store_called = 0
        self.number_of_times_train_called = 0
        self.clip_loss_function = clip_loss_function
        self.clip_reward = clip_reward
        self.collected_prediction_errors = []
        self.game_watcher = game_watcher

        self.create_variables()

    def linear_annealing(self, n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n == total:
            print "exploration period over"
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)

    def create_variables(self):
        self.target_q_network    = self.q_network.copy(scope="target_network")

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            self.observation        = tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
            self.action_scores      = tf.identity(self.q_network(self.observation), name="action_scores")
            tf.histogram_summary("action_scores", self.action_scores)
            self.predicted_actions  = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        with tf.name_scope("estimating_future_rewards"):
            # FOR PREDICTING TARGET FUTURE REWARDS
            self.next_observation          = tf.placeholder(tf.float32, (None, self.observation_size), name="next_observation")
            self.next_observation_mask     = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
            self.next_action_scores        = tf.stop_gradient(self.target_q_network(self.next_observation))
            tf.histogram_summary("target_action_scores", self.next_action_scores)
            self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            reward_batch                   = tf.reduce_mean(self.rewards)
            tf.scalar_summary('reward_batch', reward_batch)



            target_values                  = tf.reduce_max(self.next_action_scores, reduction_indices=[1,]) * self.next_observation_mask
            self.future_rewards            = self.rewards + self.discount_rate * target_values

        with tf.name_scope("tensorboard_monitoring"):
            self.last_rewards               = tf.placeholder(tf.float32, (None,))
            last_rewards                    = tf.reduce_mean(self.last_rewards)
            tf.scalar_summary('last_rewards_mean', last_rewards)
            if self.game_watcher is not None:
                self.winning_games = tf.placeholder(tf.float32, (None))
                self.got_lost = tf.placeholder(tf.float32, (None))
                self.stepped_outside = tf.placeholder(tf.float32, (None))
                winning_games_op = tf.identity(self.winning_games)
                got_lost_op = tf.identity(self.got_lost)
                stepped_outside_op = tf.identity(self.stepped_outside)
                tf.scalar_summary('games_winning', winning_games_op)
                tf.scalar_summary('games_gotlost', got_lost_op)
                tf.scalar_summary('games_steppedoutside', stepped_outside_op)

                self.winning_games_last = tf.placeholder(tf.float32, (None))
                self.got_lost_last = tf.placeholder(tf.float32, (None))
                self.stepped_outside_last = tf.placeholder(tf.float32, (None))
                winning_games_op_last = tf.identity(self.winning_games_last)
                got_lost_op_last = tf.identity(self.got_lost_last)
                stepped_outside_op_last = tf.identity(self.stepped_outside_last)
                tf.scalar_summary('gameslast_winning', winning_games_op_last)
                tf.scalar_summary('gameslast_gotlost', got_lost_op_last)
                tf.scalar_summary('gameslast_steppedoutside', stepped_outside_op_last)




        with tf.name_scope("q_value_precition"):
            # FOR PREDICTION ERROR
            self.action_mask                = tf.placeholder(tf.float32, (None, self.num_actions), name="action_mask")
            self.masked_action_scores       = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
            temp_diff                       = self.masked_action_scores - self.future_rewards
            if self.clip_loss_function:
                self.prediction_error       = tf.reduce_mean(tf.square(tf.clip_by_value(temp_diff, -1.0, 1.0,
                                                                name='clipping_loss')))
            else:
                self.prediction_error           = tf.reduce_mean(tf.square(temp_diff), name='error_function')

            gradients                       = self.optimizer.compute_gradients(self.prediction_error)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # Add histograms for gradients.
            for grad, var in gradients:
                tf.histogram_summary(var.name, var)
                if grad:
                    tf.histogram_summary(var.name + '/gradients', grad)
            self.train_op                   = self.optimizer.apply_gradients(gradients)



        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        # summaries
        tf.scalar_summary("prediction_error", self.prediction_error)
        self.prediction_error_collection_tf = tf.placeholder(tf.float32)
        self.average_prediction_error = tf.reduce_mean(self.prediction_error_collection_tf)
        tf.scalar_summary("averaged_prediction_error", self.average_prediction_error)
        self.summarize = tf.merge_all_summaries()
        self.no_op1    = tf.no_op()

    def action(self, observation, randomness=True):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert len(observation.shape) == 1, \
                "Action is performed based on single observation."

        self.actions_executed_so_far += 1
        exploration_p = self.linear_annealing(self.actions_executed_so_far,
                                              self.exploration_period,
                                              1.0,
                                              self.random_action_probability)


        if random.random() < exploration_p and randomness:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.s.run(self.predicted_actions, {self.observation: observation[np.newaxis,:]})[0]

    def store(self, observation, action, reward, newobservation):
        """Store experience, where starting with observation and
        execution action, we arrived at the newobservation and got thetarget_network_update
        reward reward

        If newstate is None, the state/action pair is assumed to be terminal
        """

        if self.number_of_times_store_called % self.store_every_nth == 0:
            self.experience.append((observation, action, reward, newobservation))
            if len(self.experience) > self.max_experience:
                self.experience.popleft()
        self.number_of_times_store_called += 1

    def training_step(self, only_tensorboard=False):
        """Pick a self.minibatch_size exeperiences from reply buffer
        and backpropage the value function.
        """
        # if self.number_of_times_train_called % self.train_every_nth == 0:
        if len(self.experience) <  self.minibatch_size or len(self.experience) \
                < self.replay_start_size:
            # print 'returning'
            return
            # if len(self.experience)< self.minibatch_size:
            #         return


        # sample experience.
        samples   = random.sample(range(len(self.experience)), self.minibatch_size)
        samples   = [self.experience[i] for i in samples]

        # bach states
        states         = np.empty((len(samples), self.observation_size))
        newstates      = np.empty((len(samples), self.observation_size))
        action_mask    = np.zeros((len(samples), self.num_actions))

        newstates_mask = np.empty((len(samples),))
        rewards        = np.empty((len(samples),))

        for i, (state, action, reward, newstate) in enumerate(samples):
            # Clip the reward to be in the range of -1 and 1 as suggested in the paper
            if self.clip_reward:
                reward = np.clip(reward, -1.0, 1.0)
            states[i] = state
            action_mask[i] = 0
            action_mask[i][action] = 1
            rewards[i] = reward
            if newstate is not None:
                newstates[i] = newstate
                newstates_mask[i] = 1
            else:
                newstates[i] = 0
                newstates_mask[i] = 0

        monitor_interval = 1000
        calculate_summaries = self.iteration % monitor_interval == 0 and \
                self.summary_writer is not None

        experiences = [0]
        if calculate_summaries:
            length_of_experience = len(self.experience)
            sample_size = 500

            if length_of_experience > sample_size:
                experiences = [self.experience[i][2] for i in range(-sample_size, -1)]


        feed_dict = {
            self.observation:            states,
            self.next_observation:       newstates,
            self.next_observation_mask:  newstates_mask,
            self.action_mask:            action_mask,
            self.rewards:                rewards,
            self.prediction_error_collection_tf: self.collected_prediction_errors[-monitor_interval:],
            self.last_rewards: np.array(experiences)
        }

        if self.game_watcher is not None:
            game_watcher = self.game_watcher
            number_of_total_games = float(game_watcher.number_of_games)
            if number_of_total_games == 0:
                number_of_total_games = 1
            feed_dict[self.winning_games] = game_watcher.number_of_reached_goals/number_of_total_games
            feed_dict[self.got_lost] = game_watcher.number_of_lost_games/number_of_total_games
            feed_dict[self.stepped_outside] = game_watcher.number_of_outside_steps/number_of_total_games
            # print game_watcher.number_of_reached_goals/number_of_total_games
            # Number encodes whether game was lost or won ...
            if len(game_watcher.collected_game_identity) >= monitor_interval:
                game_identities  = game_watcher.collected_game_identity[-monitor_interval:]
                number_of_reached_goals =game_identities.count(1)
                number_of_lost_games = game_identities.count(3)
                number_of_outside_steps = game_identities.count(2)
                feed_dict[self.winning_games_last] = number_of_reached_goals/float(monitor_interval)
                feed_dict[self.got_lost_last] = number_of_lost_games/float(monitor_interval)
                feed_dict[self.stepped_outside_last] = number_of_outside_steps/float(monitor_interval)
            else:
                feed_dict[self.winning_games_last] = 0.
                feed_dict[self.got_lost_last] = 0.
                feed_dict[self.stepped_outside_last] = 0.



        if only_tensorboard:
            summary_str = self.s.run(self.summarize, feed_dict)
            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.iteration)
        else:
            cost, _, summary_str = self.s.run([
                self.prediction_error,
                self.train_op,
                self.summarize if calculate_summaries else self.no_op1,
            ], feed_dict)

            self.s.run(self.target_network_update)

            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.iteration)
            self.number_of_times_train_called += 1
            self.collected_prediction_errors.append(cost)

        self.iteration += 1






