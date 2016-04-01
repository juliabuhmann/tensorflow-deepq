from .karpathy_game   import KarpathyGame
from .double_pendulum import DoublePendulum
from .discrete_hill   import DiscreteHill
try:
    from .neuron_maze import NeuronMaze
except ImportError:
    print('module NeuronMaze could not be loaded')
