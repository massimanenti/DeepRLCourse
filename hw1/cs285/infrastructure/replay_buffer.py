from cs285.infrastructure.utils import *


class ReplayBuffer(object):
    """
    Defines a ReplayBuffer, in which information of past paths is stored.

    Attributes
    ----------
    max_size: int
        Max size of the buffer
    paths: list of dictionaries
        Past rollouts
    obs: numpy.ndarray
        Concatenated observation array of all the past rollouts
    acs: numpy.ndarray
        Concatenated actions array of all the past rollouts
    rews: numpy.ndarray
        Concatenated rewards array of all the past rollouts
    next_obs: numpy.ndarray
        Concatenated next_observation array of all the past rollouts
    terminals: numpy.ndarray
        Concatenated terminals array of all the past rollouts
        
    Methods
    -------
    add_rollouts:
        Add a new rollout
    """

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):
        # add new rollouts into our list of rollouts

        # append the list of dictionaries (i.e., paths) to the internal variable (i.e., self.paths)
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))
        # convert_listofrollouts takes a list of rollout dictionaries and return separate arrays, 
        # where each array is a concatenation of that array from across the rollouts.
        # So, paths is a list of dictionaries. Each dictionary has the following keys:
        # dict_keys(['observation', 'action', 'reward', 'next_observation', 'terminal'])
        # With convert_listofrollouts we create 5 long arrays (obs, act, rew, next_obs, ter) which contain
        # the info present in the list of dictionaries "path".

        if self.obs is None:
            # add all the elements
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            # concatenate the new elements
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]

            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]

            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

