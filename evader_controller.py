
from FACL import FACL
import numpy as np

# This class inherits FACL and implements the :
# reward function
# state update
# saves the path the agent is taking in a given epoch
# resets the game after an epoch

class evader_controller(FACL):

    def __init__(self, state, max, min, num_mf):
        self.state = state.copy()
        self.path = state.copy()
        self.initial_position = state.copy()
        self.territory_coordinates = [0, 30]  # these will eventually be in the game class and passed into the actor
        self.r = 1 #radius of the territory
        self.v = 1.1  # unit velocity
        self.distance_away_from_terr_t_plus_1 = 0 #this gets set later
        self.distance_away_from_terr_t = self.distance_from_target()
        #self.distance_away_from_p2_t = self.distance_from_target([0, -5])
        self.reward_track =[] # to keep track of the rewards
        FACL.__init__(self, max, min, num_mf) #explicit call to the base class constructor

    def get_reward(self,coor):
        self.distance_away_from_terr_t_plus_1 = self.distance_from_target()
        if (abs(self.state[0]  - self.territory_coordinates[0]) <= self.r and abs(self.state[1] - self.territory_coordinates[1]) <= self.r):
            r = 0

        else:
            r = (self.distance_away_from_terr_t - self.distance_away_from_terr_t_plus_1)
        # print("reward", self.distance_away_from_terr_t, '-', self.distance_away_from_terr_t_plus_1, '=', r)
        self.distance_away_from_terr_t = self.distance_away_from_terr_t_plus_1
        self.update_reward_graph(r)
        return r

    def update_state(self):
        self.state[0] = self.state[0] + self.v * np.cos(self.u_t)
        self.state[1] = self.state[1] + self.v * np.sin(self.u_t)
        self.update_path(self.state)
        pass

    def reset(self):
        # Edited for each controller
        self.state = self.initial_position.copy() # set to self.initial_state, debug later???
        self.path = []
        self.path = self.initial_position.copy()
        self.reward_track = []
        self.distance_away_from_terr_t = self.distance_from_target()
        pass

    def update_path(self, state):

        self.path = np.vstack([self.path, state])
        pass

    def update_reward_graph(self, r):
        self.reward_track.append(r)

    def distance_from_target(self):
        distance_away_from_target = np.sqrt(
            (self.state[0] - self.territory_coordinates[0]) ** 2 + (self.state[1] - self.territory_coordinates[1]) ** 2)
        return distance_away_from_target

