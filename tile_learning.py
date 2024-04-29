import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        num_tiles = (state_high - state_low) / tile_width
        self.num_tiles = np.ceil(num_tiles).astype(int) + 1

        #make buckets and then flatten into weights
        sizes = np.insert(np.array(self.num_tiles).astype(int),0, int(num_tilings)) 
        self.weights = np.zeros(shape=tuple(sizes), dtype=np.float64)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return (self.num_actions * self.num_tilings * np.prod(self.num_tiles)).astype(int)


    def encodeState(self, s, a):
        all_tilings = np.zeros(self.feature_vector_len())
        for i in range(self.num_tilings):
            start = self.state_low - (i/self.num_tilings)*self.tile_width
            adjusted_state = (s - start) / (self.state_high - self.state_low)
            curr_tile = np.floor(adjusted_state * self.num_tiles)

            #make sure tile is bounded correctly
            curr_tile = np.maximum(curr_tile, np.zeros(shape=curr_tile.shape))
            curr_tile = np.minimum(curr_tile, np.full(shape=curr_tile.shape, fill_value=(self.num_tiles-1))).astype(int)
            index = np.ravel_multi_index([i,curr_tile[0],curr_tile[1],a], dims=[self.num_tilings,self.num_tiles[0],self.num_tiles[1],self.num_actions])
            all_tilings[index] = 1
        return all_tilings

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.feature_vector_len())
        else:
            return self.encodeState(s, a)

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros(X.feature_vector_len())

    for i in range(num_episode):
        state = env.reset()
        done = False
        action = epsilon_greedy_policy(state, done, w)
        x = X.__call__(state, done, action)
        z = np.zeros(X.feature_vector_len())
        Q_old = 0
        
        while not done:
            state, reward, done, info = env.step(action)
            next_action = epsilon_greedy_policy(state, done, w)
            x_prime = X(state, done, next_action)
            w_T = w.reshape(-1,1)
            Q = np.dot(x,w_T)
            Q_prime = np.dot(x_prime, w_T)
            delta = reward + gamma*Q_prime - Q 
            z_T = z.reshape(-1,1)
            z = gamma*lam*z + (1 - alpha*gamma*lam*np.dot(x, z_T))*x
            w = w + alpha*(delta + Q - Q_old)*z - alpha*(Q - Q_old)*x
            Q_old = Q_prime
            x = x_prime
            action = next_action
    return w