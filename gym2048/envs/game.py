import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from io import StringIO

right_action_results = np.zeros(shape=(18**4,), dtype=np.int32)
right_action_scores = np.zeros(shape=(18**4,), dtype=np.int32)

class TwentyFortyeight(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        self.square_size = 100 # The size of the square 
        self.window_size = 512 # The size of the PyGame window

        self.action_space = spaces.Discrete(4)
        self._tiles = np.zeros(shape=(4,4), dtype=np.int32)
        self._is_legal_actions = [False for _ in range(4)]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._tiles = np.zeros(shape=(4,4), dtype=np.int32)
        self._random_spawn()
        self._random_spawn()

        for action in range(4):
            self._is_legal_actions[action] = self._is_changed_by(action)

        return self._get_obs(), 0
    
    def step(self, action):
        if action == 0:
            row_0 = self._row_to_hash(self._tiles[0])
            row_1 = self._row_to_hash(self._tiles[1])
            row_2 = self._row_to_hash(self._tiles[2])
            row_3 = self._row_to_hash(self._tiles[3])

            result_row_0 = right_action_results[row_0]
            result_row_1 = right_action_results[row_1]
            result_row_2 = right_action_results[row_2]
            result_row_3 = right_action_results[row_3]

            self._tiles[0][0] = result_row_0 % 18
            self._tiles[0][1] = result_row_0 // 18 % 18
            self._tiles[0][2] = result_row_0 // (18**2) % 18
            self._tiles[0][3] = result_row_0 // (18**3) % 18
            self._tiles[1][0] = result_row_1 % 18
            self._tiles[1][1] = result_row_1 // 18 % 18
            self._tiles[1][2] = result_row_1 // (18**2) % 18
            self._tiles[1][3] = result_row_1 // (18**3) % 18
            self._tiles[2][0] = result_row_2 % 18
            self._tiles[2][1] = result_row_2 // 18 % 18
            self._tiles[2][2] = result_row_2 // (18**2) % 18
            self._tiles[2][3] = result_row_2 // (18**3) % 18
            self._tiles[3][0] = result_row_3 % 18
            self._tiles[3][1] = result_row_3 // 18 % 18
            self._tiles[3][2] = result_row_3 // (18**2) % 18
            self._tiles[3][3] = result_row_3 // (18**3) % 18

            reward = right_action_scores[row_0] + right_action_scores[row_1] + right_action_scores[row_2] + right_action_scores[row_3]

        elif action == 1:
            col_0 = self._row_to_hash(self._tiles[:,0])
            col_1 = self._row_to_hash(self._tiles[:,1])
            col_2 = self._row_to_hash(self._tiles[:,2])
            col_3 = self._row_to_hash(self._tiles[:,3])

            result_col_0 = right_action_results[col_0]
            result_col_1 = right_action_results[col_1]
            result_col_2 = right_action_results[col_2]
            result_col_3 = right_action_results[col_3]

            self._tiles[0][0] = result_col_0 % 18
            self._tiles[1][0] = result_col_0 // 18 % 18
            self._tiles[2][0] = result_col_0 // (18**2) % 18
            self._tiles[3][0] = result_col_0 // (18**3) % 18
            self._tiles[0][1] = result_col_1 % 18
            self._tiles[1][1] = result_col_1 // 18 % 18
            self._tiles[2][1] = result_col_1 // (18**2) % 18
            self._tiles[3][1] = result_col_1 // (18**3) % 18
            self._tiles[0][2] = result_col_2 % 18
            self._tiles[1][2] = result_col_2 // 18 % 18
            self._tiles[2][2] = result_col_2 // (18**2) % 18
            self._tiles[3][2] = result_col_2 // (18**3) % 18
            self._tiles[0][3] = result_col_3 % 18
            self._tiles[1][3] = result_col_3 // 18 % 18
            self._tiles[2][3] = result_col_3 // (18**2) % 18
            self._tiles[3][3] = result_col_3 // (18**3) % 18

            reward = right_action_scores[col_0] + right_action_scores[col_1] + right_action_scores[col_2] + right_action_scores[col_3]

        elif action == 2:
            row_0 = self._row_to_hash(self._tiles[0][::-1])
            row_1 = self._row_to_hash(self._tiles[1][::-1])
            row_2 = self._row_to_hash(self._tiles[2][::-1])
            row_3 = self._row_to_hash(self._tiles[3][::-1])

            result_row_0 = right_action_results[row_0]
            result_row_1 = right_action_results[row_1]
            result_row_2 = right_action_results[row_2]
            result_row_3 = right_action_results[row_3]

            self._tiles[0][3] = result_row_0 % 18
            self._tiles[0][2] = result_row_0 // 18 % 18
            self._tiles[0][1] = result_row_0 // (18**2) % 18
            self._tiles[0][0] = result_row_0 // (18**3) % 18
            self._tiles[1][3] = result_row_1 % 18
            self._tiles[1][2] = result_row_1 // 18 % 18
            self._tiles[1][1] = result_row_1 // (18**2) % 18
            self._tiles[1][0] = result_row_1 // (18**3) % 18
            self._tiles[2][3] = result_row_2 % 18
            self._tiles[2][2] = result_row_2 // 18 % 18
            self._tiles[2][1] = result_row_2 // (18**2) % 18
            self._tiles[2][0] = result_row_2 // (18**3) % 18
            self._tiles[3][3] = result_row_3 % 18
            self._tiles[3][2] = result_row_3 // 18 % 18
            self._tiles[3][1] = result_row_3 // (18**2) % 18
            self._tiles[3][0] = result_row_3 // (18**3) % 18

            reward = right_action_scores[row_0] + right_action_scores[row_1] + right_action_scores[row_2] + right_action_scores[row_3]
        
        elif action == 3:
            col_0 = self._row_to_hash(self._tiles[:,0][::-1])
            col_1 = self._row_to_hash(self._tiles[:,1][::-1])
            col_2 = self._row_to_hash(self._tiles[:,2][::-1])
            col_3 = self._row_to_hash(self._tiles[:,3][::-1])

            result_col_0 = right_action_results[col_0]
            result_col_1 = right_action_results[col_1]
            result_col_2 = right_action_results[col_2]
            result_col_3 = right_action_results[col_3]

            self._tiles[3][0] = result_col_0 % 18
            self._tiles[2][0] = result_col_0 // 18 % 18
            self._tiles[1][0] = result_col_0 // (18**2) % 18
            self._tiles[0][0] = result_col_0 // (18**3) % 18
            self._tiles[3][1] = result_col_1 % 18
            self._tiles[2][1] = result_col_1 // 18 % 18
            self._tiles[1][1] = result_col_1 // (18**2) % 18
            self._tiles[0][1] = result_col_1 // (18**3) % 18
            self._tiles[3][2] = result_col_2 % 18
            self._tiles[2][2] = result_col_2 // 18 % 18
            self._tiles[1][2] = result_col_2 // (18**2) % 18
            self._tiles[0][2] = result_col_2 // (18**3) % 18
            self._tiles[3][3] = result_col_3 % 18
            self._tiles[2][3] = result_col_3 // 18 % 18
            self._tiles[1][3] = result_col_3 // (18**2) % 18
            self._tiles[0][3] = result_col_3 // (18**3) % 18

            reward = right_action_scores[col_0] + right_action_scores[col_1] + right_action_scores[col_2] + right_action_scores[col_3]
        
        if self._is_legal_actions[action]:
            self._random_spawn()

        terminated = True
        legal_actions = []
        for next_action in range(4):
            self._is_legal_actions[next_action] = self._is_changed_by(next_action)
            if self._is_changed_by(next_action):
                terminated = False
                legal_actions.append(next_action)

        return self._get_obs(), reward, terminated, False, legal_actions

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
            )
            return

        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def __str__(self):
        s = ""
        for x in range(4):
            for y in range(4):
                s += str(self._get_num(x, y))
                s += ' '
            s += '\n'
        return s

    def _get_obs(self):
        return self._tiles.copy()
    
    def _get_num(self, x, y):
        assert 0 <= x <= 3 and 0 <= y <= 3
        return self._tiles[x][y]
    
    def _is_empty(self, x, y):
        return self._get_num(x, y) == 0
        
    def _spawn(self, x, y, num):
        assert 0 <= x <= 3 and 0 <= y <= 3
        assert 0 <= num <= 18
        assert self._is_empty(x, y)
        self._tiles[x][y] = num
    
    def _random_spawn(self):
        empty__tiles_x, empty__tiles_y = np.where(self._tiles == 0)
        idx = np.random.choice(len(empty__tiles_x))
        self._tiles[empty__tiles_x[idx]][empty__tiles_y[idx]] = 1 if np.random.random() > 0.1 else 2
    
    def _set_legal_actions(self):
        for action in range(4):
            self._is_legal_actions[action] = self._is_changed_by(action)

    def _row_to_hash(self, array: np.ndarray):
        return array[0] + array[1] * 18 + array[2] * (18**2) + array[3] * (18**3) 
    
    @staticmethod
    def _calculate_right_action():
        for num_0 in range(18):
            for num_1 in range(18):
                for num_2 in range(18):
                    for num_3 in range(18):

                        key = num_0 + num_1 * 18 + num_2 * (18**2) + num_3 * (18**3)

                        score_1 = 0
                        score_2 = 0
                        row_result = [num_0, num_1, num_2, num_3]
                        target_col = 3

                        for col in reversed(range(3)):
                            if row_result[col] == 0:
                                continue
                            if row_result[target_col] == 0:
                                row_result[target_col] = row_result[col]
                                row_result[col] = 0
                            else:
                                if row_result[col] == row_result[target_col]:
                                    row_result[target_col] += 1
                                    row_result[col] = 0
                                    score_2 = score_1
                                    score_1 = row_result[target_col]
                                    target_col -= 1
                                else:
                                    target_col -= 1
                                    if target_col != col:
                                        row_result[target_col] = row_result[col]
                                        row_result[col] = 0
            
                        result_key = row_result[0] + row_result[1] * 18 + row_result[2] * (18**2) + row_result[3] * (18**3)

                        right_action_results[key] = result_key
                        right_action_scores[key] = (not (score_1 == 0)) * (1<<score_1) + (not (score_2 == 0)) * (1<<score_2)

    def _is_changed_by_right(self, _tiles):
        row_0 = self._row_to_hash(_tiles[0])
        row_1 = self._row_to_hash(_tiles[1])
        row_2 = self._row_to_hash(_tiles[2])
        row_3 = self._row_to_hash(_tiles[3])

        result_row_0 = right_action_results[row_0]
        result_row_1 = right_action_results[row_1]
        result_row_2 = right_action_results[row_2]
        result_row_3 = right_action_results[row_3]

        if row_0 != result_row_0:
            return True
        if row_1 != result_row_1:
            return True
        if row_2 != result_row_2:
            return True
        if row_3 != result_row_3:
            return True
        return False
    
    def _is_changed_by(self, action):
        if action == 0:
            return self._is_changed_by_right(self._tiles)
        elif action == 1:
            return self._is_changed_by_right(np.rot90(self._tiles))
        elif action == 2:
            return self._is_changed_by_right(np.rot90(self._tiles, 2))
        elif action == 3:
            return self._is_changed_by_right(np.rot90(self._tiles, 3))
        else:
            return False
    
    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))


        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
if __name__ == "__main__":
    TwentyFortyeight._calculate_right_action()
    env = TwentyFortyeight(render_mode="rgb_array")
    obs, _ = env.reset()
    while True:
        action = int(input())
        obs, r, t, _, _ = env.step(action)
        print(env)
        if t:
            break