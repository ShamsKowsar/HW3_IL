class GridWorldEnv(gym.Env):
    global SEED
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def check_close(self, ob1, ob2):
        if (ob1[1] == ob2[1] and abs(ob1[0] - ob2[0]) == 1) or (
            ob1[0] == ob2[0] and abs(ob1[1] - ob2[1]) == 1
        ):
            return True

    def is_path_possible(self, grid_size, start, goal, obstacles):
        grid = [[0] * grid_size for _ in range(grid_size)]
        grid[start[0]][start[1]] = 1
        queue = deque([start])
        movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            curr_pos = queue.popleft()

            if curr_pos == goal:
                return True

            for move in movements:
                new_pos = (curr_pos[0] + move[0], curr_pos[1] + move[1])

                if 0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size:
                    if grid[new_pos[0]][new_pos[1]] == 0 and new_pos not in obstacles:
                        grid[new_pos[0]][new_pos[1]] = 1
                        queue.append(new_pos)

        return False

    def check_validity(self, obstacle, obstacles, size):
        count = 0
        for ob in obstacles:
            if ob[0] == 0 or ob[1] == 0 or ob[0] == size - 1 or ob[1] == size-1:
                count += 1
        if (
            obstacle[0] == 0
            or obstacle[1] == 0
            or obstacle[0] == size - 1
            or obstacle[1] == size-1
        ):
            count += 1
        if count > 2:
            return False
        counts = np.zeros(len(obstacles))
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                if self.check_close(obstacles[i], obstacles[j]):
                    counts[i] += 1
                    counts[j] += 1
        new_ob_count=0
        for i in range(len(obstacles)):
            if self.check_close(obstacles[i], obstacle):
                counts[i] += 1
                new_ob_count+=1

        # print(counts)
        # print(new_ob_count)
        # print('-------------')
        if new_ob_count>1:
          return False

        for _ in counts:
            if _ > 1:
                return False
        return True
    def draw_map(self,grid_size, start, end, obstacles):
      # Create an empty grid
      grid = np.zeros((grid_size,grid_size))

      # Mark the start cell with value 1
      grid[start[0], start[1]] = 1

      # Mark the end cell with value 2
      grid[end[0], end[1]] = 2

      # Mark obstacle cells with value 3
      # print(grid)
      for obstacle in obstacles:
          # print(obstacle)
          grid[obstacle[0], obstacle[1]] = 3

      cmap = plt.cm.get_cmap('coolwarm', 4)  # Choose a different colormap
      bounds = np.arange(5) - 0.5
      norm = plt.Normalize(bounds.min(), bounds.max())

      # Create a figure and axis
      fig, ax = plt.subplots()

      # Draw the grid with colors
      ax.imshow(grid, cmap=cmap, interpolation='none', norm=norm)

      # Remove grid lines
      ax.set_xticks([])
      ax.set_yticks([])

      # Add index labels to each cell
      for i in range(grid_size):
          for j in range(grid_size):
              ax.text(j, i, str(i * grid_size + j), ha='center', va='center', color='black')

      # Show the plot
      plt.show()

    def generate_map(self, size, num_obstacles):
        agent_and_goal_positions = [
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
            [(5, 5), (5, 4), (5, 3), (4, 5), (4, 4), (3, 5)],
        ]
        index = random.randint(0, 1)
        free_cells = []
        start = agent_and_goal_positions[index][
            random.randint(0, len(agent_and_goal_positions[index]) - 1)
        ]
        end = agent_and_goal_positions[1 - index][
            random.randint(0, len(agent_and_goal_positions[1 - index]) - 1)
        ]
        for i in range(size):
                for j in range(size):
                    if (i == start[0] and j == start[1]) or (
                        i == end[0] and j == end[1]
                    ):
                        continue
                    else:
                        free_cells.append([i, j])
        free_cells_1=list(free_cells)
        while True:
            for i in range(size):
                for j in range(size):
                    if (i == start[0] and j == start[1]) or (
                        i == end[0] and j == end[1]
                    ):
                        continue
                    else:
                        free_cells.append([i, j])
            free_cells = random.sample(free_cells, len(free_cells))
            obstacles = []
            while len(obstacles) < 8 and len(free_cells) > 0:
                obstacle = free_cells[0]
                new_obstacles = list(obstacles)
                new_obstacles.append(obstacle)
                if self.check_validity(
                    obstacle, obstacles, size
                ) and self.is_path_possible(size, start, end, new_obstacles) and obstacle not in obstacles:
                    obstacles.append(obstacle)
                free_cells.pop(0)
            if len(obstacles) == 8:
                self.draw_map(self.size,start,end,obstacles)
                return [start, end, obstacles,free_cells]
    def get_state_size(self):
      return self.size**2
    def get_action_size(self):
      # TODO
      return 4
    def convert_location_to_state(self,location):
      # print(location)
      return location[0]*self.size+location[1]
    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.count_of_obstacles=8
        self.reward=0
        self.health=100
        self.battery=100
        self.start,self.end,self.obstacles,self.free_cells=self.generate_map(self.size,self.count_of_obstacles)
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0,size-1, shape=(2,), dtype=int),
                "target": spaces.Box(0,size-1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):

        super().reset(seed=SEED)
        self.battery=100
        self.health=100


        # Choose the agent's location uniformly at random
        self._agent_location = np.array(self.start)
        self.free_cells=self.free_cells
        # print(self.free_cells)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array(self.end)
        self.obstacles=self.obstacles
        observation = self._get_obs()
        info = self._get_info()
        self.reward=0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    def give_reward(self,action):
      # action=0: hit obstacle
      # action=1: move between cells
      # action=2: move to target cell
      if action==0:
        reward=np.random.normal(loc=-1, scale=0.5)
        # print(f'hit the obstacle and get reward={reward}')

        return reward
      if action==1:
        reward=np.random.normal(loc=-0.5, scale=0.25)
        # print(f'move between cells and get reward={reward}')
        return reward
      if action==2:
        reward=np.random.normal(loc=25, scale=5)
        # print(f'move to target cell and get reward={reward}')
        return reward
        # return np.random.normal(loc=25, scale=5)
    def is_obstacle(self,location):
      for _ in self.obstacles:
        if _[0]==location[0] and _[1]==location[1]:
          return True
      return False
    def step(self, action):
        terminated=False
        direction = self._action_to_direction[action]
        action_with_wind_prob=random.uniform(0,1)

        if(0.8<action_with_wind_prob<0.9):
          direction*=-1
        elif action_with_wind_prob>0.9:
          direction=[0,0]
        self.battery-=np.random.normal(loc=0.35, scale=0.15)
        new_location=np.clip(self._agent_location + direction, 0, self.size - 1)
        if new_location[0]== self._target_location[0] and new_location[1]== self._target_location[1] :
          # print(f'new_location={new_location}')
          terminated=True
          reward=self.give_reward(2)
        elif self.is_obstacle(new_location):
          reward=self.give_reward(0)
          self.health-=np.random.normal(loc=0.2, scale=0.2)

        else:
          reward=self.give_reward(1)
          self._agent_location=np.array(new_location)




        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # print(f'{self._agent_location}-{self._target_location}')
        observation = self._get_obs()
        # print(observation)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
