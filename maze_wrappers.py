import gymnasium as gym
import numpy as np
from collections import deque


class AntMazeGeodesicWrapper(gym.Wrapper):
    """
    AntMazeGeodesicWrapper replaces deceptive Euclidean straight-line distance
    with true topological corridor shortest-path distance (geodesic distance)
    using BFS over the maze graph, and awards a sparse completion bonus upon goal reach.

    Key Features:
    1. Monotonic Corridor Progress:
       Reward monotonically increases as the Ant navigates along corridors towards the exit.
       Moving towards dividing walls no longer increases reward, eliminating local traps.
    2. Dense Distance Penalty:
       r_dense = -min(1.0, d_geodesic / max_dist)
       Kept non-positive so that ending the episode at the goal is never worse than
       hovering just outside the goal radius (a positive per-step reward plus
       terminate-on-success makes hovering worth more than the success bonus).
    3. Goal Completion Bonus:
       When within goal radius (0.45m), awards success_bonus (default +50.0).
    4. Optional Early Termination:
       If terminate_on_success=True, marks terminated=True on goal reach to boost sample efficiency.
    """
    def __init__(
        self,
        env: gym.Env,
        reward_mode: str = "sparse",
        success_bonus: float = 50.0,
        max_corridor_dist: float = 25.0,
        terminate_on_success: bool = True
    ):
        super().__init__(env)
        self.reward_mode = reward_mode.lower()
        self.success_bonus = success_bonus
        self.max_dist = max_corridor_dist
        self.terminate_on_success = terminate_on_success
        self.has_succeeded = False
        self.prev_d_geo = None

        # Extract maze attributes
        self.maze = self.env.unwrapped.maze
        self.grid = np.array(self.maze.maze_map, dtype=int)
        self.open_cells = np.argwhere(self.grid == 0)

    def _get_valid_cell(self, row: int, col: int) -> tuple[int, int]:
        """Clamps coordinate to nearest open cell (0) in the maze grid."""
        r = int(np.clip(row, 0, self.grid.shape[0] - 1))
        c = int(np.clip(col, 0, self.grid.shape[1] - 1))
        if self.grid[r, c] == 0:
            return (r, c)
        dists = np.sum(np.abs(self.open_cells - np.array([r, c])), axis=1)
        nearest = self.open_cells[np.argmin(dists)]
        return (int(nearest[0]), int(nearest[1]))

    def _bfs_path(self, start_rc: tuple[int, int], goal_rc: tuple[int, int]) -> list[tuple[int, int]]:
        """Finds shortest corridor cell path using Breadth-First Search (BFS)."""
        if start_rc == goal_rc:
            return [start_rc]
        queue = deque([[start_rc]])
        visited = {start_rc}
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == goal_rc:
                return path
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1] and self.grid[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(path + [(nr, nc)])
        return [start_rc, goal_rc]

    def compute_geodesic_distance(self, p_start: np.ndarray, p_goal: np.ndarray) -> float:
        """
        Computes continuous piecewise-linear corridor distance through maze waypoints.
        """
        rc_start = self._get_valid_cell(*self.maze.cell_xy_to_rowcol(p_start))
        rc_goal = self._get_valid_cell(*self.maze.cell_xy_to_rowcol(p_goal))

        path = self._bfs_path(rc_start, rc_goal)
        if len(path) <= 1:
            return float(np.linalg.norm(p_start - p_goal))

        # Build sequence of continuous 2D waypoints: start -> cell centers -> goal
        pts = [p_start] + [self.maze.cell_rowcol_to_xy(rc) for rc in path[1:-1]] + [p_goal]
        dist = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
        return float(dist)

    def reset(self, **kwargs):
        self.has_succeeded = False
        self.prev_d_geo = None
        obs, info = self.env.reset(**kwargs)
        info["success"] = False
        info["success_bonus_awarded"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, dict):
            achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
            desired = np.asarray(obs["desired_goal"], dtype=np.float64)
        else:
            achieved = np.asarray(obs[:2], dtype=np.float64)
            desired = np.asarray(obs[2:4], dtype=np.float64)

        # 1. Compute true topological geodesic corridor distance
        d_geo = self.compute_geodesic_distance(achieved, desired)

        if self.reward_mode == "sparse":
            step_reward = 0.0
        elif self.reward_mode == "progress":
            step_reward = float(self.prev_d_geo - d_geo) if self.prev_d_geo is not None else 0.0
        else: # "dense"
            step_reward = -float(min(1.0, d_geo / self.max_dist))
        self.prev_d_geo = d_geo

        reward = step_reward

        # 2. Check goal touch
        dist_eucl = float(np.linalg.norm(achieved - desired))
        is_success = bool(info.get("success", False) or (dist_eucl <= 0.45))
        info["success"] = is_success
        info["geodesic_distance"] = d_geo

        # 3. Sparse completion bonus
        if is_success and not self.has_succeeded:
            reward += float(self.success_bonus)
            self.has_succeeded = True
            info["success_bonus_awarded"] = True
            if self.terminate_on_success:
                terminated = True
        else:
            info["success_bonus_awarded"] = False

        return obs, reward, terminated, truncated, info
