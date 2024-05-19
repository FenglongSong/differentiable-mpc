import numpy as np
from os import path
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from stable_baselines3.common.env_checker import check_env


INIT_X_MAX = 1.0
INIT_V_MAX = 1.0


class DoubleIntegratorEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env for a (continuous-time) double integrator.
    The example is adapted from here: https://github.com/araffin/rl-tutorial-jnrr19?tab=readme-ov-file
    """

    # metadata = {"render_modes": ["console"]}

    def __init__(
            self, 
            mass: float = 1.0, 
            damping: float = 0., 
            max_force: float = 1.,
            dt: float = 1e-2, 
            render_mode: Optional[str] = None
        ):
        super().__init__()
        
        self.mass = mass
        self.damping = damping
        self.max_force = max_force
        self.dt = dt
        self.state = np.zeros(2)

        # self.render_mode = render_mode
        # self.screen_dim = 500
        # self.screen = None
        # self.clock = None
        # self.isopen = True


        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e3*np.ones(2), high=1e3*np.ones(2), dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        if options is None:
            high = np.array([INIT_X_MAX, INIT_V_MAX])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else INIT_X_MAX
            v = options.get("v_init") if "v_init" in options else INIT_V_MAX
            x = utils.verify_number_and_cast(x)
            v = utils.verify_number_and_cast(v)
            high = np.array([x, v])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        # if self.render_mode == "human":
        #     self.render()

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return self.state.astype(np.float32), {}  # empty info dict

    def step(self, action):
        force = np.clip(action, -self.max_force, self.max_force)[0]
        pos, vel = self.state[0], self.state[1]

        cost = 100. * pos**2 + vel**2 + 0.01 * force**2
        reward = -cost

        pos += vel * self.dt
        vel += (force - self.damping * vel) / self.mass * self.dt
        self.state = np.array([pos, vel])

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            self.state.astype(np.float32),
            reward,
            False,
            False,
            info,
        )

    # def render(self):
    #     if self.render_mode is None:
    #         assert self.spec is not None
    #         gym.logger.warn(
    #             "You are calling render method without specifying any render mode. "
    #             "You can specify the render_mode at initialization, "
    #             f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
    #         )
    #         return

    #     try:
    #         import pygame
    #         from pygame import gfxdraw
    #     except ImportError as e:
    #         raise DependencyNotInstalled(
    #             "pygame is not installed, run `pip install gymnasium[classic-control]`"
    #         ) from e

    #     if self.screen is None:
    #         pygame.init()
    #         if self.render_mode == "human":
    #             pygame.display.init()
    #             self.screen = pygame.display.set_mode(
    #                 (self.screen_dim, self.screen_dim)
    #             )
    #         else:  # mode in "rgb_array"
    #             self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
    #     if self.clock is None:
    #         self.clock = pygame.time.Clock()

    #     self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
    #     self.surf.fill((255, 255, 255))

    #     bound = 2.2
    #     scale = self.screen_dim / (bound * 2)
    #     offset = self.screen_dim // 2

    #     rod_length = 1 * scale
    #     rod_width = 1 * scale
    #     l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
    #     coords = [(l, b), (l, t), (r, t), (r, b)]
    #     transformed_coords = []
    #     for c in coords:
    #         c = pygame.math.Vector2(c)
    #         c = (c[0] + offset, c[1] + offset)
    #         transformed_coords.append(c)
    #     gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
    #     gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

    #     gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
    #     gfxdraw.filled_circle(
    #         self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
    #     )

    #     rod_end = (rod_length, 0)
    #     rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
    #     rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
    #     gfxdraw.aacircle(
    #         self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    #     )
    #     gfxdraw.filled_circle(
    #         self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    #     )

    #     # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
    #     # img = pygame.image.load(fname)
    #     # if self.last_u is not None:
    #     #     scale_img = pygame.transform.smoothscale(
    #     #         img,
    #     #         (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
    #     #     )
    #     #     is_flip = bool(self.last_u > 0)
    #     #     scale_img = pygame.transform.flip(scale_img, is_flip, True)
    #     #     self.surf.blit(
    #     #         scale_img,
    #     #         (
    #     #             offset - scale_img.get_rect().centerx,
    #     #             offset - scale_img.get_rect().centery,
    #     #         ),
    #     #     )

    #     # drawing axle
    #     gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
    #     gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

    #     self.surf = pygame.transform.flip(self.surf, False, True)
    #     self.screen.blit(self.surf, (0, 0))
    #     if self.render_mode == "human":
    #         pygame.event.pump()
    #         self.clock.tick(self.metadata["render_fps"])
    #         pygame.display.flip()

    #     else:  # mode == "rgb_array":
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
    #         )

    # def close(self):
    #     if self.screen is not None:
    #         import pygame

    #         pygame.display.quit()
    #         pygame.quit()
    #         self.isopen = False

    def close(self):
        pass


def main():
    env = DoubleIntegratorEnv()
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True, skip_render_check=False)


if __name__ == "__main__":
    main()
