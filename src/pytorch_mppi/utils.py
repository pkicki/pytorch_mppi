import jax
import numpy as np
from copy import deepcopy
import gymnasium as gym
from brax import envs
from brax.io.image import render
from brax.io import html
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
import torch
from car_env.envs.vec_env import SingleTrackVecEnv

from dynamics_models.acrobot import Acrobot
from dynamics_models.go1 import Go1
from dynamics_models.half_cheetah import HalfCheetah
from dynamics_models.hopper import Hopper
from dynamics_models.humanoid import Humanoid, HumanoidBrax
from dynamics_models.humanoid_standup import HumanoidStandup
from dynamics_models.pendulum import Pendulum
from dynamics_models.single_track import SingleTrack
from dynamics_models.swimmer import Swimmer
from dynamics_models.walker_2d import Walker2D

class EnvWrapper:
    def __init__(self, env, if_render=False):
        self.env = env
        self.state = []
        self.if_render = if_render
        if "brax" in str(type(self.env)):
            self._step = jax.jit(self.env.step)
            self.reset()
            self._step(self.state[-1], jax.numpy.zeros(self.env.action_size))
        #if "brax" in str(type(self.env)) and self.if_render:
        #    self.mj_data = mujoco.MjData(self.env.sys.mj_model)
        #    self.mujoco_renderer = MujocoRenderer(
        #        self.env.sys.mj_model,
        #        self.mj_data,
        #        width=640,
        #        height=480,
        #    )

    def step(self, action):
        if "gym" in str(type(self.env)):
            return self.env.step(action)
        elif "brax" in str(type(self.env)):
            action_min = self.sys.actuator.ctrl_range[:, 0]._value
            action_max = self.sys.actuator.ctrl_range[:, 1]._value
            action = (action - action_min) / (action_max - action_min) * 2 - 1
            #self.state.append(self.env.step(self.state[-1], action))
            self.state.append(self._step(self.state[-1], action))
            next_state = np.concatenate([self.state[-1].pipeline_state.q, self.state[-1].pipeline_state.qd], axis=-1)
            next_state = self._get_obs()
            reward = self.state[-1].reward._value
            info = self.state[-1].info
            terminated = self.state[-1].info['truncation']
            return next_state, reward, terminated, info
        elif "SingleTrackVecEnv" in str(type(self.env)):
            s, r, t, i =  self.env.step(action[None].astype(np.float32))
            return s[0], r[0], t[0], i[0]
        else:
            return self.env.step(action)

    def reset(self):
        if "gym" in str(type(self.env)):
            self.env.reset()
        elif "brax" in str(type(self.env)):
            self.state = [self.env.reset(jax.random.PRNGKey(0))]
        else:
            self.env.reset()

    def render(self, trajectories=None, costs=None):
        if not self.render:
            return None
        if "gym" in str(type(self.env)):
            return self.env.render()
        elif "brax" in str(type(self.env)):
            pass
            #self.mj_data.qpos[:] = self.state[-1].pipeline_state.q[:self.env.sys.nq]
            #self.mj_data.qvel[:] = self.state[-1].pipeline_state.qd[:self.env.sys.nv]
            #self.mujoco_renderer.render("human")
            #render(self.env.sys, [self.state.pipeline_state])
        elif "SingleTrackVecEnv" in str(type(self.env)):
            return self.env.render(trajectories=trajectories, costs=costs)
        else:
            return self.env.render()

    def save_rendering(self, path):
        if "brax" in str(type(self.env)):
            with open(path, 'w') as f:
                f.write(html.render(self.env.sys, [x.pipeline_state for x in self.state]))

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def __deepcopy__(self, memo):
        return EnvWrapper(deepcopy(self.env))

    def _get_obs(self):
        return np.concatenate([self.state[-1].pipeline_state.q, self.state[-1].pipeline_state.qd], axis=-1)
    
    def get_state(self):
        if "gym" in str(type(self.env)):
            return torch.tensor(self.env.unwrapped._get_obs().copy())
        elif "brax" in str(type(self.env)):
            return torch.tensor(self._get_obs().copy())
        elif "SingleTrackVecEnv" in str(type(self.env)):
            return self.env.unwrapped.simulator.get_state()[0]
        return self.env.get_state()

def load_env_and_model(env_name, simulator, n_envs, render=False):
    assert simulator in ["gym", "brax"], "Unknown simulator, only gym and brax are supported"
    print(f"Loading {env_name} environment")

    if env_name == "pendulum":
        model = Pendulum()
        env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    elif env_name == "acrobot":
        model = Acrobot()
        env = gym.make("dm_control/acrobot-swingup-v0", render_mode="human" if render else None)
        from shimmy.utils.dm_env import dm_obs2gym_obs
        def get_obs(self):
            return np.concat([self.physics.data.qpos, self.physics.data.qvel])
        #env.unwrapped._get_obs = get_obs
        setattr(type(env.unwrapped), "_get_obs", get_obs)
    elif env_name == "go1":
        if simulator == "gym":
            env = gym.make(
                'Ant-v4',
                xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
                forward_reward_weight=1,  # kept the same as the 'Ant' environment
                ctrl_cost_weight=0.05,  # changed because of the stronger motors of `Go1`
                contact_cost_weight=5e-4,  # kept the same as the 'Ant' environment
                healthy_reward=1,  # kept the same as the 'Ant' environment
                main_body=1,  # represents the "trunk" of the `Go1` robot
                healthy_z_range=(0.195, 0.75),
                include_cfrc_ext_in_observation=False,
                exclude_current_positions_from_observation=False,
                reset_noise_scale=0.1,
                frame_skip=25,
                terminate_when_unhealthy=False,
                render_mode="human" if render else None,
            )
            model = Go1(deepcopy(env))
        elif simulator == "brax":
            raise NotImplementedError("Go1 is not implemented in Brax")
    elif env_name == "half_cheetah":
        if simulator == "gym":
            env = gym.make("HalfCheetah-v4", render_mode="human" if render else None, exclude_current_positions_from_observation=False)
            model = HalfCheetah(deepcopy(env))
        elif simulator == "brax":
            raise NotImplementedError("HalfCheetah is not implemented in Brax")
    elif env_name == "walker":
        if simulator == "gym":
            env = gym.make("Walker2d-v4", render_mode="human" if render else None,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
            model = Walker2D(deepcopy(env))
        elif simulator == "brax":
            raise NotImplementedError("Walker2d is not implemented in Brax")
    elif env_name == "humanoid":
        if simulator == "gym":
            env = gym.make("Humanoid-v4", render_mode="human" if render else None,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False,
                           include_cinert_in_observation=False,
                           include_cvel_in_observation=False,
                           include_qfrc_actuator_in_observation=False,
                           include_cfrc_ext_in_observation=False)
            model = Humanoid(deepcopy(env))
        elif simulator == "brax":
            import jax.numpy as jnp
            def get_obs(state):
                return jnp.concatenate([state.pipeline_state.q,
                                       state.pipeline_state.qd], axis=-1).block_until_ready()
            env = envs.create('humanoid', backend='mjx', auto_reset=False,
                              terminate_when_unhealthy=True,
                              exclude_current_positions_from_observation=False)
            #env_ = envs.create('humanoid', backend='mjx',
            #                  terminate_when_unhealthy=False,
            #                  exclude_current_positions_from_observation=False)
            #s = env.reset(jax.random.PRNGKey(0))
            #state_jnp = get_obs(s)
            #nq = env.sys.nq
            #nv = env.sys.nv
            #rngs = jax.random.split(jax.random.PRNGKey(0), n_envs)
            ##s_ = env_.reset(rngs)
            #s_ = env_.reset(jax.random.PRNGKey(1))
            #s_next = env.step(s, np.zeros(env.action_size))
            #s_next_obs = get_obs(s_next)
            ##s_s = s_.tree_replace({"pipeline_state.q": state_jnp[:nq],
            ##                      "pipeline_state.qd": state_jnp[nq:nq+nv]})
            #from brax.mjx import pipeline
            #pipeline_state = pipeline.init(env.sys, state_jnp[:nq], state_jnp[nq:nq+nv], None, False)
            #s_s = s.replace(pipeline_state=pipeline_state)
            #s_next_ = env_.step(s_s, np.zeros(env.action_size))
            #s_next_obs_ = get_obs(s_next_)

            
            model = HumanoidBrax(env, n_envs=n_envs)
        #noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
    elif env_name == "humanoid_standup":
        if simulator == "gym":
            env = gym.make("HumanoidStandup-v4", render_mode="human" if render else None,
                        exclude_current_positions_from_observation=False,
                        include_cinert_in_observation=False,
                        include_cvel_in_observation=False,
                        include_qfrc_actuator_in_observation=False,
                        include_cfrc_ext_in_observation=False)
            model = HumanoidStandup(deepcopy(env))
        elif simulator == "brax":
            raise NotImplementedError("HumanoidStandup is not implemented in Brax")
    elif env_name == "hopper":
        if simulator == "gym":
            #env = gym.make("Hopper-v4", frame_skip=8, render_mode="human" if render else None,
            env = gym.make("Hopper-v4", render_mode="human" if render else None,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
            model = Hopper(deepcopy(env))
        elif simulator == "brax":
            raise NotImplementedError("Hopper is not implemented in Brax")
    elif env_name == "swimmer":
        if simulator == "gym":
            env = gym.make("Swimmer-v4", render_mode="human" if render else None,
                           exclude_current_positions_from_observation=False)
            model = Swimmer(deepcopy(env))
        elif simulator == "brax":
            raise NotImplementedError("Swimmer is not implemented in Brax")
    elif env_name == "car":
        dt = 0.05
        #track = "oval"
        track = "icra_2023"
        reward_type = "mppi"
        #reward_type = "rl"
        env = SingleTrackVecEnv(num_envs=1, reset_if_off_track=False, two_way_tracks=False,
                                reward_type=reward_type, compile=False, tracks=[track], dt=dt)
        model = SingleTrack(SingleTrackVecEnv(num_envs=n_envs, reset_if_off_track=False,
                                              two_way_tracks=False, reward_type=reward_type,
                                              compile=True, tracks=[track], dt=dt))
        #model = None
    else:
        raise ValueError("Unknown environment")
    return EnvWrapper(env, render), model