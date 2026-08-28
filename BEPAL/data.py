import sys
import gym
from env_wrappers import *

# ic3net_envs and rware_envs are imported inside the branches that need them,
# so each environment can be installed and run on its own.

def _make(env_id):
    # gym >= 0.21 wraps gym.make() in OrderEnforcing, whose reset(**kwargs)
    # signature hides the underlying env's `epoch` parameter from the
    # getargspec() check in env_wrappers.GymWrapper.reset(). Unwrap to get the
    # bare env these wrappers expect (the starcraft branch below does the same
    # with env.env).
    return gym.make(env_id).unwrapped

def init(env_name, args, final_init=True):
    if env_name == 'levers':
        import ic3net_envs
        env = _make('Levers-v0')
        env.multi_agent_init(args.total_agents, args.nagents)
        env = GymWrapper(env)
    elif env_name == 'number_pairs':
        import ic3net_envs
        env = _make('NumberPairs-v0')
        m = args.max_message
        env.multi_agent_init(args.nagents, m)
        env = GymWrapper(env)
    elif env_name == 'predator_prey':
        import ic3net_envs
        env = _make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        import ic3net_envs
        env = _make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'starcraft':
        env = gym.make('StarCraftWrapper-v0')
        env.multi_agent_init(args, final_init)
        env = GymWrapper(env.env)
    elif env_name == 'rware':
        # RwareEnv already implements the GymWrapper interface, so it is
        # returned unwrapped.
        import rware_envs
        env = rware_envs.RwareEnv(args)
    else:
        raise RuntimeError("wrong env name")

    return env
