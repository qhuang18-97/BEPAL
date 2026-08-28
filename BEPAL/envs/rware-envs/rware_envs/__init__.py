# Unlike ic3net_envs, there is nothing to register with gym here: the `rware`
# package registers its own env ids (e.g. rware-tiny-4ag-v1) on import.
# RwareEnv is a plain wrapper around one of those, exposing the same interface
# as env_wrappers.GymWrapper so that data.init can return it unwrapped.
from .rware_env import RwareEnv
from .teacher import move
