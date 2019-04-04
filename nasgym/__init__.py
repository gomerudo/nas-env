"""The different NAS environments implemented."""
from gym.envs.registration import register

register(
    id='NAS-v0',
    entry_point='nasgym.envs:DefaultNASEnv',
)
