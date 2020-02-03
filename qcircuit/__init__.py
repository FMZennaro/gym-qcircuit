from gym.envs.registration import register

register(
    id='qcircuit-v0',
    entry_point='qcircuit.envs:QCircuitEnv0',
)
register(
    id='qcircuit-v1',
    entry_point='qcircuit.envs:QCircuitEnv1',
)
