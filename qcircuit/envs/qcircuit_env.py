"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math,sys
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, Aer, execute

def get_prob_amplitude(c):
    return c.real**2 + c.imag**2

def approx(query, target, approx=10**-4):
    if(np.abs(query-target) < approx):
        return True
    else:
        return False



class QCircuitEnv0(gym.Env):
    """
    Description:
        An empty quantum circuit with 1 qubit (and 1 classical bit) is given. Set the qubit in perfect superposition.
    Observation: 
        Type: Box(4)
        Num    Observation                 Min         Max
        0   |0> real                     -1            1
        1   |0> complex                  -1            1
        2   |1> real                     -1            1
        3   |1> complex                  -1            1   
    Actions:
        Type: Discrete(3)
        Num    Action
        0    Add x gate on q0
        1    Add h gate on q0
        2   Remove last gate
    Reward: 
        -1 for every gate added or removed, +100 for termination.
    Starting State:
        Empty circuit with 1 quantum and 1 classical bit.
    Episode Termination:
        Bit is in perfect superposition.
    """
    
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # State
        self.simulator = Aer.get_backend('unitary_simulator')
        self.circuit = QuantumCircuit(1,1)
        self.startstate = np.array([1+0j, 0+0j])
        self.qstate = self._get_qstate()
        self.state = self._get_state()
                
        # Observation space
        high = np.array([1.]*4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space
        self.action_space = spaces.Discrete(3)
        
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        if action==0:
            self.circuit.x(0)
        elif action==1:
            self.circuit.h(0)
        else:
            if(len(self.circuit.data)>=1):
                self.circuit.data.pop(-1)
        
        self.qstate = self._get_qstate()
        self.state = self._get_state()
        
        done = (approx(get_prob_amplitude(self.qstate[0]),.5)) and (approx(get_prob_amplitude(self.qstate[1]),.5))
        done = bool(done)
        
        if not done:
            reward = -1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 100.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {'circuit_img':self.render()}

    def reset(self):
        self.circuit = QuantumCircuit(1,1)
        self.startstate = np.array([1+0j, 0+0j])
        self.qstate = self._get_qstate()
        self.state = self._get_state()        
        self.steps_beyond_done = None
        return np.array(self.state)
    
    def _get_qstate(self):
        result = execute(self.circuit,backend=self.simulator).result()
        unitary = result.get_unitary()
        return np.dot(unitary,self.startstate)
    
    def _get_state(self):
        state = []
        for i in range(self.qstate.size):
            state.append(self.qstate[i].real)
            state.append(self.qstate[i].imag)
        return np.array(state)
    
    def render(self, mode='human'):
        return self.circuit.draw(output='mpl')
        
    def close(self):
        return
    
    
class QCircuitEnv1(gym.Env):
    """
    Description:
        An empty quantum circuit with 2 qubit (and 2 classical bit) is given. Set the qubit in the state |Phi+> = 1/sqrt(2) |00> + 1/sqrt(2) |11>.
    Observation: 
        Type: Box(8)
        Num    Observation                 Min         Max
        0   |00> real                     -1            1
        1   |00> complex                  -1            1  
        2   |01> real                     -1            1
        3   |01> complex                  -1            1
        4   |10> real                     -1            1
        5   |10> complex                  -1            1
        6   |11> real                     -1            1
        7   |11> complex                  -1            1 
    Actions:
        Type: Discrete(3)
        Num    Action
        0    Add x gate on q0
        1    Add h gate on q0
        2    Add x gate on q1
        3    Add h gate on q1
        4    Add cnot gate on (q0,q1)
        5    Add cnot gate on (q1,q0)
        6   Remove last gate
    Reward: 
        -1 for every gate added or removed, +100 for termination.
    Starting State:
        Empty circuit with 2 quantum and 2 classical bit. Quantum bits in |00> state.
    Episode Termination:
        Bits entangled in state |Phi> or
        Reached maxsteps of 10
    """
    
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # State
        self.simulator = Aer.get_backend('unitary_simulator')
        self.circuit = QuantumCircuit(2,2)
        self.startstate = np.array([1+0j, 0+0j, 0+0j, 0+0j])
        self.qstate = self._get_qstate()
        self.state = self._get_state()
                
        # Observation space
        high = np.array([1.]*8)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space
        self.action_space = spaces.Discrete(7)
        
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None
        self.maxsteps = 10
        self.nsteps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        if action==0:
            self.circuit.x(0)
        elif action==1:
            self.circuit.h(0)
        elif action==2:
            self.circuit.x(1)
        elif action==3:
            self.circuit.h(1)
        elif action==4:
            self.circuit.cx(0,1)
        elif action==5:
            self.circuit.cx(1,0)
        else:
            if(len(self.circuit.data)>=1):
                self.circuit.data.pop(-1)
        
        self.qstate = self._get_qstate()
        self.state = self._get_state()
        
        self.nsteps += 1
        
        done = ((approx(self.qstate[0].real,1./np.sqrt(2))) and (approx(self.qstate[3].real,1./np.sqrt(2)))) or (self.nsteps > self.maxsteps)
        done = bool(done)
        
        if not done:
            reward = -1.0
        elif (self.steps_beyond_done is None) and (self.nsteps > self.maxsteps):
            reward = -1.0
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is None:
            reward = 100.0
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {'circuit_img':self.render()}

    def reset(self):
        self.circuit = QuantumCircuit(2,2)
        self.startstate = np.array([1+0j, 0+0j, 0+0j, 0+0j])
        self.qstate = self._get_qstate()
        self.state = self._get_state()        
        self.steps_beyond_done = None
        self.nsteps = 0
        return np.array(self.state)
    
    def _get_qstate(self):
        result = execute(self.circuit,backend=self.simulator).result()
        unitary = result.get_unitary()
        return np.dot(unitary,self.startstate)
    
    def _get_state(self):
        state = []
        for i in range(self.qstate.size):
            state.append(self.qstate[i].real)
            state.append(self.qstate[i].imag)
        return np.array(state)
    
    def render(self, mode='human'):
        return self.circuit.draw(output='mpl')
        
    def close(self):
        return
