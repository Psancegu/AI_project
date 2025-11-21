"""
Q-Learning implementation for trading.
This module contains the Q-learning algorithm for trading strategies.
"""

import numpy as np


class QLearningAgent:
    """
    Q-Learning agent for trading.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.01, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
    
    def act(self, state, training=True):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and np.random.random() <= self.epsilon:
            # Explore: random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update rule
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save file
        """
        np.save(filepath, self.q_table)
    
    def load(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load file
        """
        self.q_table = np.load(filepath)

