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
                For this trading project: 54 (3×3×2×3 discrete states)
            action_size: Number of possible actions
                For this trading project: 5 (Hold, Buy Conservative, Buy Aggressive, 
                Sell Conservative, Sell Aggressive)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        
        Note:
            For the trading environment, initialize with:
            agent = QLearningAgent(state_size=54, action_size=5)
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
        
        # For convergence tracking: store previous Q-table
        self.previous_q_table = None
        self.convergence_history = []  # Track convergence over episodes
    
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
        # Reset convergence tracking when loading
        self.previous_q_table = None
        self.convergence_history = []
    
    def end_episode(self):
        """
        Call this at the end of each episode to update convergence tracking.
        Stores the current Q-table as the previous one for next episode comparison.
        """
        # Store a copy of the current Q-table
        self.previous_q_table = self.q_table.copy()
    
    def calculate_convergence(self):
        """
        Calculate convergence metric: sum of absolute differences between
        current Q-table and previous Q-table.
        
        According to QLearningStructure.md:
        "We will check convergence with the sum of absolute differences between 
        the Q-Table at the end of the current episode and the previous episode."
        
        Returns:
            Sum of absolute differences, or None if no previous Q-table exists
        """
        if self.previous_q_table is None:
            return None
        
        # Calculate sum of absolute differences
        convergence_value = np.sum(np.abs(self.q_table - self.previous_q_table))
        
        return convergence_value
    
    def check_convergence(self, threshold=0.01):
        """
        Check if the Q-table has converged based on the convergence metric.
        
        Args:
            threshold: Convergence threshold. If the sum of absolute differences
                        is below this value, we consider the Q-table converged.
                        Default: 0.01
        
        Returns:
            Tuple (is_converged, convergence_value)
            - is_converged: Boolean indicating if convergence is reached
            - convergence_value: The convergence metric (sum of absolute differences)
        """
        convergence_value = self.calculate_convergence()
        
        if convergence_value is None:
            return False, None
        
        is_converged = convergence_value < threshold
        
        return is_converged, convergence_value
    
    def get_convergence_history(self):
        """
        Get the history of convergence values across episodes.
        
        Returns:
            List of convergence values (one per episode)
        """
        return self.convergence_history.copy()
    
    def track_convergence(self):
        """
        Calculate and track convergence for the current episode.
        Call this at the end of each episode after end_episode().
        
        Returns:
            Convergence value (sum of absolute differences), or None if first episode
        """
        convergence_value = self.calculate_convergence()
        if convergence_value is not None:
            self.convergence_history.append(convergence_value)
        return convergence_value

