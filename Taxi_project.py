# ========================================================================
# 1. Import Dependencies
# ========================================================================
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time  # For controlling animation speed

# ========================================================================
# ========================================================================
# 2. Enhanced Training Function with Visualization (TAXI VERSION)
# ========================================================================
def train_agent(episodes=5000, render_training=False):
    """Train the AI with live environment visualization"""
    
    # ====================================================================
    # 3. Taxi Environment Setup
    # ====================================================================
    env = gym.make('Taxi-v3', render_mode='human' if render_training else None)
    
    # ====================================================================
    # 4. Q-Table Initialization
    # ====================================================================
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    # ====================================================================
    # 5. Learning Parameters
    # ====================================================================
    learning_rate = 0.8
    discount = 0.95
    epsilon = 1.0
    epsilon_decay = 0.001
    rng = np.random.default_rng()
    
    # ====================================================================
    # 6. Training Loop with Live Visualization
    # ====================================================================
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0
        
        # Show initial state
        if render_training and episode % 100 == 0:
            env.render()
            time.sleep(0.1)
        
        while not done:
            # Exploration vs Exploitation
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            
            # Perform action
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-table
            q_table[state, action] += learning_rate * (
                reward + discount * np.max(q_table[new_state, :]) - q_table[state, action]
            )
            
            # Visual update
            if render_training and episode % 100 == 0:
                env.render()
                time.sleep(0.1)
                
            state = new_state
            steps += 1
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: Steps {steps}, Reward {reward}")
            
        # Decay exploration rate
        epsilon = max(epsilon - epsilon_decay, 0.01)
    
    env.close()
    return q_table

# ========================================================================
# 7. Live Demonstration Function (TAXI VERSION)
# ========================================================================
def demonstrate_agent(q_table):
    """Show trained agent performing 5 complete runs"""
    env = gym.make('Taxi-v3', render_mode='human')
    
    for _ in range(5):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q_table[state, :])
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.5)
        print(f"Total reward: {total_reward}")
        time.sleep(1)
    
    env.close()

# ========================================================================
# 8. Main Execution Flow
# ========================================================================
if __name__ == '__main__':
    # Phase 1: Training
    print("Training Taxi agent...")
    taxi_q = train_agent(episodes=5000, render_training=True)
    
    # Phase 2: Demonstration
    print("\nTrained Taxi in action:")
    demonstrate_agent(taxi_q)
    
    # Save results
    with open('taxi_q_table.pkl', 'wb') as f:
        pickle.dump(taxi_q, f)