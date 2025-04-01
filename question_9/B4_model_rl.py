import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from DSA3101_group9.data.subB_data_preprocessing import banking_marketing_train_encoded, banking_marketing_test_encoded
from sklearn.preprocessing import StandardScaler


banking_marketing_train_rl = banking_marketing_train_encoded
banking_marketing_test_rl = banking_marketing_test_encoded

# # Initialize the scaler
# scaler = StandardScaler()

# # Standardize the training data
# xtrain = scaler.fit_transform(banking_marketing_train_rl.drop(columns=['y']))

# # Do not standardize the target variable
# ytrain = banking_marketing_train_rl['y'].values

# # Standardize the test data using the parameters from the training set
# xtest = scaler.transform(banking_marketing_test_rl.drop(columns=['y']))

# # Do not standardize the target variable in the test set
# ytest = banking_marketing_test_rl['y'].values

# define
xtrain = banking_marketing_train_rl.drop(columns=['y'])
ytrain = banking_marketing_train_rl['y'].values
xtest = banking_marketing_test_rl.drop(columns=['y'])
ytest = banking_marketing_test_rl['y'].values

# Extract feature names for later use
features = banking_marketing_train_rl.drop(columns=['y']).columns

# extract index of certain columns
clv_index = features.index('CLV')
cost_index = features.index('cost')

# Target variable (no need to standardize)
target = ['y']

# Hyperparameters
STATE_SIZE = xtrain.shape[1]
ACTION_SIZE = 3  # Broad, Group-Based, AI-Based
GAMMA = 0.95  # Discount Factor
LR = 0.001  # Learning Rate
EPSILON = 1.0  # Initial exploration
EPSILON_DECAY = 0.98  # Smoother decay
EPSILON_MIN = 0.01
MEMORY_SIZE = 5000  # Increased memory for better experience replay
BATCH_SIZE = 64  # Larger batch size for stability

# Define Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

# Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON  # Exploration rate
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(self.model(state_tensor)).item()

    def train(self):
        if len(self.memory.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)

        q_values = self.model(states_tensor).gather(1, actions_tensor).squeeze()
        next_q_values = self.model(next_states_tensor).max(1)[0].detach()
        target_q_values = rewards_tensor + GAMMA * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# Reward function
def calculate_reward(conversion, cost, retention_rate, clv, budget_left):
    profit = conversion * clv
    penalty = 0.2 * cost  # Reduced penalty to balance strategy
    budget_penalty = -10 if budget_left < 0 else 0
    return profit - penalty + (0.7 * retention_rate) + budget_penalty

# Extract state representation
def get_state(customer):
    return np.array(customer).flatten()

# Strategy-based retention rate
def calculate_strategy_based_retention_rate(action):
    return [np.random.uniform(0.2, 0.4), np.random.uniform(0.4, 0.6), np.random.uniform(0.6, 0.8)][action]

# Define strategies and costs
strategies = {0: 'Broad', 1: 'Group-Based', 2: 'AI-Based'} # Broad, Group-Based, AI-Based
# costs = {0: 1, 1: 1.84, 2: 3} # personalisation cost
retention_rates = {0: 0.3, 1: 0.6, 2: 0.9}
campaign_rewards = {0: 50, 1: 55, 2: 60}

# initiate agent
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

num_campaigns = 10
total_rewards = {0: 0, 1: 0, 2: 0}
epsilon = 1.0  # Start with high exploration

for campaign in range(num_campaigns):
    print(f"\n🔹 Running Campaign {campaign + 1}...")

    # input budget amount
    ###
    budget = 10000 
    ###

    indices = np.arange(len(xtrain))
    np.random.shuffle(indices)  # Shuffle training order

    for idx in indices:
        customer = xtrain[idx]
        state = get_state(customer)

        action = agent.select_action(state) if np.random.rand() > epsilon else np.random.choice(list(strategies.keys()))
        
        # cost = costs[action]
        cost = customer[cost_index] # cost column
        # cost = customer['cost']
        retention_rate = calculate_strategy_based_retention_rate(action)

        conversion = ytrain[idx]
        clv = customer[clv_index] if conversion else 0 # CLV column
        # clv = customer['CLV'] if conversion else 0 # CLV column

        reward = calculate_reward(conversion, cost, retention_rate, clv, budget)
        budget -= cost

        # Stop if budget is exhausted
        if budget <= 0:
            break

        next_state = get_state(xtrain[np.random.randint(len(xtrain))])  # Random next state
        agent.memory.add((state, action, reward, next_state))
        agent.train()

        # Update total rewards: Accumulate the individual reward - cost for each customer interaction
        total_rewards[action] += reward

    # Reduce epsilon (less exploration over time)
    epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

# Select best strategy
best_strategy = max(total_rewards, key=total_rewards.get)
print(f"\n✅ Final Recommended Strategy after {num_campaigns} campaigns: {strategies[best_strategy]}")




# test
# Prepare for evaluation
test_rewards = {0: 0, 1: 0, 2: 0}  # Track total rewards for each strategy

# Run evaluation on test data
for idx in range(len(xtest)):  # Iterate over the test data
    customer = xtest[idx]  # Get customer data from the test set
    state = get_state(customer)  # Extract the state representation

    # **Use the trained model to select the best action**
    action = agent.select_action(state)  # Exploit the learned policy

    # Get cost, retention rate, and other necessary info for the selected action
    # cost = costs[action]
    cost = customer[cost_index]
    # cost = customer['cost']
    retention_rate = retention_rates[action]

    # Get actual conversion rate from ytest
    conversion = ytest[idx]  # Test data labels (converted or not)
    clv = customer[clv_index] if conversion else 0  # CLV is 0 if not converted
    # clv = customer['CLV'] if conversion else 0 # CLV column


    # Calculate reward based on conversion, cost, retention, and CLV
    reward = calculate_reward(conversion, cost, retention_rate, clv, budget_left=0)  # Assuming no budget for simplicity
    test_rewards[action] += reward  # Accumulate rewards for each strategy

# Print the final evaluation metrics
print("Evaluation on Test Data:")
for action in strategies.keys():
    print(f"Strategy {strategies[action]}: Total Reward = {test_rewards[action]}")

# Optionally, you can calculate and print the average reward across all strategies
total_test_reward = sum(test_rewards.values())
avg_test_reward = total_test_reward / len(test_rewards)

print(f"Average Reward across all strategies: {avg_test_reward}")

# Select best strategy
best_strategy = max(test_rewards, key=test_rewards.get)
print(f"\n✅ Final Recommended Strategy: {strategies[best_strategy]}")
