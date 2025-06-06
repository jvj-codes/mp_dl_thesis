# %% lab_5.py
#   deep reinforcement learning with JAX


# %% Imports, packages/lib
import gymnasium as gym  # not jax based, a library with different envoirments
from jax import random, grad, nn   #
import jax.numpy as jnp
from tqdm import tqdm    # information about loops
from collections import deque, namedtuple #a list a person can add stuff to, add stuff for some structure - store states etc
import pickle 
import imageio

# %% Constants 
env = gym.make("CartPole-v1",  render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=2500) 
# %% pickle file functions, save and load best model
def save_model_pickle(best_params, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(best_params, f)
    
def load_model_pickle(file_name):    
    with open(file_name, 'rb') as f:
        best_model = pickle.load(f)
    return best_model

# %% MODEL
def DL_model(params: list):
        rng, layers, hidden_units, scale, obs_size, out_size = params
        
        rng, *keys = random.split(rng, layers)
        
        weights_biases = []
    
        #first layer
        w = random.normal(keys[0], (obs_size, hidden_units)) * scale
        b = random.normal(keys[0], (hidden_units,)) * scale
        weights_biases.append((w, b))
        
        ##hidden layers
        for i in range(1, layers-1):
            w = random.normal(keys[i], (hidden_units, hidden_units)) * scale #create random weights 
            b = random.normal(keys[i], (hidden_units,)) * scale
            weights_biases.append((w, b))
        
        #output layer
        w_out = random.normal(keys[-1], (hidden_units, out_size)) * scale
        b_out = random.normal(keys[-1], (out_size,)) * scale
        weights_biases.append((w_out, b_out))
        return weights_biases

# %% FORWARD PASS
def forward_pass(obs, params):
    # obs is the current observation (input to the network)
    x = obs  # Input layer
    
    #loop through each layer (weights and biases) in the network
    for i, (w, b) in enumerate(params[:-1]):  #hidden layers
        x = jnp.dot(x, w) + b  #linear combination: W*x + b
        x = nn.relu(x)  #apply ReLU activation function
    
    #for the output layer, no activation function
    w_out, b_out = params[-1]  #output layer weights and biases
    logits = jnp.dot(x, w_out) + b_out  #compute logits for the action
    return logits

# %% POLICY 
def policy_fn(rng, obs, params, epsilon=1.0):
    #epsilon-greedy action selection
    if random.uniform(rng, ()) < epsilon:
        #random action with probability epsilon
        return int(random.randint(rng, (1,), 0, env.action_space.n)[0])
    
    else:
        #compute action probabilities
        logits = forward_pass(obs, params)
        
        #choose action with highest probability
        action = jnp.argmax(logits)
        return int(action)
# %% SAMPLE BATCH
def sample_batch(rng, memory, batch_size):
    #choose indices randomnly and select those and put memories into a batch
    indices = random.choice(rng, jnp.arange(len(memory)), (batch_size,), replace=False)
    batch = [memory[i] for i in indices]
    return batch

# %% LOSS FUNCTION
def ql_loss_function(params: list, batch: jnp.array, model_copy: None, gamma=0.98) -> float:
    
    #fetch values in arrays from batch
    obs_batch = jnp.array([entry.obs for entry in batch])
    action_batch = jnp.array([entry.action for entry in batch])
    reward_batch = jnp.array([entry.reward for entry in batch])
    next_obs_batch = jnp.array([entry.next_obs for entry in batch])
    done_batch = jnp.array([entry.done for entry in batch])
    
    #compute Q-values for the current observations
    q_values = forward_pass(obs_batch, params)
    
    #compute Q-values for the next observations
    next_q_values = forward_pass(next_obs_batch, model_copy)
        
    #get the Q-values for the selected actions
    selected_q_values = jnp.take_along_axis(q_values, action_batch[:, None], axis=1).squeeze()
    
    #compute target Q-values using Bellman equation
    target_q_values = reward_batch + gamma * jnp.max(next_q_values, axis=1) * (1-done_batch.astype(int))
    
    #compute mean squared error loss between target and predicted Q-values
    loss = jnp.mean((target_q_values - selected_q_values) ** 2)
    
    return loss

# %% UPDATE FUNCTION
def update_fn(params, batch, model_copy: None, learning_rate = 0.01, gamma = 0.99):
    #compute gradients of loss with respect to parameters
    grads = grad(ql_loss_function)(params, batch, model_copy, gamma)
    
    #compute new parameters using gradient and learning rate
    new_params = [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params
# %% Environment #########################################################
obs, info = env.reset()


##initial parameters for neural network
scale = 0.01 #scaling the initial weights and biases
layers = 4   #one input layer, 2 hidden layers, 1 output layer
hidden_units = 64 #neurons in the hidden layers
obs_size = env.observation_space.shape[0] #observation size of the envoirment (Position, velocity, angle, angle velocity) 
out_size = env.action_space.n #output layer, left or right for the cartpoole.

##put initial parameters into a list
init_params = [rng, layers, hidden_units, scale, obs_size, out_size]

##neural network model using initial parameters
params = DL_model(init_params)

##make a copy of the model
model_copy = params.copy()

##hyperparameters
min_epsilon = 0.02
epsilon = 1
decay = 0.9975

learning_rate = 0.1 #0.02bad
gamma = 0.95
T_copy  = 30
batch_size = 128
episodes = 2000


best_score = 0 #initialize best score (sum of rewards)

for i in tqdm(range(episodes)): #run the episodes
    rng, key = random.split(rng) #store key and random number generator for reproducibility
    
    done = False #initialize done false
    
    epsilon = max(min_epsilon, epsilon *  decay) #calculate epsilon, max of min eps and decay of eps
    
    reward_sum = 0 #set the reward sum to zero for each episode
    while not done: #when the game is not done (truncated or termianted) we run the following
        rng, key = random.split(rng) #store key and random number generator for reproducibility
        action = policy_fn(key, obs, params, epsilon) #action decided by the policy function
    
        next_obs, reward, terminated, truncated, info = env.step(action) #use the action as the next step and store information from the envoirment
        memory.append(entry(obs, action, reward, next_obs, terminated)) #append observations / info to mememory
        obs, info = next_obs, info if not (terminated | truncated) else env.reset() #store observation and info if the game is not done, else reset the envoirment
        
        done = (terminated | truncated) #boolean, set done to true if either terminated or truncated
        
     
        reward_sum += reward #sum the reward from actions, +1 for staying alive
    
    if reward_sum > best_score:  #if the reward sum of the current play is greater than previous best scores
        best_params = params.copy() #copy the params for the best current model
        best_score = reward_sum #store the best current best core
        
        # if best_score == 500: #incase the model now perfectly finishes the game break the loop
        #     break             #the max score is 500 of the cartpool game
    
    if len(memory) >= batch_size:      #when we reach a certain batch size we can start train the model  
        rng, sample_key = random.split(rng) #store key and random number generator for reproducibility
        
        batch = sample_batch(sample_key, memory, batch_size) #fetch random batch using the memory
    
        if i % T_copy == 0: #every T step copy parameters, used for fixed q targets algorithm
            model_copy = params.copy()
        
        loss = ql_loss_function(params, batch, model_copy, gamma) #calculate the loss for visuals
        params = update_fn(params, batch, model_copy, learning_rate, gamma) #update parameters using gradient descent
        
        tqdm.write(f"Episode {i}, Loss: {loss:.5f}, Reward: {reward_sum}, Epsilon: {epsilon:.5f}") #print for visuals
    

env.close() #close the envoirment

file_name = 'best_model_params.pkl' #file name for best model parameters
save_model_pickle(best_params, file_name) #save the best parameters in pickle file

# %% Run the envoirment using the best model and save a gif of the player playing

env = gym.make("CartPole-v1", render_mode="rgb_array")

obs, info = env.reset()
done = False
total_reward = 0
frames = []
best_model = load_model_pickle(file_name)

while not done:
    rng, key = random.split(rng)

    #use the trained model to select the action
    action = policy_fn(key, obs, best_model, epsilon=0.0)  # epsilon=0.0 to select the best action (no exploration)

    #take the action in the environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    frame = env.render() 
    frames.append(frame) #append frame to list

    #update observation
    obs = next_obs

    #check if the episode is done
    done = terminated or truncated

env.close()
print(f"Total reward achieved: {total_reward}")
gif_filename = "trained_agent_cartpole.gif"
imageio.mimsave(gif_filename, frames, fps=30)

# %% Run the envoirment with a random agent playing
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Reset the environment to get the first observation
obs, info = env.reset()

# Initialize the list to store frames
frames = []
done = False
total_reward = 0

#run the random agent in the environment
#while not done:
 
for i in range(1, 10):
     #take a random action from the environment's action space
    action = env.action_space.sample()

    #take the action in the environment and collect reward
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    #append the current frame (RGB array) to the frames list
    frame = env.render()
    frames.append(frame)

    #update the observation
    obs = next_obs

    #check if the episode is done
    #done = terminated or truncated
    if terminated == True:
        obs, info = env.reset()

env.close()

#save the frames as a .gif file using imageio
gif_filename = "random_agent_cartpole.gif"
imageio.mimsave(gif_filename, frames, fps=30)

print(f"GIF saved as {gif_filename}")

# %%
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Initialize the list to store frames
frames = []

# Set a few trials for failure
num_episodes = 10
#failure_probability = 0.5  # 50% chance of early termination to simulate failure

# Run the random agent in the environment
for episode in range(num_episodes):
    # Reset the environment to get the first observation
    obs, info = env.reset()

    done = False
    total_reward = 0
    episode_frames = []
    
    while not done:
        # Take a random action from the environment's action space
        action = env.action_space.sample()

        # Take the action in the environment and collect reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Append the current frame (RGB array) to the frames list
        frame = env.render()
        episode_frames.append(frame)

        # Update the observation
        obs = next_obs

        # Randomly terminate the episode (simulating failure)
        # if random.random() < failure_probability:
        #    terminated = True

        # Check if the episode is done
        if terminated or truncated:
            frames.extend(episode_frames)
            break

    # Reset the environment after the episode is done
    if terminated == True:
        obs, info = env.reset()

env.close()

# Save the frames as a .gif file using imageio
gif_filename = "random_agent_cartpole_with_failures.gif"
imageio.mimsave(gif_filename, frames, fps=30)

print(f"GIF saved as {gif_filename}")

