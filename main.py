import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, time, datetime, os, copy
import matplotlib.pyplot as plt

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

env = JoypadSpace(env,SIMPLE_MOVEMENT)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """return every skip-th frame"""
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """
        for everu skiped frame and action taken in that frame
        add reward on acumulated reward
        and in the end return the state, acumulated reward, done, trunc and info
        for the state where agent came after all skiped frames
        """
        total_reward = 0.0

        for i in range(self.skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward

            if done:
                break
        return obs, total_reward, done, trunc, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        """
        wrapper that will transform
        rgb frames into grayscale frames
        this custom wrapper inherits the Observation wrapper from gym
        and as argument it takse the enviornment
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2] #from enviornment obseravtion shape we take just size of frame and not the channel size
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self,observation):
        """permute [H,W,C] to [C,H,W]"""
        observation = np.transpose(observation, (2, 0, 1)) #here we change the positions of channel, height and width
        observation = torch.tensor(observation.copy(), dtype=torch.float)# create tenosr with these infos
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)# call for changing positions of c, h, w
        transform = T.Grayscale() #instace of GrayScale transform function
        observation = transform(observation) #GrayScale Trasnforamtion
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        """
        custom Resize wrapper that will resize frame to new size
        it also inherits Obseravtion wrapper from gym
        and takes env and shaep(int) for arguments
        """
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape) #if shape is instance of int crate tuple (shape, shape)
        else:
            self.shape = tuple(shape) #if it is not instace its array so create tuple from array

        obs_shape = self.shape + self.observation_space.shape[2:] #new observation shape will be self.shape which
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transform = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transform(observation).squeeze(0)
        return observation

#print(env.observation_space.shape)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

#print(env.observation_space.shape)



class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        """ agent class"""

        self.state_dim = state_dim #(4, 84, 84) the size of frame
        self.action_dim = action_dim #number of action that agent can take
        self.save_dir = save_dir #directorium where 

        self.device = "cpu" #using gpu for computation

        self.net = MarioNet(self.state_dim, self.action_dim).float() #initializing agent's DQN
        self.net = self.net.to(device=self.device) #to gpu

        self.exploration_rate = 1 # exploratio rate , if bigger the bigger chance for exploring
        self.exploration_rate_decay = 0.99999975 # value that will decrease exploration rate
        self.exploration_rate_min = 0.1 # the minimun exploration rate
        self.curr_step = 0 #varibel that will save the number of steps that are taken

        self.save_every = 5e5 #enveiroment situtation will be saved at every 500000-th step

        self.memory = deque(maxlen=80000)  #list where agents experience will be saved after every action
        #experience = next_state, state, action, reward, done
        self.batch_size = 32 #number of samples that will be pulled out of memory and used for model updating

        self.gamma = 0.9 #discount factor

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025) #optimization algorithm that will be used to optimize loss function
        self.loss_fn = torch.nn.SmoothL1Loss() #function that will be used to calculate deviation between expected and calculated result of aproximation on q funciton

        self.burnin = 1e4 #minimum of experience before learning
        self.learn_every = 3 #number of experience between updates of Q network
        self.sync_every = 1e4 #number of experience between syncornisation of Q target network


    def act(self, state):

        if np.random.rand() < self.exploration_rate: #if exploration rate is bigger than random number from 0 to 1 than do random action
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__() #transform to array
            state = torch.tensor(state, device=self.device).unsqueeze(0) #create tensor from state data
            action_values = self.net(state, model="online") #se
            action_idx = torch.argmax(action_values, axis=1).item() #will get id of action that gives the biggest reward

        self.exploration_rate *= self.exploration_rate_decay #make exploratio rate smaller by multiplying it with decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate) #exploratio rate can not be less than its minimum

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        transform data from tuples to array, make tensors with that data and push them into memeory as tuple
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        pull random 32 samples from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))# first it will
        #unpack batch which means that every tensor inside tuple will be separet agument inside zip function,
        #than it will concatenate all the tensors along the new dimension
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze() #returns the experience
        #squeeze will remove all the dimension of size 1

    def td_esitmate(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]

        return (reward + (1-done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_taget(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (self.save_dir/f"mario_net{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_taget()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_esitmate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)



class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        class tha inherits nn.Module which is base class for all neural network modules
        """
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {h}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()



use_cuda = torch.cuda.is_available()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40000
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

# state = env.reset()
# next_state, reward, done, trunc, info = env.step(env.action_space.sample())
# print(len(next_state.__array__()))
# #print("\n")
# #print(next_state.__array__()[0])
#
# print(f"{next_state.__array__()} state, \n {reward} reward, \n {done} done, \n {info} info")

# episodes = 2
# for e in range(episodes):
#     state = env.reset()
#     while True:
#         env.render()
#         state, reward, done, trunc, info = env.step(env.action_space.sample())
#
#         if done:
#             break
#
# env.close()