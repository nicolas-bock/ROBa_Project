import torch
import random
import numpy as np
from collections import deque
from robot_motions import Direction, Game, Robot
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.2 # randomness
        self.gamma = 0.8   # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.five_last_states = []

    def get_state(self, game):
        next_l = Robot(x=game.robot.x - game.robot.speed, y=game.robot.y, radius=game.robot.radius)
        next_r = Robot(x=game.robot.x + game.robot.speed, y=game.robot.y, radius=game.robot.radius)
        next_u = Robot(x=game.robot.x, y=game.robot.y - game.robot.speed, radius=game.robot.radius)
        next_d = Robot(x=game.robot.x, y=game.robot.y + game.robot.speed, radius=game.robot.radius)

        dir_l = game.robot.direction == Direction.LEFT
        dir_r = game.robot.direction == Direction.RIGHT
        dir_u = game.robot.direction == Direction.UP
        dir_d = game.robot.direction == Direction.DOWN

        border_dist_l = abs(game.map.border_x1) - game.robot.x
        border_dist_r = abs(game.map.border_x2) - game.robot.x
        border_dist_u = abs(game.map.border_y1) - game.robot.y
        border_dist_d = abs(game.map.border_y2) - game.robot.y
        min_border_dist = min(border_dist_l, border_dist_r, border_dist_u, border_dist_d)

        state = [
            (dir_l and next_l.collision_detection(game)) or
            (dir_r and next_r.collision_detection(game)) or
            (dir_u and next_u.collision_detection(game)) or
            (dir_d and next_d.collision_detection(game)),

            (dir_l and next_u.collision_detection(game)) or
            (dir_r and next_d.collision_detection(game)) or
            (dir_u and next_r.collision_detection(game)) or
            (dir_d and next_l.collision_detection(game)),

            (dir_l and next_d.collision_detection(game)) or
            (dir_r and next_u.collision_detection(game)) or
            (dir_u and next_l.collision_detection(game)) or
            (dir_d and next_r.collision_detection(game)),

            (dir_l and next_r.collision_detection(game)) or
            (dir_r and next_l.collision_detection(game)) or
            (dir_u and next_d.collision_detection(game)) or
            (dir_d and next_u.collision_detection(game)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # New area nearby
            border_dist_l == min_border_dist,
            border_dist_r == min_border_dist,
            border_dist_u == min_border_dist,
            border_dist_d == min_border_dist
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game {agent.n_games}, Score: {score}, Record: {record}")

            agent.five_last_states.append((score, record))
            if len(agent.five_last_states) > 20:
                agent.five_last_states.pop(0)

            if all(x == agent.five_last_states[0] for x in agent.five_last_states) or (agent.n_games > 50 and record == 1):
                game.generate_map()

            # Plot the score and mean score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()