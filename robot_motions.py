import pygame
import sys
import random
import numpy as np
from enum import Enum

pygame.init()
pygame.display.set_caption("Robot Motions")

# Directions
class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# Colors
class Colors():
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255
    BLUE = 0, 0, 255
    RED = 255, 0, 0
    GREEN = 0, 255, 0

class Game:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.time = pygame.time
        self.score = 0

        self.map = Map(width=self.width, height=self.height)
        self.robot = Robot(self.width // 2, self.height // 2, speed=5, radius=20)

    def play_step(self, action, bias=False):
        self.screen.fill(Colors.WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return self.robot.reward, True, self.score

        self.robot._update_robot_position(action, bias)
        collision = self.robot.collision_detection(self, True)

        game_over = collision or self.robot.reward < -1000

        self.map._discover_new_area(self.robot)
        self.map._draw_map(self.screen, self.robot, self.time)

        pygame.display.flip()
        self.time.Clock().tick(60)

        return self.robot.reward, game_over, self.score

    def _reset_robot(self):
        self.robot.x = self.width // 2
        self.robot.y = self.height // 2
        reward_per_move = self.robot.reward // self.robot.total_moves if self.robot.total_moves > 0 else 0
        self.score = (self.robot.total_moves + reward_per_move) if self.robot.total_moves + reward_per_move > 0 else 0
        self.robot.total_moves = 0

    def reset(self):
        self.robot.reward = 0
        self._reset_robot()
        self.score = 0

    def generate_map(self):
        self.map = Map(width=self.width, height=self.height)

class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.areas = {}
        self.border_x1 = 0
        self.border_y1 = 0
        self.border_x2 = self.width
        self.border_y2 = self.height

        self._add_area(0, 0)

    def _add_area(self, x_offset, y_offset):
        area_coords = (x_offset, y_offset)
        if area_coords not in self.areas:
            new_area = Area(x_offset, y_offset, x_offset + self.width, y_offset + self.height)
            self.areas[area_coords] = new_area

            self.border_x1 = min(self.border_x1, x_offset)
            self.border_y1 = min(self.border_y1, y_offset)
            self.border_x2 = max(self.border_x2, x_offset + self.width)
            self.border_y2 = max(self.border_y2, y_offset + self.height)

    def _is_area_discovered(self, x1, y1, x2, y2):
        for area in self.areas.values():
            if area.x1 == x1 and area.y1 == y1 and area.x2 == x2 and area.y2 == y2:
                return True
        return False
    
    def _get_current_area_coords(self, robot):
        x1 = (robot.x // self.width) * self.width
        y1 = (robot.y // self.height) * self.height
        x2 = x1 + self.width
        y2 = y1 + self.height
        return x1, y1, x2, y2

    def _discover_new_area(self, robot):
        x1, y1, x2, y2 = self._get_current_area_coords(robot)

        if not self._is_area_discovered(x1, y1, x2, y2):
            self._add_area(x1, y1)
            robot.reward += 50

    def _draw_map(self, screen, robot, time):
        offset_x = self.width // 2 - robot.x
        offset_y = self.height // 2 - robot.y

        for area in self.areas.values():
            rect = pygame.Rect(area.x1 + offset_x, area.y1 + offset_y, self.width, self.height)
            pygame.draw.rect(screen, Colors.BLACK, rect, 2)

            for obstacle in area.obstacles:
                if obstacle.visible:
                    pygame.draw.rect(screen, Colors.RED, (obstacle.x + offset_x, obstacle.y + offset_y, obstacle.size, obstacle.size))

        pygame.draw.circle(screen, Colors.BLUE, (self.width // 2, self.height // 2), robot.radius)


class Area:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.obstacles = self._generate_obstacles(50)

    def _generate_obstacles(self, obstacle_size, n=10):
    ### area (4n-tuples): area where to generate the obstacles
    ### n (int): number of obstacles to generate
        obstacles = []
        for _ in range(n):
            x = random.randint(self.x1, self.x2 - obstacle_size)
            y = random.randint(self.y1, self.y2 - obstacle_size)
            obstacles.append(Obstacle(x, y, obstacle_size))
        return obstacles

class Obstacle:
    def __init__(self, x, y, obstacle_size):
        self.x = x
        self.y = y
        self.size = obstacle_size
        self.visible = True

class Robot:
    def __init__(self, x, y, speed=5, radius=20):
        self.x = x
        self.y = y

        self.speed = speed
        self.radius = radius
        self.reward = 0
        self.best_reward = 0

        self.last_moves = []

        self.direction = Direction.UP
        self.total_moves = 0
        self.best_total_moves = 0

    def _update_robot_position(self, action, bias):
        # AI CONTROL
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)

        if bias:
            new_dir = clock_wise[random.choice([x for x in range(4) if x != idx])]
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = clock_wise[(idx - 1) % 4]
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = clock_wise[(idx + 2) % 4]
        else:
            new_dir = clock_wise[idx]

        self.direction = new_dir
        if self.direction == Direction.UP:
            self.y -= self.speed
            self.total_moves += 1
        elif self.direction == Direction.DOWN:
            self.y += self.speed
            self.total_moves += 1
        elif self.direction == Direction.LEFT:
            self.x -= self.speed
            self.total_moves += 1
        elif self.direction == Direction.RIGHT:
            self.x += self.speed
            self.total_moves += 1

        self.last_moves.append(self.direction)
        if len(self.last_moves) > 4:
            self.last_moves.pop(0)

        self._check_bad_moves()
        self._alive_reward()

        # MANUAL CONTROL
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_UP]:
        #     self.y -= self.speed
        #     self.total_moves += 1
        # elif keys[pygame.K_DOWN]:
        #     self.y += self.speed
        #     self.total_moves += 1
        # if keys[pygame.K_LEFT]:
        #     self.x -= self.speed
        #     self.total_moves += 1
        # elif keys[pygame.K_RIGHT]:
        #     self.x += self.speed
        #     self.total_moves += 1

    def _check_bad_moves(self):
        if len(self.last_moves) == 4:
            # back and forth
            if (self.last_moves[:3] == [Direction.UP, Direction.DOWN, Direction.UP] or \
                self.last_moves[:3] == [Direction.DOWN, Direction.UP, Direction.DOWN] or \
                self.last_moves[:3] == [Direction.LEFT, Direction.RIGHT, Direction.LEFT] or \
                self.last_moves[:3] == [Direction.RIGHT, Direction.LEFT, Direction.RIGHT]):
                self.reward -= 5

            # clockwise loop
            elif (self.last_moves == [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT] or \
                  self.last_moves == [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] or \
                  self.last_moves == [Direction.DOWN, Direction.LEFT, Direction.UP, Direction.RIGHT] or \
                  self.last_moves == [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN]):
                self.reward -= 10

            # counter clockwise loop
            elif (self.last_moves == [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT] or \
                  self.last_moves == [Direction.LEFT, Direction.DOWN, Direction.RIGHT, Direction.UP] or \
                  self.last_moves == [Direction.DOWN, Direction.RIGHT, Direction.UP, Direction.LEFT] or \
                  self.last_moves == [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]):
                self.reward -= 10

    def _alive_reward(self):
        self.reward += 1
        if self.total_moves > self.best_total_moves and self.reward > self.best_reward:
            self.reward += 100
            self.best_total_moves = self.total_moves
            self.best_reward = self.reward

    def collision_detection(self, game, action=False):
        for area in game.map.areas.values():
            for obstacle in area.obstacles:
                if (self.x - self.radius < obstacle.x + obstacle.size and 
                    self.x + self.radius > obstacle.x and 
                    self.y - self.radius < obstacle.y + obstacle.size and 
                    self.y + self.radius > obstacle.y):
                    self.reward -= 20
                    if action:
                        obstacle.visible = True
                        game._reset_robot() 
                    return True
        return False


if __name__ == "__main__":
    WIDTH = 600
    HEIGHT = 600

    game = Game(WIDTH, HEIGHT)

    while True:
        stop, score = game.play_step()
    
        if stop:
            break

    print("Final score:", score)

    pygame.quit()