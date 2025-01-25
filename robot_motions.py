import pygame
import sys
import random
import math
import numpy as np
from enum import Enum
from maps import scale_map, scale_doors, MAP_1, DOORS, ORIGINAL_DIMENSIONS

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
    CYAN = 0, 255, 255

class Game:
    def __init__(self, width=600, height=600, generated_map=None, generated_doors=None, nb_particles=0):
        if generated_map:
            self.width, self.height = self.get_map_size(generated_map)
            self.map = Map(width=self.width, height=self.height, generated_map=generated_map, generated_doors=generated_doors)
        else:
            self.width = width
            self.height = height
            self.generate_map(generated_map, generated_doors)
        
        # Initialize Game and Clock
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.time = pygame.time
        self.score = 0
        
        # Initialize the robot
        self.robot = Robot(self.width // 2, self.height // 2, speed=5, radius=20)

        self.nb_particles = nb_particles

    def get_map_size(self, map):
        width = 0
        height = 0
        for key in map:
            for x, y, _ in map[key]:
                width = max(width, x + 600)
                height = max(height, y + 600)
        return width, height
    
    def play_step(self, action, bias=False):
        self.screen.fill(Colors.WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return self.robot.reward, True, self.score
        # Update the robot position and check for collisions
        self.robot._update_robot_position(action, bias)
        collision = self.robot.collision_detection(self, True)

        game_over = collision or self.robot.reward < -1000

        self.map._discover_new_area(self.robot)
        self.map._draw_map(self.screen, self.robot)

        pygame.display.flip()
        self.time.Clock().tick(60)

        return self.robot.reward, game_over, self.score
    
    # Reset the robot position along with the score
    def _reset_robot(self):
        # Set the robot's position to the center of the map (not colliding with obstacles or walls)
        self.robot.x = self.width // 2 + self.robot.radius
        self.robot.y = self.height // 2 + self.robot.radius
        self.robot.direction = Direction.UP

        reward_per_move = self.robot.reward // self.robot.total_moves if self.robot.total_moves > 0 else 0
        self.score = (self.robot.total_moves + reward_per_move) if self.robot.total_moves + reward_per_move > 0 else 0
        self.robot.total_moves = 0

    def reset(self):
        self.robot.reward = 0
        self._reset_robot()
        self.score = 0

    def generate_map(self, map, doors):
        self.map = Map(width=self.width, height=self.height, generated_map=map, generated_doors=doors)

class Map:
    def __init__(self, width, height, generated_map=None, generated_doors=None):
        self.width = width
        self.height = height

        self.areas = {}
        self.border_x1 = 0
        self.border_y1 = 0
        self.border_x2 = self.width
        self.border_y2 = self.height

        self.doors = generated_doors

        if generated_map:
            self._load_generated_map(generated_map)
        else:
            ValueError("Provide a map")

    def _load_generated_map(self, map):
        # Load the obstacles into the map
        for area_coords, obstacle_data in map.items():
            x_offset, y_offset = area_coords
            area = Area(x_offset, y_offset, x_offset + self.width, y_offset + self.height, obstacle_data)
            self.areas[area_coords] = area

    # def _add_area(self, x_offset, y_offset):
    #     area_coords = (x_offset, y_offset)
    #     if area_coords not in self.areas:
    #         new_area = Area(x_offset, y_offset, x_offset + self.width, y_offset + self.height)
    #         self.areas[area_coords] = new_area
    # 
    #         self.border_x1 = min(self.border_x1, x_offset)
    #         self.border_y1 = min(self.border_y1, y_offset
    #         self.border_x2 = max(self.border_x2, x_offset + self.width)
    #         self.border_y2 = max(self.border_y2, y_offset + self.height)

    def _is_area_discovered(self, x1, y1, x2, y2):
        # Check whether the area has been already visited or not
        for area in self.areas.values():
            if area.x1 == x1 and area.y1 == y1 and area.x2 == x2 and area.y2 == y2:
                return True
        return False
    # Retrieve the area the robot is currently in
    def _get_current_area_coords(self, robot):
        x1 = (robot.x // self.width) * self.width
        y1 = (robot.y // self.height) * self.height
        x2 = x1 + self.width
        y2 = y1 + self.height
        return x1, y1, x2, y2

    def _discover_new_area(self, robot):
        x1, y1, x2, y2 = self._get_current_area_coords(robot)
        # Reward the robot for discovering new areas
        if not self._is_area_discovered(x1, y1, x2, y2):
        #     self._add_area(x1, y1)
            robot.reward += 50

    def _show_obstacles(self):
        for area in self.areas.values():
            for obstacle in area.obstacles:
                obstacle.visible = True

    def _draw_map(self, screen, robot):
        offset_x = self.width // 2 - robot.x
        offset_y = self.height // 2 - robot.y

        # Draws black rectangles as the wall of each areas
        for area in self.areas.values():
            rect = pygame.Rect(area.x1 + offset_x, area.y1 + offset_y, self.width, self.height)
            pygame.draw.rect(screen, Colors.BLACK, rect, 2)

        # Draw doors as green lines
        for door in self.doors.keys():
            x1, y1, x2, y2 = door
            pygame.draw.line(screen, Colors.GREEN, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), 5)

        # Draw obstacles as red squares
        for area in self.areas.values():
            for obstacle in area.obstacles:
                if obstacle.visible:
                    pygame.draw.rect(screen, Colors.RED, (obstacle.x + offset_x, obstacle.y + offset_y, obstacle.size, obstacle.size))

        # Draw the robot
        pygame.draw.circle(screen, Colors.BLUE, (self.width // 2, self.height // 2), robot.radius)

        # Draw the robot's sensors as green beams
        theta = robot.convert()
        measurements = robot.scanning(self)
        arc_angle = math.pi / 2
        angle_increment = arc_angle / (len(measurements) - 1)
        start_angle = theta - arc_angle / 2
        
        for i, distance in enumerate(measurements):
            ray_angle = start_angle + i * angle_increment

            # Calculate the endpoint of the beam based on the angle and the distance
            end_x = robot.x + math.cos(ray_angle) * distance
            end_y = robot.y + math.sin(ray_angle) * distance

            # Draw the beam starting from the robot's position
            pygame.draw.line(screen, Colors.GREEN, (robot.x + offset_x, robot.y + offset_y), (end_x + offset_x, end_y + offset_y), 1)

class Area:
    def __init__(self, x1, y1, x2, y2, obstacle_data=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
        if obstacle_data:
            self.obstacles = self._generate_obstacles(obstacle_data)
        else:
            self.obstacles = []

    def _generate_obstacles(self, obstacle_data):
        obstacles = []
        for obs in obstacle_data:
            x, y, size = obs
            obstacles.append(Obstacle(x, y, size))
        return obstacles

class Obstacle:
    def __init__(self, x, y, obstacle_size):
        self.x = x
        self.y = y
        self.size = obstacle_size
        self.visible = False

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
        
    # Update the robot position based on input action
    def _update_robot_position(self, action=None, bias=False):
        # AI CONTROL
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)

        # if bias:
        #     new_dir = clock_wise[random.choice([x for x in range(4) if x != idx])]
        # elif np.array_equal(action, [0, 1, 0, 0]):
        #     new_dir = clock_wise[(idx + 1) % 4]
        # elif np.array_equal(action, [0, 0, 1, 0]):
        #     new_dir = clock_wise[(idx - 1) % 4]
        # elif np.array_equal(action, [0, 0, 0, 1]):
        #     new_dir = clock_wise[(idx + 2) % 4]
        # else:
        #     new_dir = clock_wise[idx]

        # self.direction = new_dir
        # if self.direction == Direction.UP:
        #     self.y -= self.speed
        #     self.total_moves += 1
        # elif self.direction == Direction.DOWN:
        #     self.y += self.speed
        #     self.total_moves += 1
        # elif self.direction == Direction.LEFT:
        #     self.x -= self.speed
        #     self.total_moves += 1
        # elif self.direction == Direction.RIGHT:
        #     self.x += self.speed
        #     self.total_moves += 1

        # self.last_moves.append(self.direction)
        # if len(self.last_moves) > 4:
        #     self.last_moves.pop(0)

        # self._check_bad_moves()
        # self._alive_reward()

        # MANUAL CONTROL
        keys = pygame.key.get_pressed()
        robot_moved = False
        if keys[pygame.K_UP]:
            self.direction = Direction.UP
            self.y -= self.speed
            self.total_moves += 1
            robot_moved = True
        elif keys[pygame.K_DOWN]:
            self.direction = Direction.DOWN
            self.y += self.speed
            self.total_moves += 1
            robot_moved = True
        elif keys[pygame.K_LEFT]:
            self.direction = Direction.LEFT
            self.x -= self.speed
            self.total_moves += 1
            robot_moved = True
        elif keys[pygame.K_RIGHT]:
            self.direction = Direction.RIGHT
            self.x += self.speed
            self.total_moves += 1
            robot_moved = True

        return robot_moved

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

    # Reward the robot for staying alive
    def _alive_reward(self):
        self.reward += 1
        if self.total_moves > self.best_total_moves and self.reward > self.best_reward:
            self.reward += 100
            self.best_total_moves = self.total_moves
            self.best_reward = self.reward

    # Check if the robot collides with obstacles or walls and handle out of bounds cases
    def collision_detection(self, game, action=False):
        current_area = None
        # Retrieve the robot's current location
        for area_coords, area in game.map.areas.items():
            if area.x1 <= self.x <= area.x2 and area.y1 <= self.y <= area.y2:
                current_area = area
                break
        # Reset if the robot is not in a valid area
        if not current_area:
            if action:
                self.reward -= 50
                game._reset_robot()
            return True
        # Check for collision with obstacles
        for obstacle in current_area.obstacles:
            if (
                self.x - self.radius < obstacle.x + obstacle.size and
                self.x + self.radius > obstacle.x and
                self.y - self.radius < obstacle.y + obstacle.size and
                self.y + self.radius > obstacle.y
            ):
                self.reward -= 20
                if action:
                    obstacle.visible = True
                    game._reset_robot()
                return True
        # Check for collision with walls
        wall_collision = (
            (self.x - self.radius < current_area.x1 and not self._is_on_door(game, current_area.x1, "vertical")) or
            (self.x + self.radius > current_area.x2 and not self._is_on_door(game, current_area.x2, "vertical")) or
            (self.y - self.radius < current_area.y1 and not self._is_on_door(game, current_area.y1, "horizontal")) or
            (self.y + self.radius > current_area.y2 and not self._is_on_door(game, current_area.y2, "horizontal"))
        )
        if wall_collision:
            self.reward -= 50
            game._reset_robot()
            return True

        return False
    
    # Check if the robot is on a door (Green Line)
    def _is_on_door(self, game, coord, orientation):
        for door in game.map.doors.keys():
            # Verify for Vertical doors
            if orientation == "vertical":
                if door[0] == door[2] == coord and door[1] <= self.y <= door[3]:
                    return True
            # Verify for Horizontal doors
            elif orientation == "horizontal":
                if door[1] == door[3] == coord and door[0] <= self.x <= door[2]:
                    return True
        return False

    # Simulate a beam to detect obstacles in a straight line
    def beam_cast(self, x, y, theta, max, areas):
        # Starts at max range
        closest_distance = max
        for area_coords, area in areas.items():
            for obstacle in area.obstacles:
                # Calculate the intersection of the ray with an obstacle
                distance = self.calc_intersection(x, y, theta, obstacle)
                if distance and distance < closest_distance:
                    closest_distance = distance
                    obstacle.visible = True

            # Check for intersections with the area's walls
            if area.x1 <= x <= area.x2 and area.y1 <= y <= area.y2:
                for side in [(area.x1, area.y1, area.x2, area.y1),
                            (area.x2, area.y1, area.x2, area.y2),
                            (area.x1, area.y2, area.x2, area.y2),
                            (area.x1, area.y1, area.x1, area.y2)]:
                # Check for intersections with every sides of the area
                    ix, iy = self.det_intersection((x, y, x + math.cos(theta) * 1000, y + math.sin(theta) * 1000), side)
                    if ix is not None and iy is not None:
                        distance = math.hypot(ix - x, iy - y)
                        if distance < closest_distance:
                            closest_distance = distance

        return closest_distance

    # Simulate the sensor by casting multiple beams within a given angle
    def calc_arc(self, x, y, theta, arc_angle=math.pi / 2, num_rays=8, max=200, areas=None):
        measurements = []
        angle_increment = arc_angle / (num_rays - 1)
        start_angle = theta - arc_angle / 2
        # Cast beams at incremental angles within the beam arc
        for i in range(num_rays):
            ray_angle = start_angle + i * angle_increment
            distance = self.beam_cast(x, y, ray_angle, max, areas)
            measurements.append(distance)

        return measurements
    
    # Calculate the intersection of a beam with an obstacle
    def calc_intersection(self, x, y, theta, obstacle):
        obs_x, obs_y, obs_size = obstacle.x, obstacle.y, obstacle.size
        obs_rect = pygame.Rect(obs_x, obs_y, obs_size, obs_size)
        dx = math.cos(theta)
        dy = math.sin(theta)
        min_dist = None
        # Check for intersections with every sides of the obstacle
        for side in [(obs_x, obs_y, obs_x + obs_size, obs_y),
                    (obs_x + obs_size, obs_y, obs_x + obs_size, obs_y + obs_size),
                    (obs_x, obs_y + obs_size, obs_x + obs_size, obs_y + obs_size),
                    (obs_x, obs_y, obs_x, obs_y + obs_size)]:
            
            ix, iy = self.det_intersection((x, y, x + dx * 1000, y + dy * 1000), side)
            if ix is not None and iy is not None:
                distance = math.hypot(ix - x, iy - y)
                if min_dist is None or distance < min_dist:
                    min_dist = distance

        return min_dist
    
    # Determine the intersection point of two line
    def det_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        # Handle exceptions
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None, None
        
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        # Verify if the intersection point is within the bounds of both line
        if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
                min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
            return px, py

        return None, None
    
    # Convert a direction into an angle in radians
    def convert(self):
        if self.direction == Direction.DOWN:
            return math.pi / 2
        elif self.direction == Direction.RIGHT:
            return 0
        elif self.direction == Direction.UP:
            return -math.pi / 2
        elif self.direction == Direction.LEFT:
            return math.pi
        else:
            ValueError("Unknown direction")

    # Simulate the robot's sensor using an arc of beams
    def scanning(self, map):
        theta = self.convert()
        measurements = self.calc_arc(self.x, self.y, theta, arc_angle=math.pi/2, num_rays=8, max=200, areas=map.areas)
        return measurements
    
class MonteCarlo:

    def __init__(self, steps, map):
        self.steps = steps
        self.particles = []
        self.nb_particles_per_step = []
        self.max_weight = 0.0
        self.id_best_particle = None
        self.map = map

    def _init_particles(self, nb_particles, cost_map):
        '''
        Initialize the particles in the map.
        The cost map is needed because we do not want to put particles in a non obstacle free cell
        '''
        width, height = self.map.width, self.map.height

        for _ in range(nb_particles):
            x = random.randint(0, width)
            y = random.randint(0, height)
            while cost_map[y][x] != 0:
                x = random.randint(0, width)
                y = random.randint(0, height)
            self.particles.append(Particle(x, y, 1 / nb_particles))

        self.nb_particles_per_step.append(nb_particles)
    
    def _display_particles(self, game):
        '''
        Display the particles on the map.
        '''
        offset_x = game.width // 2 - game.robot.x
        offset_y = game.height // 2 - game.robot.y

        for particle in self.particles:
            pygame.draw.circle(game.screen, Colors.CYAN, (particle.x + offset_x, particle.y + offset_y), 4)
        pygame.display.flip()

    def _evaluate_particles(self, robot):
        '''
        Evaluate the particles based on the sensor data.
        '''
        for particle in self.particles:
            min_distance = float('inf')
            for area in self.map.areas.values():
                for obstacle in area.obstacles:
                    distance = np.sqrt((particle.x - obstacle.x) ** 2 + (particle.y - obstacle.y) ** 2)
                    min_distance = min(min_distance, distance)
            
            measured_distances = robot.scanning(self.map)
            particle.weight = np.exp(-((min_distance - np.min(measured_distances)) ** 2) / (2 * 10 ** 2))

    def _resample_particles(self):
        '''
        Resample the particles based on their weight.
        '''
        new_particles = []
        weights = [particle.weight for particle in self.particles]
        weights_sum = sum(weights)
        weights = [weight / weights_sum for weight in weights]

        for _ in range(len(self.particles)):
            new_particles.append(self._resample(weights))

        self.particles = new_particles

    def _resample(self, weights):
        '''
        Resample a particle based on its weight.
        '''
        random_number = random.random()
        cumulative_sum = 0.0
        for i, weight in enumerate(weights):
            cumulative_sum += weight
            if cumulative_sum > random_number:
                return Particle(self.particles[i].x, self.particles[i].y, 1 / len(self.particles))

class Particle:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight

    def __repr__(self):
        return f'Particle({self.x}, {self.y}, {self.weight})'
    
class Cost_Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cost_map = [[0 for _ in range(width)] for _ in range(height)]

    def _update_cost_map(self, game):
        for area in game.map.areas.values():
            for obstacle in area.obstacles:
                self.cost_map[obstacle.y][obstacle.x] = 1

    def __getitem__(self, index):
        return self.cost_map[index]
    
    def __setitem__(self, index, value):
        self.cost_map[index] = value


if __name__ == '__main__':
    pygame.init()

    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h

    scaled_map = scale_map(screen_width, screen_height, MAP_1, ORIGINAL_DIMENSIONS)
    scaled_doors = scale_doors(screen_width, screen_height, DOORS, ORIGINAL_DIMENSIONS)

    game = Game(generated_map=scaled_map, generated_doors=scaled_doors, nb_particles=100)

    cost_map = Cost_Map(game.width, game.height)
    cost_map._update_cost_map(game)

    mc = MonteCarlo(100, game.map)
    mc._init_particles(500, cost_map)

    game._reset_robot()

    game.screen.fill(Colors.WHITE)
    game.map._draw_map(game.screen, game.robot)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if game.robot._update_robot_position():
            game.screen.fill(Colors.WHITE)
            game.map._draw_map(game.screen, game.robot)
        
            mc._display_particles(game)
            mc._evaluate_particles(game.robot)
            mc._resample_particles()

        game.time.Clock().tick(60)
        pygame.display.flip()

    pygame.quit()