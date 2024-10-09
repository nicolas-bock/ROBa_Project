import pygame
import sys
import random

SCREEN_SIZE = WIDTH, HEIGHT = 800, 600

# Colors
BLACK = 0, 0, 0
WHITE = 255, 255, 255
BLUE = 0, 0, 255
RED = 255, 0, 0
GREEN = 0, 255, 0

# Objects
ROBOT_SIZE = 30
OBSTACLE_SIZE = 50
MOVE_SPEED = 5
ROBOT_MOVED = False

# Initialization
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Robot Motions")
clock = pygame.time.Clock()

# Robot position
robot_coords = [WIDTH//2, HEIGHT//2]

# Obstacles
obstacles = []
for _ in range(5):
    x = random.randint(0, WIDTH - OBSTACLE_SIZE)
    y = random.randint(0, HEIGHT - OBSTACLE_SIZE)
    obstacle_coords = [x, y]
    obstacles.append(obstacle_coords)

# Offset for moving the world
world_offset_x, world_offset_y = 0, 0

def generate_new_obstacles():
    new_obstacles = []
    for _ in range(5):
        side = random.choice(['left', 'right', 'top', 'bottom'])
        if side == 'left':
            x = random.randint(-WIDTH, -OBSTACLE_SIZE)
            y = random.randint(0, HEIGHT - OBSTACLE_SIZE)
        elif side == 'right':
            x = random.randint(WIDTH, WIDTH * 2 - OBSTACLE_SIZE)
            y = random.randint(0, HEIGHT - OBSTACLE_SIZE)
        elif side == 'top':
            x = random.randint(0, WIDTH - OBSTACLE_SIZE)
            y = random.randint(-HEIGHT, -OBSTACLE_SIZE)
        elif side == 'bottom':
            x = random.randint(0, WIDTH - OBSTACLE_SIZE)
            y = random.randint(HEIGHT, HEIGHT * 2 - OBSTACLE_SIZE)

        new_obstacles.append([x + world_offset_x, y + world_offset_y])
    return new_obstacles

# Main Loop
while True:
    screen.fill(WHITE)

    # Draw Robot
    robot_x, robot_y = robot_coords
    pygame.draw.circle(screen, BLUE, robot_coords, ROBOT_SIZE)

    # Draw Obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, RED, (obstacle[0] - world_offset_x, obstacle[1] - world_offset_y, OBSTACLE_SIZE, OBSTACLE_SIZE))

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Robot Motions
    keys = pygame.key.get_pressed()

    move_x, move_y = 0, 0
    if keys[pygame.K_UP]:
        world_offset_y -= MOVE_SPEED
        ROBOT_MOVED = True
    elif keys[pygame.K_DOWN]:
        world_offset_y += MOVE_SPEED
        ROBOT_MOVED = True
    if keys[pygame.K_LEFT]:
        world_offset_x -= MOVE_SPEED
        ROBOT_MOVED = True
    elif keys[pygame.K_RIGHT]:
        world_offset_x += MOVE_SPEED
        ROBOT_MOVED = True

    # Generate new obstacles if robot moves to the edge of the window
    if (ROBOT_MOVED and 
        (((world_offset_x % (WIDTH//2)) == 0 and world_offset_x > 0) or
        ((world_offset_x % (WIDTH//2)) == 0 and world_offset_x < 0) or
        ((world_offset_y % (HEIGHT//2)) == 0 and world_offset_y > 0) or
        ((world_offset_y % (HEIGHT//2)) == 0 and world_offset_y < 0))):

        obstacles.extend(generate_new_obstacles())
        ROBOT_MOVED = False

    # Collision Detection
    for obstacle in obstacles:
        obstacle_x, obstacle_y = obstacle[0] - world_offset_x, obstacle[1] - world_offset_y
        if (robot_coords[0] + ROBOT_SIZE > obstacle_x and
            robot_coords[0] - ROBOT_SIZE < obstacle_x + OBSTACLE_SIZE and
            robot_coords[1] + ROBOT_SIZE > obstacle_y and
            robot_coords[1] - ROBOT_SIZE < obstacle_y + OBSTACLE_SIZE):
            
            # Reset the world if collision happens
            world_offset_x, world_offset_y = 0, 0
            obstacles = []
            for _ in range(5):
                x = random.randint(0, WIDTH - OBSTACLE_SIZE)
                y = random.randint(0, HEIGHT - OBSTACLE_SIZE)
                obstacles.append([x, y])
        
    pygame.display.flip()
    clock.tick(60)