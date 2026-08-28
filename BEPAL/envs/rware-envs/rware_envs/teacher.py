import rware
import gym
import numpy as np
import time
import collections

def bfs(grid, start, goal):
    queue = collections.deque([[start]])
    seen = set([start])
    height, width = len(grid), len(grid[0])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == goal:
            yield path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] == 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))

def best_path(env, agent, goal_location_x, goal_location_y, grid_map=None):
    current_agent = agent
    current_location_x = current_agent.x
    current_location_y = current_agent.y
    
    if grid_map is None:
        grid_map = env.grid[1].copy()

        for agent in env.agents:
            grid_map[agent.y][agent.x] = 1  # Mark agent positions as obstacles

    return bfs(grid_map, (current_location_x, current_location_y), (goal_location_x, goal_location_y))
    
def turn_to_face(env, current_location_x, current_location_y, goal_location_x, goal_location_y, agent):
    current_agent = agent
    direction = current_agent.dir.value

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    grid = env.grid
    targets = env.request_queue

    # If facing in the right direction, return
    if (current_location_x == goal_location_x) and (current_location_y == goal_location_y):
        return 0 # Done
    
    if (current_location_y > goal_location_y) and (direction != UP):
        if (direction == LEFT):
            return 3 # Turn Right
        else:
            return 2 # Turn Left
        
    elif (current_location_y < goal_location_y) and (direction != DOWN):
        if (direction == LEFT):
            return 2 # Turn Left
        else:
            return 3 # Turn Right
        
    elif (current_location_x < goal_location_x) and (direction != RIGHT):
        if (direction == UP):
            return 3 # Turn Right
        else:
            return 2 # Turn Left
        
    elif (current_location_x > goal_location_x) and (direction != LEFT):
        if (direction == UP):
            return 2 # Turn Left
        else:
            return 3 # Turn Right
    else:
        return 1 # Forward

def move(env):
    action = [0] * len(env.agents)  # Initialize action for each agent
    for agent in env.agents:
        is_on_target = False
        for request in env.request_queue:
            if agent.x == request.x and agent.y == request.y:
                is_on_target = True
                break
        
        # If agent is not carrying a shelf and is not on target, make it walk towards the target
        if agent.carrying_shelf is None and not is_on_target:
            # Find the closest target not being carried by another agent
            closest_target = None
            closest_distance = float('inf')
            for request in env.request_queue:
                skip = False
                # Check if the request is being carried by another agent
                for other_agent in env.agents:
                    if other_agent.carrying_shelf == request.id and other_agent.id != agent.id:
                        # This request is being carried by another agent, skip it
                        skip = True
                        break
                
                if skip:
                    continue

                distance = abs(agent.x - request.x) + abs(agent.y - request.y)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_target = request
                
            if closest_target is not None:
                path = list(best_path(env, agent, closest_target.x, closest_target.y, env.grid[0].copy()))
                if (len(path) != 0 and len(path[0]) > 1):
                    path = path[0][1]
                    turn = turn_to_face(env, agent.x, agent.y, path[0], path[1], agent)

                    action[agent.id - 1] = turn
                else:
                    # Take a random action if no path is found
                    action[agent.id - 1] = np.random.choice([0, 1, 2, 3])
                

        # If agent is not carrying a shelf and is on target, pick up shelf
        if agent.carrying_shelf is None and is_on_target:
            action[agent.id - 1] = 4

        # Agent is carrying and is on target
        if agent.carrying_shelf is not None and is_on_target:
            # Make agent walk towards closest dropoff point
            closet_dropoff = None
            closet_distance = float('inf')
            for dropoff in env.goals:
                distance = abs(agent.x - dropoff[0]) + abs(agent.y - dropoff[1])
                if distance < closet_distance:
                    closet_distance = distance
                    closet_dropoff = dropoff
            
            dropoff_x = closet_dropoff[0]
            dropoff_y = closet_dropoff[1]

            path = list(best_path(env, agent, dropoff_x, dropoff_y))

            if (len(path) != 0 and len(path[0]) > 1):
                path = path[0][1]
                turn = turn_to_face(env, agent.x, agent.y, path[0], path[1], agent)

                action[agent.id - 1] = turn
            else:
                # Take a random action if no path is found
                action[agent.id - 1] = np.random.choice([0, 1, 2, 3])

        # Agent is carrying a shelf and is not on target
        if agent.carrying_shelf is not None and not is_on_target:
            # Make agent return box
            highways = env.highways
            highways = np.where((highways == 0)|(highways == 1), highways^1, highways)

            # Agent is on a highway tile, return box
            if highways[agent.y][agent.x] == 1:
                action[agent.id - 1] = 4
            # Agent is not on a highway tile, find closest highway tile
            else:
                open_tiles = []
                for y in range(len(highways)):
                    for x in range(len(highways[0])):
                        if highways[y][x] == 1 and env.grid[1][y][x] == 0:
                            # Found a highway tile
                            open_tiles.append((x, y))
                
                if len(open_tiles) > 0:
                    # Find the closest highway tile to the agent
                    closest_tile = min(open_tiles, key=lambda tile: abs(tile[0] - agent.x) + abs(tile[1] - agent.y))
                    path = list(best_path(env, agent, closest_tile[0], closest_tile[1]))
                    if (len(path) != 0 and len(path[0]) > 1):
                        path = path[0][1]
                        turn = turn_to_face(env, agent.x, agent.y, path[0], path[1], agent)

                        action[agent.id - 1] = turn
                    else:
                        # Take a random action if no path is found
                        action[agent.id - 1] = np.random.choice([0, 1, 2, 3])
    
    final_action = np.zeros((len(env.agents), 5))
    
    for i in range(len(action)):
        final_action[i][action[i]] = 100

    return final_action