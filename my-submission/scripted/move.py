import numpy as np
import random as rand
from queue import PriorityQueue, Queue
import nmmo
from nmmo.lib import material

#from ijcai2022nmmo.scripted import utils
import scripted.utils as utils


def adjacentPos(pos):
    r, c = pos
    return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]


def inSight(dr, dc, vision):
    return (dr >= -vision and dc >= -vision and dr <= vision and dc <= vision)


def vacant(tile):
    Tile = nmmo.Serialized.Tile
    occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)
    matl = nmmo.scripting.Observation.attribute(tile, Tile.Index)

    return matl in material.Habitable and not occupied # material.Habitable: grass, scrub, forest


def random(config, ob, actions):
    direction = rand.choice(nmmo.action.Direction.edges)
    actions[nmmo.action.Move] = {nmmo.action.Direction: direction} # channge the parameter 'actions' itself.


def towards(direction):
    if direction == (-1, 0):
        return nmmo.action.North
    elif direction == (1, 0):
        return nmmo.action.South
    elif direction == (0, -1):
        return nmmo.action.West
    elif direction == (0, 1):
        return nmmo.action.East
    else:
        # return rand.choice(nmmo.action.Direction.edges) # Possible reason for jumping into Lava
        return None


def pathfind(config, ob, actions, rr, cc):
    direction = aStar(config, ob, actions, rr, cc)
    direction = towards(direction) # turn the direction tuple to action.Direction object
    if direction is not None:
        actions[nmmo.action.Move] = {nmmo.action.Direction: direction}


def meander(config, ob, actions):
    agent = ob.agent
    Entity = nmmo.Serialized.Entity
    Tile = nmmo.Serialized.Tile

    r = nmmo.scripting.Observation.attribute(agent, Entity.R)
    c = nmmo.scripting.Observation.attribute(agent, Entity.C)

    cands = []
    if vacant(ob.tile(-1, 0)):
        cands.append((-1, 0))
    if vacant(ob.tile(1, 0)):
        cands.append((1, 0))
    if vacant(ob.tile(0, -1)):
        cands.append((0, -1))
    if vacant(ob.tile(0, 1)):
        cands.append((0, 1))
    if not cands:
        return (-1, 0)

    direction = rand.choices(cands)[0]
    direction = towards(direction)
    actions[nmmo.action.Move] = {nmmo.action.Direction: direction}


def explore(config, ob, actions, spawnR, spawnC):
    vision = config.NSTIM
    sz = config.TERRAIN_SIZE
    Entity = nmmo.Serialized.Entity
    Tile = nmmo.Serialized.Tile

    agent = ob.agent
    r = nmmo.scripting.Observation.attribute(agent, Entity.R)
    c = nmmo.scripting.Observation.attribute(agent, Entity.C)
    
    # TODO: more efficient exploration
    centR, centC = sz // 2, sz // 2
    vR, vC = centR - spawnR, centC - spawnC

    mmag = max(abs(vR), abs(vC))
    rr = int(np.round(vision * vR / mmag))
    cc = int(np.round(vision * vC / mmag))
    pathfind(config, ob, actions, rr, cc)


def explore_square(config, ob, actions, spawnR, spawnC, current_target):
    '''explore in counter-clockwise or clockwise direction'''
    entID = nmmo.scripting.Observation.attribute(ob.agent, nmmo.Serialized.Entity.ID)
    pop = nmmo.scripting.Observation.attribute(ob.agent, nmmo.Serialized.Entity.Population)
    r = nmmo.scripting.Observation.attribute(ob.agent, nmmo.Serialized.Entity.R)
    c = nmmo.scripting.Observation.attribute(ob.agent, nmmo.Serialized.Entity.C)

    inSquadOne = utils.inSquadOne(entID, pop)
    rr, cc, current_target = squad_target(config, ob, actions, spawnR, spawnC, r, c, inSquadOne, current_target)
    pathfind(config, ob, actions, rr, cc)

    return current_target


def squad_target(config, ob, actions, spawnR, spawnC, r, c, inSquadOne, current_target):
    vision = config.NSTIM

    UP_LEFT = (16, 16)
    UP_RIGHT = (16, 128)
    DOWN_LEFT = (128, 16)
    DOWN_RIGHT = (128, 128)

    targetsList = [UP_LEFT, DOWN_LEFT, DOWN_RIGHT, UP_RIGHT]
    spawnLeftBottom = utils.spawnLeftBottom(config, spawnR, spawnC)
    current_pos = (r, c)

    if current_target is None:
        # initial position
        if spawnC <= config.TERRAIN_BORDER:
            if inSquadOne:
                targR, targC = UP_LEFT
            elif not inSquadOne:
                targR, targC = DOWN_LEFT
        
        if spawnC >= config.TERRAIN_BORDER + config.TERRAIN_CENTER:
            if inSquadOne:
                targR, targC = UP_RIGHT
            elif not inSquadOne:
                targR, targC = DOWN_RIGHT
        
        if spawnR <= config.TERRAIN_BORDER:
            if inSquadOne:
                targR, targC = UP_LEFT
            elif not inSquadOne:
                targR, targC = UP_RIGHT
        
        if spawnR >= config.TERRAIN_BORDER + config.TERRAIN_CENTER:
            if inSquadOne:
                targR, targC = DOWN_LEFT
            elif not inSquadOne:
                targR, targC = DOWN_RIGHT

        current_target = (targR, targC)

    else:
        if goal_reached(current_pos, current_target):
            current_target = utils.nextTarget(current_target, targetsList, inSquadOne, spawnLeftBottom)

    targR, targC = current_target
    vR, vC = targR - spawnR, targC - spawnC
    mmag = max(abs(vR), abs(vC))
    rr = int(np.round(vision * vR / mmag))
    cc = int(np.round(vision * vC / mmag))

    return rr, cc, current_target


def goal_reached(start, goal, bar=4):
    return utils.lInfty(start, goal) <= bar


def evade(config, ob, actions, attacker):
    Entity = nmmo.Serialized.Entity

    sr = nmmo.scripting.Observation.attribute(ob.agent, Entity.R)
    sc = nmmo.scripting.Observation.attribute(ob.agent, Entity.C)

    gr = nmmo.scripting.Observation.attribute(attacker, Entity.R)
    gc = nmmo.scripting.Observation.attribute(attacker, Entity.C)

    # TODO: intelligent evade from attacker
    rr, cc = (2 * sr - gr, 2 * sc - gc)

    pathfind(config, ob, actions, rr, cc)


def forageDijkstra(config, ob, actions, food_max, water_max, cutoff=100):
    vision = config.NSTIM
    Entity = nmmo.Serialized.Entity
    Tile = nmmo.Serialized.Tile

    food = nmmo.scripting.Observation.attribute(ob.agent, Entity.Food)
    water = nmmo.scripting.Observation.attribute(ob.agent, Entity.Water)

    best = -1000
    start = (0, 0)
    goal = (0, 0)

    reward = {start: (food, water)}
    backtrace = {start: None}

    queue = Queue()
    queue.put(start)

    while not queue.empty():
        cutoff -= 1
        if cutoff <= 0:
            break

        cur = queue.get() # remove and return an item from the queue
        for nxt in adjacentPos(cur):
            if nxt in backtrace:
                continue

            if not inSight(*nxt, vision):
                continue

            tile = ob.tile(*nxt)
            matl = nmmo.scripting.Observation.attribute(tile, Tile.Index)
            occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)

            if not vacant(tile):
                continue

            food, water = reward[cur]
            food = max(0, food - 1)
            water = max(0, water - 1)

            if matl == material.Forest.index:
                food = min(food + food_max // 2, food_max)
            for pos in adjacentPos(nxt):
                # as long as nxt tile has ajacent water
                if not inSight(*pos, vision):
                    continue

                tile = ob.tile(*pos)
                matl = nmmo.scripting.Observation.attribute(tile, Tile.Index)

                if matl == material.Water.index:
                    water = min(water + water_max // 2, water_max)
                    break

            reward[nxt] = (food, water)

            total = min(food, water)
            if total > best or (total == best
                                and max(food, water) > max(reward[goal])):
                best = total
                goal = nxt

            queue.put(nxt)
            backtrace[nxt] = cur

    while goal in backtrace and backtrace[goal] != start:
        # backtrace from goal to start's next
        goal = backtrace[goal]

    direction = towards(goal)
    if direction is not None:
        actions[nmmo.action.Move] = {nmmo.action.Direction: direction}


def aStar(config, ob, actions, rr, cc, cutoff=100):
    Entity = nmmo.Serialized.Entity
    Tile = nmmo.Serialized.Tile
    vision = config.NSTIM

    start = (0, 0)
    goal = (rr, cc)

    if start == goal:
        return (0, 0)

    pq = PriorityQueue()
    pq.put((0, start))

    backtrace = {}
    cost = {start: 0}

    closestPos = start
    closestHeuristic = utils.l1(start, goal)
    closestCost = closestHeuristic

    while not pq.empty():
        # Use approximate solution if budget exhausted
        cutoff -= 1
        if cutoff <= 0:
            if goal not in backtrace:
                goal = closestPos
            break

        priority, cur = pq.get()

        if cur == goal:
            break

        for nxt in adjacentPos(cur):
            if not inSight(*nxt, vision):
                continue

            tile = ob.tile(*nxt)
            matl = nmmo.scripting.Observation.attribute(tile, Tile.Index)
            occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)

            #if not vacant(tile):
            #   continue

            if occupied:
                continue

            #Omitted water from the original implementation. Seems key
            if matl in material.Impassible:
                continue

            newCost = cost[cur] + 1
            if nxt not in cost or newCost < cost[nxt]:
                cost[nxt] = newCost
                heuristic = utils.lInfty(goal, nxt)
                priority = newCost + heuristic

                # Compute approximate solution
                if heuristic < closestHeuristic or (heuristic == closestHeuristic
                                                    and priority < closestCost):
                    closestPos = nxt
                    closestHeuristic = heuristic
                    closestCost = priority

                pq.put((priority, nxt))
                backtrace[nxt] = cur

    #Not needed with scuffed material list above
    #if goal not in backtrace:
    #   goal = closestPos

    while goal in backtrace and backtrace[goal] != start:
        goal = backtrace[goal]

    return goal
