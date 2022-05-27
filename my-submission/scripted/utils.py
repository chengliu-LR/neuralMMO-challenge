import nmmo
from nmmo.lib import material

def l1(start, goal):
    sr, sc = start
    gr, gc = goal
    return abs(gr - sr) + abs(gc - sc)


def l2(start, goal):
    sr, sc = start
    gr, gc = goal
    return 0.5 * ((gr - sr)**2 + (gc - sc)**2)**0.5


def lInfty(start, goal):
    sr, sc = start
    gr, gc = goal
    return max(abs(gr - sr), abs(gc - sc))


def adjacentPos(pos):
    r, c = pos
    return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]


def adjacentDeltas():
    return [(-1, 0), (1, 0), (0, 1), (0, -1)]


def inSight(dr, dc, vision):
    return (dr >= -vision and dc >= -vision and dr <= vision and dc <= vision)


def vacant(tile):
    Tile = nmmo.Serialized.Tile
    occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)
    matl = nmmo.scripting.Observation.attribute(tile, Tile.Index)

    lava = material.Lava.index
    water = material.Water.index
    grass = material.Grass.index
    scrub = material.Scrub.index
    forest = material.Forest.index
    stone = material.Stone.index
    orerock = material.Orerock.index

    return matl in (grass, scrub, forest) and not occupied


def inSquadOne(entID, pop):
    '''In squad 1'''
    return entID % 8 <= 4 and entID // 8 == pop


def inSquadTwo(entID, pop):
    '''In squad 2'''
    return entID % 8 > 4 or entID // 8 == pop + 1


def spawnLeftBottom(config, spawnR, spawnC):
    if spawnC <= config.TERRAIN_BORDER or (
        spawnR >= config.TERRAIN_BORDER + config.TERRAIN_CENTER):
        return True
    else:
        return False


def nextTarget(cur, targetsList, inSquadOne, spawnLeftBottom):
    cur_index = targetsList.index(cur)
    if inSquadOne:
        if spawnLeftBottom:
            next_index = cur_index - 1
        else:
            next_index = cur_index + 1
    else:
        if spawnLeftBottom:
            next_index = cur_index + 1
        else:
            next_index = cur_index - 1
    next_index = next_index % len(targetsList)
    
    return targetsList[next_index]