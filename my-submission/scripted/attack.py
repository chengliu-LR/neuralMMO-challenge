import numpy as np
import nmmo
#from ijcai2022nmmo.scripted import utils
import scripted.utils as utils

def closestTarget(config, ob):
    shortestDist = np.inf
    closestAgent = None

    Entity = nmmo.Serialized.Entity
    agent = ob.agent

    sr = nmmo.scripting.Observation.attribute(agent, Entity.R)
    sc = nmmo.scripting.Observation.attribute(agent, Entity.C)
    start = (sr, sc)

    for target in ob.agents:
        exists = nmmo.scripting.Observation.attribute(target, Entity.Self) # Entity.Self indicates whether its valuable information
        if not exists:
            continue

        tr = nmmo.scripting.Observation.attribute(target, Entity.R)
        tc = nmmo.scripting.Observation.attribute(target, Entity.C)

        # check if its NPC
        entityID = nmmo.scripting.Observation.attribute(target, Entity.ID)

        goal = (tr, tc)
        dist = utils.l1(start, goal)

        if dist < shortestDist and dist != 0:
            shortestDist = dist
            closestAgent = target

    if closestAgent is None:
        return None, None

    return closestAgent, shortestDist


def teamateAttacker(config, ob):
    Entity = nmmo.Serialized.Entity
    agent = ob.agent

    sr = nmmo.scripting.Observation.attribute(agent, Entity.R)
    sc = nmmo.scripting.Observation.attribute(agent, Entity.C)
    start = (sr, sc)

    for target in ob.agents:
        exists = nmmo.scripting.Observation.attribute(target, Entity.Self)
        if not exists:
            continue

        entityPop = nmmo.scripting.Observation.attribute(target, Entity.Population)
        selfPop = nmmo.scripting.Observation.attribute(agent, Entity.Population)

        if selfPop == entityPop:
            teamateAttackerID = nmmo.scripting.Observation.attribute(target,
                                                        Entity.AttackerID)
            if teamateAttackerID == 0:
                return None, None

            for targ in ob.agents:
                identity = nmmo.scripting.Observation.attribute(targ, Entity.ID)
                if identity == teamateAttackerID:
                    tr = nmmo.scripting.Observation.attribute(targ, Entity.R)
                    tc = nmmo.scripting.Observation.attribute(targ, Entity.C)
                    goal = (tr, tc)
                    targDist = utils.l1(start, goal)

                    return targ, targDist

        return None, None


def attacker(config, ob):
    Entity = nmmo.Serialized.Entity

    sr = nmmo.scripting.Observation.attribute(ob.agent, Entity.R)
    sc = nmmo.scripting.Observation.attribute(ob.agent, Entity.C)

    attackerID = nmmo.scripting.Observation.attribute(ob.agent,
                                                      Entity.AttackerID)

    if attackerID == 0:
        return None, None

    for target in ob.agents:
        identity = nmmo.scripting.Observation.attribute(target, Entity.ID)
        if identity == attackerID:
            tr = nmmo.scripting.Observation.attribute(target, Entity.R)
            tc = nmmo.scripting.Observation.attribute(target, Entity.C)
            dist = utils.l1((sr, sc), (tr, tc))
            return target, dist
    return None, None


def target(config, actions, style, targetID):
    actions[nmmo.action.Attack] = {
        nmmo.action.Style: style,
        nmmo.action.Target: targetID
    }
