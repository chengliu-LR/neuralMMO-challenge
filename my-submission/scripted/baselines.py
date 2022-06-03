import nmmo
from nmmo import scripting
from nmmo.lib import colors
import scripted.attack as attack
import scripted.move as move

class Scripted(nmmo.Agent):
    '''Template class for scripted models.

    You may either subclass directly or mirror the __call__ function'''
    scripted = True
    color = colors.Neon.SKY

    def __init__(self, config, idx):
        '''
        Args:
           config : A forge.blade.core.Config object or subclass object
        '''
        super().__init__(config, idx)
        self.food_max = 0
        self.water_max = 0

        self.spawnR = None
        self.spawnC = None

        # for protoss combat state machine
        self.frozen_count = 0

        # for square exploration
        self.current_target_point = None

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water'''
        # this parameter can be tuned
        food_min_level = 0.8 * self.food_max
        water_min_level = 0.8 * self.water_max
        return self.food <= food_min_level or self.water <= water_min_level

    def forage(self):
        '''Min/max food and water using Dijkstra's algorithm'''
        move.forageDijkstra(self.config, self.ob, self.actions, self.food_max, self.water_max)

    def explore(self):
        '''Route away from spawn'''
        self.current_target_point = move.explore(self.config, self.ob, self.actions, self.spawnR,
                                            self.spawnC, self.current_target_point)


    def explore_square(self):
        '''Rout away in square from spawn'''
        self.current_target_point = move.explore_square(self.config, self.ob, self.actions, self.spawnR,
                                                  self.spawnC, self.current_target_point)

    def explore_hybrid(self):
        self.current_target_point = move.explore_hybrid(self.config, self.ob, self.actions, self.spawnR,
                                                self.spawnC, self.current_target_point)

    @property
    def downtime(self):
        '''Return true if agent is not occupied with a high-priority action'''
        return not self.forage_criterion and self.attacker is None


    def evade(self):
        '''Target and path away from an attacker'''
        self.target = self.attacker
        self.targetID = self.attackerID
        self.targetDist = self.attackerDist

        # freeze and then evade
        attack.target(self.config, self.actions, nmmo.action.Mage, self.targetID)
        move.evade(self.config, self.ob, self.actions, self.attacker)


    def attack(self):
        '''Attack the current target'''
        if self.target is not None:
            assert self.targetID is not None
            attack.target(self.config, self.actions, self.style, self.targetID)
            #targetHealth = scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Health)
            #print("target health ID {} and health {}\n".format(self.targetID, targetHealth))

    
    def hit_and_run(self):
        '''Hit and run to dynamicly attack & evade from the target agent'''
        pass


    def select_combat_style(self):
        '''Select a combat style based on distance from the current target'''
        if self.target is None:
            return

        if self.targetDist <= self.config.COMBAT_MELEE_REACH:
            self.style = nmmo.action.Melee
        elif self.targetDist <= self.config.COMBAT_RANGE_REACH:
            self.style = nmmo.action.Range
        else:
            self.style = nmmo.action.Mage


    def protoss_combat(self):
        '''An more reasonable combat style selection'''
        if self.target is None:
            return

        frozen = scripting.Observation.attribute(
            self.target, nmmo.Serialized.Entity.Freeze)
        if not (frozen and self.frozen_count >= 1):
            self.style = nmmo.action.Mage
            self.frozen_count = 3
            return
        elif self.targetDist <= self.config.COMBAT_MELEE_REACH:
            self.style = nmmo.action.Melee
        elif self.targetDist <= self.config.COMBAT_RANGE_REACH:
            self.style = nmmo.action.Range
        else:
            self.style = nmmo.action.Mage

        self.frozen_count -= 1
        self.frozen_count = max(0, self.frozen_count)


    def protoss_hybrid_combat(self):
        '''Hit with range and mage'''
        if self.target is None:
            return


    def scan_agents(self):
        '''Scan the nearby area for agents'''
        self.closest, self.closestDist = attack.closestTarget(
            self.config, self.ob)
        self.attacker, self.attackerDist = attack.attacker(
            self.config, self.ob)

        self.closestID = None
        if self.closest is not None:
            self.closestID = scripting.Observation.attribute(
                self.closest, nmmo.Serialized.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
            self.attackerID = scripting.Observation.attribute(
                self.attacker, nmmo.Serialized.Entity.ID)

        self.style = None
        self.target = None
        self.targetID = None
        self.targetDist = None


    def target_weak(self):
        '''Target the nearest agent if it is weak'''
        # TODO: this does not make sense because you need to target
        # at the most valuable agent rather than the nearest
        # You can change it to a queue of potential targets
        if self.closest is None:
            return False

        selfLevel = scripting.Observation.attribute(
            self.ob.agent, nmmo.Serialized.Entity.Level)
        targLevel = scripting.Observation.attribute(
            self.closest, nmmo.Serialized.Entity.Level)
        targPopulation = scripting.Observation.attribute(
            self.closest, nmmo.Serialized.Entity.Population)

        # this can be an aggresive attack strategy
        if selfLevel >= targLevel or (
            targPopulation == -1 and selfLevel >= targLevel - 10) or (   # passive npc
            targPopulation == -2 and selfLevel >= targLevel - 5) or (   # neutral npc
            targPopulation == -3 and selfLevel >= targLevel - 2):       # hostile npc

            self.target = self.closest
            self.targetID = self.closestID
            self.targetDist = self.closestDist


    def adaptive_control_and_targeting(self, explore=True):
        '''Balanced foraging, evasion, and exploration'''
        self.scan_agents()

        if self.attacker is not None:
            selfLevel = scripting.Observation.attribute(
                self.ob.agent, nmmo.Serialized.Entity.Level)
            attackerLevel = scripting.Observation.attribute(
                self.attacker, nmmo.Serialized.Entity.Level)

            if attackerLevel <= selfLevel <= 3 or (selfLevel >= attackerLevel - 3 and selfLevel >= 3):
                # if the level is higher than attacker
                self.target = self.attacker
                self.targetID = self.attackerID
                self.targetDist = self.attackerDist
                return

            else:
                self.evade()
                return

        if self.forage_criterion or not explore:
            self.forage()
        else:
            #self.explore()
            self.explore_hybrid()

        self.target_weak()


    def __call__(self, obs):
        '''Process observations and return actions

        Args:
           obs: An observation object from the environment. Unpack with scripting.Observation
        '''
        self.actions = {}

        self.ob = scripting.Observation(self.config, obs)
        agent = self.ob.agent

        self.food = scripting.Observation.attribute(
            agent, nmmo.Serialized.Entity.Food)
        self.water = scripting.Observation.attribute(
            agent, nmmo.Serialized.Entity.Water)

        if self.food > self.food_max:
            self.food_max = self.food
        if self.water > self.water_max:
            self.water_max = self.water

        if self.spawnR is None:
            self.spawnR = scripting.Observation.attribute(
                agent, nmmo.Serialized.Entity.R)
        if self.spawnC is None:
            self.spawnC = scripting.Observation.attribute(
                agent, nmmo.Serialized.Entity.C)


    @property
    def targets(self):
        return [
            x for x in [
                scripting.Observation.attribute(target,
                                                nmmo.Serialized.Entity.ID)
                for target in self.ob.agents # ob.agents = scripting.Observation['Entity']['Continuous']
            ] if x
        ] # return target ID of observable agents


class Protoss(Scripted):
    '''Forages, fights, and explores'''
    name = 'Protoss_'

    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()

        #self.style = nmmo.action.Range
        #self.select_combat_style()
        self.protoss_combat()
        self.attack()

        return self.actions
