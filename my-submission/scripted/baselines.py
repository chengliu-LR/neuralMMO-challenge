import nmmo
from nmmo import scripting
from nmmo.lib import colors
import scripted.attack as attack
import scripted.move as move
from collections import deque

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

        # to get out of local optimum
        self.local_trap_r = deque(maxlen=25)
        self.local_trap_c = deque(maxlen=25)
        self.stuck_steps = [0]
        self.pathFound = [True]
        self.rr = [None]
        self.cc = [None]
        

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
                                            self.spawnC, self.current_target_point, self.pathFound, self.rr, self.cc)


    def explore_square(self):
        '''Rout away in square from spawn'''
        self.current_target_point = move.explore_square(self.config, self.ob, self.actions, self.spawnR,
                                                  self.spawnC, self.current_target_point, self.local_trap_r, self.local_trap_c, self.stuck_steps)

    def explore_hybrid(self):
        self.current_target_point = move.explore_hybrid_squad(self.config, self.ob, self.actions, self.spawnR,
                                                self.spawnC, self.current_target_point, self.local_trap_r, self.local_trap_c, self.stuck_steps)

    @property
    def downtime(self):
        '''Return true if agent is not occupied with a high-priority action'''
        return not self.forage_criterion and self.attacker is None


    def evade(self):
        '''Freeze the target and path away from an attacker'''
        attack.target(self.config, self.actions, nmmo.action.Mage, self.targetID)
        move.evade(self.config, self.ob, self.actions, self.target)


    def hunt(self):
        '''Dynamicly attack & hunt the target agent'''
        move.hunt(self.config, self.ob, self.actions, self.target)


    def attack(self):
        '''Attack the current target'''
        if self.target is not None:
            assert self.targetID is not None
            attack.target(self.config, self.actions, self.style, self.targetID)
            #targetHealth = scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Health)
            #print("target health ID {} and health {}\n".format(self.targetID, targetHealth))


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

        frozen = scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Freeze)
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


    def scan_agents(self):
        '''Scan the nearby area for agents'''
        self.closest, self.closestDist = attack.closestTarget(self.config, self.ob)
        self.attacker, self.attackerDist = attack.attacker(self.config, self.ob)

        self.closestID = None
        if self.closest is not None:
            self.closestID = scripting.Observation.attribute(self.closest, nmmo.Serialized.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
            self.attackerID = scripting.Observation.attribute(self.attacker, nmmo.Serialized.Entity.ID)

        self.style = None
        self.target = None
        self.targetID = None
        self.targetDist = None


    def aim_at_target(self):
        '''Target the nearest agent if it is weak or target attacker'''
        # You can change it to a queue of potential targets
        # aggresive attack strategy
        #if selfLevel >= targLevel or self.is_npc(targPopulation):
        self.target = self.closest
        self.targetID = self.closestID
        self.targetDist = self.closestDist
        
        if self.attacker is not None:
            self.target = self.attacker
            self.targetID = self.attackerID
            self.targetDist = self.attackerDist

        self.selfLevel = scripting.Observation.attribute(self.ob.agent, nmmo.Serialized.Entity.Level)
        self.targetLevel = scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Level)
        self.targetPopulation = scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Population)


    def is_npc(self, targPop):
        return targPop == -1 or targPop == -2 or targPop == -3


    def adaptive_control_and_targeting(self):
        '''Balanced foraging, evasion, and exploration'''
        self.scan_agents()
        if self.forage_criterion and self.attacker is None:
            self.forage()
        elif self.closest is not None:
            self.aim_at_target()
            if self.target is not None:
                if self.is_npc(self.targetPopulation) or self.selfLevel >= self.targetLevel:
                    if self.targetDist > self.config.COMBAT_MAGE_REACH:
                        self.hunt()
                else:
                    self.evade()
        else:
            self.explore()
            if not self.pathFound[0]:
                self.forage()


    def __call__(self, obs):
        '''Process observations and return actions

        Args:
           obs: An observation object from the environment. Unpack with scripting.Observation
        '''
        self.actions = {}

        self.ob = scripting.Observation(self.config, obs)
        agent = self.ob.agent

        self.food = scripting.Observation.attribute(agent, nmmo.Serialized.Entity.Food)
        self.water = scripting.Observation.attribute(agent, nmmo.Serialized.Entity.Water)

        if self.food > self.food_max:
            self.food_max = self.food
        if self.water > self.water_max:
            self.water_max = self.water

        if self.spawnR is None:
            self.spawnR = scripting.Observation.attribute(agent, nmmo.Serialized.Entity.R)
        if self.spawnC is None:
            self.spawnC = scripting.Observation.attribute(agent, nmmo.Serialized.Entity.C)


    @property
    def targets(self):
        return [x for x in [scripting.Observation.attribute(target, nmmo.Serialized.Entity.ID)
                for target in self.ob.agents] if x] # return target ID of observable agents


class Protoss(Scripted):
    '''Forages, fights, and explores'''
    name = 'Protoss_'

    def __call__(self, obs):
        super().__call__(obs)
        self.adaptive_control_and_targeting()
        #self.style = nmmo.action.Mage
        self.select_combat_style()
        #self.protoss_combat()
        self.attack()

        return self.actions
