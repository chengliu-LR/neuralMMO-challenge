import nmmo
from typing import Dict, Type, List

from ijcai2022nmmo.evaluation.team import Team
import scripted.baselines as baselines

class ScriptedTeam(Team):
    agent_klass: Type = None
    agents: List[baselines.Scripted]

    def __init__(
        self,
        team_id: str,
        env_config: nmmo.config.Config,
        **kwargs,
    ) -> None:
        if "policy_id" not in kwargs:
            kwargs["policy_id"] = self.agent_klass.__name__
        super().__init__(team_id, env_config, **kwargs)
        self.reset()

    def reset(self):
        assert self.agent_klass
        self.agents = [
            self.agent_klass(self.env_config, i) # initialize agents here
            for i in range(self.env_config.TEAM_SIZE)
        ]

    def act(self, observations: Dict[int, dict]) -> Dict[int, dict]:
        actions = {i: self.agents[i](obs) for i, obs in observations.items()}
        for i in actions:
            for atn, args in actions[i].items():
                for arg, val in args.items():
                    if len(arg.edges) > 0:
                        actions[i][atn][arg] = arg.edges.index(val)
                    else:
                        targets = self.agents[i].targets
                        actions[i][atn][arg] = targets.index(val)
        return actions


class rickyProtoss(ScriptedTeam):
    agent_klass = baselines.Protoss


class Submission:
    team_klass = rickyProtoss
    init_params = {}
