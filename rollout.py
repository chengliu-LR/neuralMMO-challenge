import nmmo
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv, scripted


class Config(CompetitionConfig):
    SAVE_REPLAY = "demo"


def save_replay():
    """Demo for saving replay"""
    config = Config()
    env = TeamBasedEnv(config=config)
    scripted_ai = scripted.CombatTeam(None, config)
    obs = env.reset()
    t, horizon = 0, 32
    while True:
        env.render()
        decision = {}
        for team_id, o in obs.items():
            decision[team_id] = scripted_ai.act(o)
        env.step(decision)
        t += 1
        if t >= horizon:
            break
    env.terminal()


def load_replay():
    """Demo for loading replay"""
    replay = nmmo.Replay.load(Config.SAVE_REPLAY + ".replay")
    replay.render()


if __name__ == "__main__":
    save_replay()
    # load_replay()