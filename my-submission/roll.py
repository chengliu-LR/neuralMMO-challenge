from ijcai2022nmmo import CompetitionConfig, scripted, submission, RollOut

config = CompetitionConfig()
config.SAVE_REPLAY = "../replays/rickyProtoss"

my_team = submission.get_team_from_submission(
    submission_path="./",
    team_id="rickyProtoss",
    env_config=config,
)
# Or initialize the team directly
# my_team = MyTeam("Myteam", config, ...)

teams = []
teams.append(my_team)
teams.extend([scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
teams.extend([scripted.ForageTeam(f"Forage-{i}", config) for i in range(5)])
teams.extend([scripted.RandomTeam(f"Random-{i}", config) for i in range(7)])

ro = RollOut(config, teams, parallel=True, show_progress=True)
ro.run(n_timestep=1024, n_episode=1, render=False)
