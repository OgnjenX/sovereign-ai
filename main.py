from sovereign_ai.architecture import CognitiveArchitecture
from sovereign_ai.environment import GridWorld


def main() -> None:
    env = GridWorld(size=5, seed=7)
    agent = CognitiveArchitecture(
        input_dim=env.observation_dim,
        action_count=env.action_count,
        max_categories=12,
        seed=3,
        debug=True,
    )

    agent.run(env, steps=24)


if __name__ == "__main__":
    main()
