from pathlib import Path

from spine_rl import SpineFixConfig, SpineFixEnv


def main() -> None:
    config = SpineFixConfig(model_path=Path("assets/spine_model.xml"), num_fractures=3)
    env = SpineFixEnv(config=config, render_mode="human")
    obs, info = env.reset()
    print("Fractured bodies:", info["fractured_bodies"])

    for _ in range(config.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
