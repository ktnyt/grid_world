from grid_world.agents.policy_gradient import PolicyGradient
from grid_world.envs.grid_world import GridWorld


def main():
    agent = PolicyGradient()
    env = GridWorld()
    current_state = env.reset()
    next_action = agent.random_action()

    for _ in range(1000):
        env.render()
        current_state, reward, done, info = env.step(next_action)
        next_action = agent.step(current_state, reward)
        if done:
            current_state = env.reset()
    env.close()


if __name__ == '__main__':
    main()
