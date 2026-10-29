from Agent import Agent
from GridWorld import GridWorld

def main():
    env = GridWorld()
    agent = Agent()
    data = [[0]*4 for i in range(4)]
    gamma = 1
    alpha = 0.0001

    for k in range(50000):
        done = False
        history = []
        while not done:
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x, y, reward))
        env.reset()

        # 매 에피소드가 끝나고 바로 해당 데이터를 이용해 테이블 업데이터
        cum_reward = 0
        for transition in history[::-1]:
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][x])
            cum_reward = cum_reward + gamma * reward

    for row in data:
        for i in row:
            print(f'{i:.3f}', end=' ') # 출력창 더러워서 반올림
        print()


main()