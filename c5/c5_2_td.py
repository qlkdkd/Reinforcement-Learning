from GridWorld import GridWorld
from Agent import Agent

#def main():
env = GridWorld()
agent = Agent()
data = [[0]*4 for i in range(4)]
gamma = 1
alpha = 0.01

for k in range(50000):
    done = False
    while not done:
        x, y = env.get_state()
        action = agent.select_action()
        (x_prime, y_prime), reward, done = env.step(action)
        x_prime, y_prime = env.get_state()

        # 한 번의 스텝이 진행되자마자 바로 테이블의 데이터를 업데이트
        data[x][y] = data[x][y] + alpha * \
            (reward + gamma * data[x_prime][y_prime] - data[x][y])
    env.reset()

for row in data:
    for i in row:
        print(f'{i:.3f}', end=' ') # 출력창 더러워서 반올림
    print()
