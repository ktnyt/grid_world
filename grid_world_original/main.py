import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
from policy_gradient import network
from environments import MazeGrid
from renderers import GridRenderer

if __name__ == "__main__":

  grid_size = 32
  ylen = MazeGrid.ylen
  xlen = MazeGrid.xlen
  renderer = GridRenderer(width=xlen*grid_size, height=ylen*grid_size,
                          grid_size=grid_size)
  env = MazeGrid(renderer, map_type=1)
  env.reset()
  observation = env.reset()
  reward_sum = 0
  episode_number = 0
  running_reward = None

  pg=network(ylen*xlen,200,4)

  while True:
    env.render()
    #x = pg.preprop(observation)
    """ ID to input """
    x = np.zeros(shape=(xlen* ylen), dtype=np.int32)
    x[observation[1]*xlen + observation[0]] = 1
    x[env.reward_pos[1]*xlen + env.reward_pos[0]] = 1
    #pg.record_x (x)
    # print (env.reward_pos)
    #print (x)

    #今のネズミの位置
    prev_pos = observation


    #チーズの位置
    prev_reward_pos = env.reward_pos

    #aprob, h = pg.policy_forward(x)
    aprob=pg.forward(x)
    action = pg.select_action(aprob)
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    #行動した後のネズミの位置
    pos = observation


    #チーズと行動前の位置との距離
    p_dist = abs(prev_reward_pos[0] - prev_pos[0]) + \
             abs(prev_reward_pos[1] - prev_pos[1])

    #チーズと行動した後の位置との距離
    c_dist = abs(prev_reward_pos[0] - pos[0]) + \
             abs(prev_reward_pos[1] - pos[1])
    # print ("prev_reward_pos, prev_pos, pos", prev_reward_pos, prev_pos, pos)
    # print ("p_dist , c_dist", p_dist, c_dist)

    #行動前より行動後の方がチーズに近ければ報酬を与える
    if p_dist > c_dist:
      reward += 1.
    else:
      reward -= 1.


    pg.record_reward(reward)


    if done: # an episode finished
      episode_number += 1

      if reward_sum != 0:
        #print ("reward_sum : ", reward_sum)

        #勾配を求める
        #pg.calc_grad()
        pg.backward()

      # perform rmsprop parameter update every batch_size episodes
      if episode_number % pg.batch_size == 0:
        #パラメータを更新
        #pg.update_params()
        pg.update()

      running_reward = reward_sum if running_reward is None \
                       else running_reward * 0.99 + reward_sum * 0.01
      print ('ep %d:  episode reward total was %f. running mean: %f'
             % (episode_number, reward_sum, running_reward))

      if episode_number % 100 == 0:
        pg.save_model(filename="save.p")

      f = open("res.txt", "a")
      f.write(str(reward_sum) + "\n")
      f.close()

      reward_sum = 0
      observation = env.reset() # reset env
