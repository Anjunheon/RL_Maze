import random
import numpy as np
import tkinter
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from collections import deque
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.initializers import RandomUniform

import pprint

import game


class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        # self.conv = Conv2D(32, (1, 1), strides=(1, 1), activation='relu', input_shape=state_size)
        # self.flatten = Flatten()
        # self.fc = Dense(64, activation='relu')
        # self.dropout = Dropout(0.5)
        # self.fc_out = Dense(action_size)

        self.fc1 = Dense(32, activation='tanh', input_shape=state_size)
        self.flatten = Flatten()
        self.fc2 = Dense(32, activation='tanh')
        self.fc3 = Dense(64, activation='tanh')
        self.fc_out = Dense(action_size, activation='softmax')


    def call(self, x):
        # x = self.conv(x)
        # x = self.flatten(x)
        # x = self.dropout(x)
        # x = self.fc(x)
        # q = self.fc_out(x)

        x = self.fc1(x)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, action_size=3, state_size=(None, 2 * 2 + 1, 2 * 2 + 1, 1)):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        # self.exploration_steps = 1000000.
        # self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        # self.epsilon_decay_step /= self.exploration_steps
        self.epsilon_decay_step = 0.99
        self.max_step = 200
        self.batch_size = 32
        self.train_start = 5000
        self.train_freq = 5
        self.update_target_rate = 20

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=20000)
        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 50

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(self.learning_rate, clipnorm=1.)

        # self.optimizer = RMSprop(self.learning_rate, clipnorm=10.1)

        # 타깃 모델 초기화
        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/maze_dqn')
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history)
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_value = self.model.call(history)
            return np.argmax(q_value)

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                              self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                              self.avg_loss / float(step), step=episode)

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            # self.epsilon -= self.epsilon_decay_step
            self.epsilon *= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch = random.sample(self.memory, self.batch_size)

        # history = np.array([sample[0][0] for sample in batch],
        #                    dtype=np.float32)
        state = np.array([sample[0][0] for sample in batch],
                         dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] for sample in batch],
                                dtype=np.float32)
        next_state = np.array([sample[3] for sample in batch],
                              dtype=np.float32)
        dones = np.array([[sample[4]] for sample in batch])

        # 학습 파라미터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model.call(np.float32(state))
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=-1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model.call(np.float32(next_state))

            # 벨만 최적 방정식을 구성하기 위한 타깃과 큐함수의 최대 값 계산
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + np.transpose(1 - dones) * self.discount_factor * max_q

            # 1) 벨만 최적 방정식을 이용한 업데이트 타깃
            loss = tf.reduce_mean(tf.square(targets[0] - predicts))

            # 2) 후버로스 계산
            # error = tf.abs(targets[0] - predicts)
            # quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            # linear_part = error - quadratic_part
            # loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            self.avg_loss += loss.numpy()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


agent = DQNAgent(action_size=3, state_size=(None, game.rsize * 2 + 1, game.csize * 2 + 1, 1))

agent.model.build(input_shape=(None, game.rsize * 2 + 1, game.csize * 2 + 1, 1))
agent.target_model.build(input_shape=(None, game.rsize * 2 + 1, game.csize * 2 + 1, 1))

agent.model.summary()
agent.target_model.summary()

global_step = 0
score_avg = 0
score_max = 0

degree = 0  # 회전 각도
rotate = {0: 0, 1: 90, 2: -90}

acc_deg = 0  # 누적 회전 각도

move_delay = 0.00

num_episode = 50000
for e in range(0, num_episode):
    reward = 0
    done = False

    step, score = 0, 0

    # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
    # state = np.float32(mazeMap)
    # state = np.float32(mazeMap) / 10.
    state = np.float32(game.get_maze())
    state = np.reshape([state], (1, game.rsize * 2 + 1, game.csize * 2 + 1, 1))

    while not done:
        time.sleep(random.randint() / 10.)
        global_step += 1
        step += 1

        # 바로 전 state를 입력으로 받아 행동을 선택
        # 0: 0도, 1: 90도, 2: -90도
        game.key = agent.get_action(np.float32(state))
        action = game.key
        game.input_flag = True

        # 도착지점 도착 시
        if game.mazeMap[game.posY][game.posX] == 2:
            done = True
            reward = 1

        if not done:
            # 이전에 방문했던 블럭 재방문 시
            # if mazeMap[posY][posX] == 4 or mazeMap[posY][posX] == 3:
            #     # 중간보상 방식
            #     reward = -1
            #
            #     # 에피소드 종료 방식
            #     # reward = -1
            #     # done = True
            # else:
            #     reward = 0

            # 스텝마다 - 보상
            reward = -0.01

        # pprint.pprint(mazeMap)

        # 중간 보상 : 맨해튼 거리
        # if not done:
        #     if USE_MAX_STEP:
        #         if step == agent.max_step:
        #             reward = -1
        #             done = True
            # else:
            # reward = -1 / 400 + ((np.abs(destX-posX) + np.abs(destY-posY))) / 10000
            # reward = -round((np.abs(destX-posX) + np.abs(destY-posY)) / 10, 3)
            # reward = -1 / ((np.abs(destX-posX) + np.abs(destY-posY) + 0.001))
            # reward = -round((np.abs(destX-posX) + np.abs(destY-posY)) / 8, 2) + 0.5
            # reward += -round(1 / 900, 3)
            # reward = -((np.abs(destX-posX) + np.abs(destY-posY))) / 10
        # print(reward)

        # 각 타임스텝마다 상태 전처리
        next_state = np.float32(game.get_maze())
        # next_state = np.float32(mazeMap) / 10.
        next_state = np.reshape([next_state], (1, game.rsize * 2 + 1, game.csize * 2 + 1, 1))

        # 가장 큰 Q값 가산
        agent.avg_q_max += np.amax(agent.model.call(np.float32([state])))

        score += reward
        reward = np.clip(reward, -1., 1.)
        # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
        agent.append_sample(state, action, reward, next_state, done)

        # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
        if len(agent.memory) >= agent.train_start:
            if step % agent.train_freq == 0:
                agent.train_model()
                # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

        if done:
            # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
            # agent.update_target_model()

            # 각 에피소드 당 학습 정보를 기록
            if global_step > agent.train_start:
                agent.draw_tensorboard(score, step, e)

            score_avg = score / float(step)
            score_max = score if score > score_max else score_max

            log = "episode: {:5d} | ".format(e)
            log += "step: {:5d} | ".format(step)
            log += "global step: {:5d} | ".format(global_step)
            log += "score: {:4.1f} | ".format(score)
            # log += "score max : {:4.1f} | ".format(score_max)
            log += "score avg: {:4.1f} | ".format(score_avg)
            log += "memory length: {:5d} | ".format(len(agent.memory))
            log += "epsilon: {:.3f} | ".format(agent.epsilon)
            log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
            log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
            print(log)

            agent.avg_q_max, agent.avg_loss = 0, 0

    # 100 에피소드마다 모델 저장
    if e % 100 == 0:
        agent.model.save_weights("./save_model/model", save_format="tf")

