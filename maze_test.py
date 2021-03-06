import random
import numpy as np
import tkinter
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.initializers import RandomUniform

import pprint

tm = time.localtime()
start_time = str(tm.tm_year) + str(tm.tm_mon) + str(tm.tm_mday) + str(tm.tm_hour) + str(tm.tm_min)

PLAY_MODE = 0
GAME_SPEED = 1  # 1~10
ROTATION_MODE = False  # 미로 회전 그래픽 출력
ROTATE_DELAY = 0.0 / GAME_SPEED
MOVE_DELAY = 0.0 / GAME_SPEED
USE_MAX_STEP = False

CUSTOM = False
RANDOM_MAZE = True
MAZE_NUM = 5  # 미로 종류

# 미로 크기 설정(홀수)
if not CUSTOM:
    rsize = 10
    csize = 10
else:
    if MAZE_NUM <= 4:
        rsize = 7
        csize = 7
    elif MAZE_NUM == 5:
        rsize = 11
        csize = 11
    elif MAZE_NUM == 6:
        rsize = 15
        csize = 15

rsize = int(rsize/2)
csize = int(csize/2)

state_size = (None, rsize*2+1, csize*2+1)


# 0: 방문하지 않은 블럭, 1: 벽, 2: 도착지, 3: 피스, 4: 방문했던 블럭
class Room:
    def __init__(self, r, c):
        self.r, self.c = r, c
        self.visit = 0
        self.prev = None
        self.drct = [(r + 1, c), (r, c + 1),
                     (r - 1, c), (r, c - 1)]
        random.shuffle(self.drct)


def make(prev, room, _maze, _rsize, _csize):
    room.prev = prev
    if room.prev is None:
        mazeMap[0][1] = 0
    else:
        r = prev.r - room.r
        c = prev.c - room.c
        mazeMap[(room.r + 1) * 2 - 1 + r][(room.c + 1) * 2 - 1 + c] = 0

    room.visit = 1
    mazeMap[(room.r + 1) * 2 - 1][(room.c + 1) * 2 - 1] = 0
    # print('\n현재좌표: {0},{1}'.format(room.r, room.c))

    while True:
        if len(room.drct) == 0:
            break
        nr, nc = room.drct.pop()
        # print('좌표체크: {0},{1}'.format(nr, nc))

        if nr >= 0 and nr < _rsize and nc >= 0 and nc < _csize:
            if not _maze[nr][nc].visit == 1:
                # print('{0},{1} 좌표로 진행'.format(nr, nc))
                make(room, _maze[nr][nc], _maze, _rsize, _csize)
            else:
                print('방문기록있음\b\b\b\b\b\b\b', end='')
        else:
            print('진행불가\b\b\b\b\b\b\b\b\b\b', end='')


def make_maze():
    global maze, mazeMap
    global rsize, csize
    global destY, destX
    global MAZE_NUM

    maze = [[Room(r, c) for c in range(csize)] for r in range(rsize)]
    mazeMap = [[1 for c in range(csize * 2 + 1)] for r in range(rsize * 2 + 1)]

    make(None, maze[0][0], maze, rsize, csize)

    if not CUSTOM:
        while True:
            r = random.randint(1, rsize * 2)
            if mazeMap[r][-2] == 1:
                continue
            mazeMap[r][-1] = 2
            destY = r
            destX = np.shape(mazeMap)[1] - 1
            # print(destX, destY)
            break
    else:
        if MAZE_NUM == 1:
            mazeMap = np.array([[1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 2],
                                [1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]])
        elif MAZE_NUM == 2:
            mazeMap = np.array([[1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 0, 2],
                                [1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]])
        elif MAZE_NUM == 3:
            mazeMap = np.array([[1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 2],
                                [1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]])
        elif MAZE_NUM == 4:
            mazeMap = np.array([[1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 2, 1, 1, 1, 1]])
        elif MAZE_NUM == 5:
            mazeMap = np.array([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        elif MAZE_NUM == 6:
            mazeMap = np.array([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 2],
                                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def reset_maze(acc_deg):
    global tk, canvas
    global posX, posY
    global maze, mazeMap
    global done
    global ball, dest
    global destY, destX

    posX = 1
    posY = 0

    if not RANDOM_MAZE:
        if acc_deg == 0:
            mazeMap = np.array(mazeMap)
        elif 360 - acc_deg == 90:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif 360 - acc_deg == 180:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif 360 - acc_deg == 270:
            mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

        visited = np.where(np.array(mazeMap) == 4)
        for i in range(0, len(visited[0])):
            mazeMap[visited[0][i]][visited[1][i]] = 0
    else:
        make_maze()

        canvas.delete('all')

        for i, r in enumerate(mazeMap):
            for j, c in enumerate(r):
                if mazeMap[i][j] == 1:
                    canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#D2D0D1', outline='#D2D0D1',
                                            width='3')
                elif mazeMap[i][j] == 2:
                    dest.place(x=j * 50 + 5, y=i * 50 + 5)
                    dest.configure()

    mazeMap[posY][posX] = 3
    mazeMap[destY][destX] = 2

    if not RANDOM_MAZE:
        g_rotate_maze(acc_deg)
        g_move_ball(acc_deg)

    done = False

    dest.place(x=j * 50 + 5, y=i * 50 + 5)
    dest.configure()

    canvas.pack()
    tk.update()


# 미로 회전
def rotate_maze(degree):
    global ROTATION_MODE, ROTATE_DELAY
    global tk, canvas
    global posX, posY
    global maze, mazeMap
    global ball, dest

    if degree == 0:
        mazeMap = np.array(mazeMap)
    elif degree == 90:
        mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
    elif degree == -90:
        mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]


def g_rotate_maze(acc_deg):
    global ROTATION_MODE, ROTATE_DELAY
    global tk, canvas
    global posX, posY
    global maze, mazeMap
    global ball, dest

    canvas.delete('all')

    if not ROTATION_MODE:
        if acc_deg == 0:
            mazeMap = np.array(mazeMap)
        elif 360-acc_deg == 90:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif 360-acc_deg == 180:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif 360-acc_deg == 270:
            mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

    for i, r in enumerate(mazeMap):
        for j, c in enumerate(r):
            canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#242C2E', outline='#242C2E',
                                    width='0')

    for i, r in enumerate(mazeMap):
        for j, c in enumerate(r):
            if mazeMap[i][j] == 1:
                canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#D2D0D1', outline='#D2D0D1',
                                        width='3')
            elif mazeMap[i][j] == 2:
                dest.place(x=j * 50 + 5, y=i * 50 + 5)
                dest.configure()

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    ball.place(x=posX * 50 + 5, y=posY * 50 + 3)
    ball.configure()

    canvas.pack()
    tk.update()

    if not ROTATION_MODE:
        if acc_deg == 0:
            mazeMap = np.array(mazeMap)
        elif acc_deg == 90:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif acc_deg == 180:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif acc_deg == 270:
            mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    time.sleep(ROTATE_DELAY)


def move_ball():
    global mazeMap
    global posX, posY
    global rsize

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    if mazeMap[posY][posX] != 4:
        mazeMap[posY][posX] = 4

    if posY + 1 < rsize * 2 + 1:
        if mazeMap[posY + 1][posX] != 1:
            posY += 1


def g_move_ball(acc_deg):
    global ROTATION_MODE, ROTATE_DELAY
    global tk, canvas
    global mazeMap
    global ball, dest
    global posX, posY

    if not ROTATION_MODE:
        if acc_deg == 0:
            mazeMap = np.array(mazeMap)
        elif 360-acc_deg == 90:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif 360-acc_deg == 180:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif 360-acc_deg == 270:
            mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

    # 공의 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 공의 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    ball.place(x=posX * 50 + 5, y=posY * 50 + 3)
    ball.configure()

    canvas.pack()
    tk.update()

    if not ROTATION_MODE:
        if acc_deg == 0:
            mazeMap = np.array(mazeMap)
        elif acc_deg == 90:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif acc_deg == 180:
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
            mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        elif acc_deg == 270:
            mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    time.sleep(MOVE_DELAY)


class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()

        self.fc1 = Dense(16, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        # self.fc3 = Dense(32, activation='relu')
        # self.dropout = Dropout(0.5)
        self.flatten = Flatten(input_shape=state_size)
        self.fc_out = Dense(action_size, activation='linear')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, action_size=3, state_size=state_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        # self.exploration_steps = 1000000.
        # self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        # self.epsilon_decay_step /= self.exploration_steps
        self.epsilon_decay_step = 0.99
        self.max_step = 500
        self.batch_size = 32
        self.train_start = 500000
        self.train_freq = 1
        self.update_target_rate = 500

        # 리플레이 메모리, 최대 크기 1000000
        self.memory = deque(maxlen=1000000)
        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 50

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)

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
        q_value = self.model.call(history)
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_value = self.model.call(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                              self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Steps/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                              self.avg_loss / float(step), step=episode)

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            # self.epsilon -= self.epsilon_decay_step
            self.epsilon *= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch = random.sample(self.memory, self.batch_size)

        # 1) simple state
        state = np.array([sample[0][0] for sample in batch],
                         dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
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

            self.avg_loss += loss.numpy()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


def proceed():
    global ROTATION_MODE
    global rsize, csize
    global posX, posY
    global destX, destY
    global tk, canvas
    global mazeMap
    global done
    global ball
    global start_time

    time.sleep(1)

    agent = DQNAgent(action_size=3, state_size=state_size)

    agent.model.build(input_shape=state_size)
    agent.target_model.build(input_shape=state_size)

    agent.model.summary()
    agent.target_model.summary()

    global_step = 0
    score_avg = 0
    score_max = 0

    degree = 0  # 회전 각도
    rotate = {0: 0, 1: 90, 2: -90}

    acc_deg = 0  # 누적 회전 각도

    move_delay = 0.00

    num_episode = 1000
    for e in range(1, num_episode+1):
        reward = 0
        done = False

        step, score = 0, 0

        # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
        state = np.float32(mazeMap)

        state = state / 10.
        state = np.reshape([state], (1, rsize * 2 + 1, csize * 2 + 1))

        while not done:
            # time.sleep(1)
            # print(posX, posY)

            # 바로 전 state를 입력으로 받아 행동을 선택
            # 0: 0도, 1: 90도, 2: -90도
            action = agent.get_action(np.float32(state))

            global_step += 1
            step += 1

            # 현재 각도 기준으로 회전
            degree = rotate[action]

            # 회전 각도 누적 (그래픽 출력용)
            acc_deg += degree
            acc_deg %= 360

            # 미로 회전
            rotate_maze(degree)

            # 회전 후 미로 그래픽 출력
            g_rotate_maze(acc_deg)

            # 공의 하단에 길이 있을 시 공의 좌표 값 변경
            move_ball()

            # 도착 지점 도착 시
            if mazeMap[posY][posX] == 2:
                # 도착 지점을 공의 정보로 표시
                mazeMap[posY][posX] = 3

                g_move_ball(acc_deg)

                done = True
                reward = 1

            if not done:
                # 1) 이전에 방문했던 블럭 재방문 시
                if mazeMap[posY][posX] == 4 or mazeMap[posY][posX] == 3:
                    # 중간보상 방식
                    reward = -0.015
                #
                #     # 에피소드 종료 방식
                #     # reward = -1
                #     # done = True
                # else:
                #     reward = 0.005

                # 2) 스텝마다 - 보상
                # reward = -0.01

                mazeMap[posY][posX] = 3

                g_move_ball(acc_deg)

            # 중간 보상 : 맨해튼 거리
            if not done:
                if USE_MAX_STEP:
                    if step == agent.max_step:
                        reward = -1
                        done = True
                # else:
                # reward = -1 / 400 + ((np.abs(destX-posX) + np.abs(destY-posY))) / 10000
                # reward = -round((np.abs(destX-posX) + np.abs(destY-posY)) / 10, 3)
                # reward = -1 / ((np.abs(destX-posX) + np.abs(destY-posY) + 0.001))
                # reward = -round((np.abs(destX-posX) + np.abs(destY-posY)) / 8, 2) + 0.5
                # reward += -round(1 / 900, 3)
                # reward = -((np.abs(destX-posX) + np.abs(destY-posY))) / 10
            # print(reward)

            # 각 타임스텝마다 상태 전처리
            next_state = np.float32(mazeMap)

            # 1) 2D Array
            next_state = next_state / 10.
            next_state = np.reshape([next_state], (1, rsize * 2 + 1, csize * 2 + 1))

            # 가장 큰 Q값 가산
            agent.avg_q_max += np.amax(agent.model.call(np.float32([state])))

            score += reward
            reward = np.clip(reward, -1., 1.)

            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(state, action, reward, next_state, done)

            state = next_state

            # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
            if len(agent.memory) >= agent.train_start:
                if step % agent.train_freq == 0:
                    agent.train_model()
                    # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()

            if done:
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

        # 일정 에피소드마다 모델 저장
        if e % 1 == 0:
            agent.model.save_weights("save_model/" + start_time + "/model" + str(e), save_format="tf")

        degree = 0
        reset_maze(acc_deg)
        acc_deg = 0


def generate():
    global mazeMap
    global rsize, csize
    global tk, canvas
    global posX, posY
    global ball, dest

    tk = tkinter.Tk()
    tk.title('Maze Map')
    canvas = tkinter.Canvas(width=(csize * 2 + 1) * 50, height=(rsize * 2 + 1) * 50, bg='#242C2E')

    # img = tkinter.PhotoImage(file='ball.png').subsample(25)
    # dest_img = tkinter.PhotoImage(file='images/star.png').subsample(7)
    # dest_img = tkinter.PhotoImage(file='images/star2.png')
    dest_img = tkinter.PhotoImage(file='images/dest.png')
    dest = tkinter.Label(image=dest_img, borderwidth=0)

    # ball_img = tkinter.PhotoImage(file='images/ball.png').subsample(25)
    ball_img = tkinter.PhotoImage(file='images/ball2.png')
    ball = tkinter.Label(image=ball_img, borderwidth=0)

    for i, r in enumerate(mazeMap):
        for j, c in enumerate(r):
            if mazeMap[i][j] == 1:
                canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#D2D0D1', outline='#D2D0D1',
                                        width='3')
            elif mazeMap[i][j] == 2:
                dest.place(x=j * 50 + 5, y=i * 50 + 5)
                dest.configure()

    mazeMap[posY][posX] = 3

    ball.place(x=posX * 50 + 5, y=posY * 50 + 3)
    ball.configure()

    canvas.pack()

    tk.after(1000, proceed)

    tk.focus_force()
    tk.mainloop()


maze = []
mazeMap = []

posX = 1
posY = 0

if CUSTOM:
    if MAZE_NUM == 1:
        destX = 6
        destY = 1
    elif MAZE_NUM == 2 or MAZE_NUM == 3:
        destX = 6
        destY = 2
    elif MAZE_NUM == 4:
        destX = 2
        destY = 6
    elif MAZE_NUM == 5:
        destX = 10
        destY = 9
    elif MAZE_NUM == 6:
        destX = 14
        destY = 4
else:
    destX = 0
    destY = 0

tk = ''
canvas = ''

ball = ''
dest = ''

done = False

make_maze()
generate()
