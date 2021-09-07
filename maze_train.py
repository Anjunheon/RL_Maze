import random
import numpy as np
import tkinter
import os
import time
import tensorflow as tf
from collections import deque

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten


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

    maze = [[Room(r, c) for c in range(csize)] for r in range(rsize)]
    mazeMap = [[1 for c in range(csize * 2 + 1)] for r in range(rsize * 2 + 1)]
    # print(np.shape(maze))
    # print(np.shape(mazeMap))

    make(None, maze[0][0], maze, rsize, csize)

    while True:
        r = random.randint(1, rsize * 2)
        if mazeMap[r][-2] == 1:
            continue
        mazeMap[r][-1] = 2
        break


def reset_game():
    global tk, canvas
    global posX, posY
    global maze, mazeMap
    global done

    # tk.destroy()

    posX = 1
    posY = 0

    maze = []
    # mazeMap = []

    done = False

    # make_maze()
    # generate()

    tk.update()


class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        # self.fc2 = Dense(128, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, action_size=4, state_size=2):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 0.02
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.batch_size = 32
        self.train_start = 100
        self.update_target_rate = 5

        # 리플레이 메모리, 최대 크기 100,000
        self.memory = deque(maxlen=100000)
        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 30

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(self.learning_rate, clipnorm=10.)
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
            return random.randint(0, self.action_size-1)
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
            self.epsilon -= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch = random.sample(self.memory, self.batch_size)

        history = np.array([sample[0][0] for sample in batch],
                           dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] for sample in batch],
                                dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        # 학습 파라메터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model.call(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model.call(next_history)

            # 벨만 최적 방정식을 구성하기 위한 타깃과 큐함수의 최대 값 계산
            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q

            # 후버로스 계산
            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            self.avg_loss += loss.numpy()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


def nothing(gs, s):
    gs -= 1
    s -= 1


def move():
    agent = DQNAgent(action_size=4)

    global posX, posY
    global tk, canvas
    global mazeMap
    global done

    global_step = 0
    score_avg = 0
    score_max = 0

    move_delay = 0.02

    # 불필요한 행동을 없애주기 위한 딕셔너리 선언
    action_dict = {0: 1, 1: 2, 2: 3, 3: 3}

    num_episode = 50000
    for e in range(0, num_episode):
        reward = 0
        done = False

        step, score = 0, 0
        # 미로 생성
        # generate()

        # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
        state = [posX, posY]
        history = np.float32([state, state, state, state])

        while not done:
            global_step += 1
            step += 1

            # 바로 전 history를 입력으로 받아 행동을 선택
            # 0: 위, 1: 아래, 2: 오른쪽, 3: 왼쪽
            action = agent.get_action(history)

            if action == 0:
                if posY - 1 >= 0:
                    if mazeMap[posY - 1][posX] != 1:
                        posY -= 1
                    else:
                        nothing(global_step, step)
                        continue
                else:
                    nothing(global_step, step)
                    continue
            if action == 1:
                if posY + 1 < rsize*2+1:
                    if mazeMap[posY + 1][posX] != 1:
                        posY += 1
                    else:
                        nothing(global_step, step)
                        continue
                else:
                    nothing(global_step, step)
                    continue
            if action == 2:
                if posX + 1 < csize*2+1:
                    if mazeMap[posY][posX + 1] != 1:
                        posX += 1
                    else:
                        nothing(global_step, step)
                        continue
                else:
                    nothing(global_step, step)
                    continue
            if action == 3:
                if posX - 1 > 0:
                    if mazeMap[posY][posX - 1] != 1:
                        posX -= 1
                    else:
                        nothing(global_step, step)
                        continue
                else:
                    nothing(global_step, step)
                    continue

            canvas.coords('player', posX * 50 + 25, posY * 50 + 25)
            canvas.pack()

            tk.update()

            time.sleep(move_delay)

            # print(posY, posX)
            # print(mazeMap[posY][posX])

            if mazeMap[posY][posX] == 2:
                # print('Goal!')
                done = True
                reward = 1

            # 각 타임스텝마다 상태 전처리
            next_state = [posY, posX]

            next_history = np.float32([next_state, history[0], history[1], history[2]])

            # 가장 큰 Q값 가산
            agent.avg_q_max += np.amax(agent.model.call(np.float32(history)))

            score += reward
            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, done)

            # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
                # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

            if done:
                history = np.float32([next_state, next_state, next_state, next_state])
            else:
                history = next_history

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    agent.draw_tensorboard(score, step, e)

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                score_max = score if score > score_max else score_max

                log = "episode: {:5d} | ".format(e)
                # log += "score: {:4.1f} | ".format(score)
                # log += "score max : {:4.1f} | ".format(score_max)
                # log += "score avg: {:4.1f} | ".format(score_avg)
                log += "memory length: {:5d} | ".format(len(agent.memory))
                log += "epsilon: {:.3f} | ".format(agent.epsilon)
                log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
                print(log)

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights("./save_model/model", save_format="tf")

        reset_game()


def generate():
    global mazeMap
    global rsize, csize
    global tk, canvas
    global posX, posY

    tk = tkinter.Tk()
    tk.title('Maze Map')
    canvas = tkinter.Canvas(width=(csize * 2 + 1) * 50, height=(rsize * 2 + 1) * 50, bg='#242C2E')

    for i, r in enumerate(mazeMap):
        for j, c in enumerate(r):
            if mazeMap[i][j] == 1:
                canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#D2D0D1', outline='#D2D0D1', width='5')
            elif mazeMap[i][j] == 2:
                # canvas.create_oval(j * 50 + 5, i * 50 + 5, j * 50 + 50 - 5, i * 50 + 50 - 5, fill='#C3B0EA', outline='#242C2E', width='5')
                img = tkinter.PhotoImage(file='ball.png').subsample(25)
                img.zoom(50, 50)

                # canvas.create_image(j * 50 + 5, i * 50 + 5, image=img, tag='ball')
                # canvas.coords('player', j * 50 + 5, i * 50 + 5)

                label = tkinter.Label(image=img, borderwidth=0)
                label.image = img
                label.place(x=j*50+10, y=i*50+10)
                label.configure()

    img = tkinter.PhotoImage(file='player.png').subsample(6)
    img.zoom(50, 50)

    canvas.create_image(posX * 50 + 25, posY * 50 + 25, image=img, tag='player')
    canvas.pack()

    tk.after(1000, move)

    tk.focus_force()
    tk.mainloop()


rsize = 2
csize = 2

maze = []
mazeMap = []

key = 0
posX = 1
posY = 0

tk = ''
canvas = ''

action_size = 4
state_size = 2

done = False

make_maze()
generate()
