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

PLAY_MODE = 0
ROTATION_MODE = False  # 미로 회전 그래픽 출력
ROTATE_DELAY = 0.0
MOVE_DELAY = 0.0
USE_MAX_STEP = False


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

    maze = [[Room(r, c) for c in range(csize)] for r in range(rsize)]
    mazeMap = [[1 for c in range(csize * 2 + 1)] for r in range(rsize * 2 + 1)]

    make(None, maze[0][0], maze, rsize, csize)

    while True:
        r = random.randint(1, rsize * 2)
        if mazeMap[r][-2] == 1:
            continue
        mazeMap[r][-1] = 2
        destY = r
        destX = np.shape(mazeMap)[1] - 1
        # print(destX, destY)
        break


def reset_maze(acc_deg):
    global tk, canvas
    global posX, posY
    global maze, mazeMap
    global done
    global ball, star
    global destY, destX

    posX = 1
    posY = 0

    if acc_deg == 0:
        mazeMap = np.array(mazeMap)
    elif 360 - acc_deg == 90:
        mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
    elif 360 - acc_deg == 180:
        mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
        mazeMap = np.array(list(map(list, zip(*mazeMap[::-1]))))
    elif 360 - acc_deg == 270:
        mazeMap = np.array(list(map(list, zip(*mazeMap)))[::-1])

    mazeMap[posY][posX] = 3

    visited = np.where(np.array(mazeMap) == 4)
    for i in range(0, len(visited[0])):
        mazeMap[visited[0][i]][visited[1][i]] = 0

    mazeMap[destY][destX] = 2

    rotate_maze(acc_deg)
    move_player(acc_deg)

    done = False

    tk.update()


def rotate_maze(acc_deg):
    global ROTATION_MODE, ROTATE_DELAY
    global tk, canvas
    global posX, posY
    global maze, mazeMap
    global ball, star

    # print('rotate maze')
    # pprint.pprint(mazeMap)

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
                                    width='5')

    for i, r in enumerate(mazeMap):
        for j, c in enumerate(r):
            if mazeMap[i][j] == 1:
                canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#D2D0D1', outline='#D2D0D1',
                                        width='5')
            elif mazeMap[i][j] == 2:
                star.place(x=j * 50 + 5, y=i * 50 + 5)
                star.configure()

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    ball.place(x=posX * 50 + 7, y=posY * 50 + 7)
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


def move_player(acc_deg):
    global ROTATION_MODE, ROTATE_DELAY
    global tk, canvas
    global mazeMap
    global ball, star
    global posX, posY

    # print('move player')
    # pprint.pprint(mazeMap)

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

    # 플레이어 y 좌표
    posY = np.where(mazeMap == 3)[0][0]
    # 플레이어 x 좌표
    posX = np.where(mazeMap == 3)[1][0]

    ball.place(x=posX * 50 + 7, y=posY * 50 + 7)
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


def get_key(key):
    return key


def get_maze():
    global mazeMap
    return mazeMap


def get_score():
    global score
    return score


def proceed():
    global ROTATION_MODE
    global rsize, csize
    global posX, posY
    global destX, destY
    global tk, canvas
    global mazeMap
    global done
    global ball
    global input_flag, reward_flag

    rotate = {0: 0, 1: 90, 2: -90}

    acc_deg = 0  # 누적 회전 각도

    while not done:
        degree = 0  # 회전 각도

        # 현재 각도 기준으로 회전
        if input_flag:
            degree = rotate[key]
        else:
            degree = 0

        input_flag = False

        # 회전 각도 누적 (그래픽 출력용)
        acc_deg += degree
        acc_deg %= 360

        # 미로 회전
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

        # 회전 후 미로 그래픽 출력
        rotate_maze(acc_deg)

        if mazeMap[posY][posX] != 4:
            mazeMap[posY][posX] = 4

        if posY + 1 < rsize * 2 + 1:
            if mazeMap[posY + 1][posX] != 1:
                posY += 1

        # 도착지점 도착 시
        if mazeMap[posY][posX] == 2:
            # 도착지점을 공 정보로 표시
            mazeMap[posY][posX] = 3

            move_player(acc_deg)

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

            mazeMap[posY][posX] = 3

            move_player(acc_deg)

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

        while not reward_flag:
            pass

        reward_flag = False

        degree = 0
        reset_maze(acc_deg)
        acc_deg = 0

        time.sleep(0.5)


def generate():
    global mazeMap
    global rsize, csize
    global tk, canvas
    global posX, posY
    global ball, star

    tk = tkinter.Tk()
    tk.title('Maze Map')
    canvas = tkinter.Canvas(width=(csize * 2 + 1) * 50, height=(rsize * 2 + 1) * 50, bg='#242C2E')

    for i, r in enumerate(mazeMap):
        for j, c in enumerate(r):
            if mazeMap[i][j] == 1:
                canvas.create_rectangle(j * 50, i * 50, j * 50 + 50, i * 50 + 50, fill='#D2D0D1', outline='#D2D0D1',
                                        width='5')
            elif mazeMap[i][j] == 2:
                # img = tkinter.PhotoImage(file='ball.png').subsample(25)
                img = tkinter.PhotoImage(file='star.png').subsample(7)
                img.zoom(50, 50)

                star = tkinter.Label(image=img, borderwidth=0)
                star.image = img
                star.place(x=j * 50 + 5, y=i * 50 + 5)
                star.configure()

    mazeMap[posY][posX] = 3

    # img = tkinter.PhotoImage(file='player.png').subsample(6)
    img = tkinter.PhotoImage(file='ball.png').subsample(25)
    img.zoom(50, 50)

    ball = tkinter.Label(image=img, borderwidth=0)
    ball.image = img
    ball.place(x=posX * 50 + 7, y=posY * 50 + 7)
    ball.configure()

    canvas.pack()

    tk.after(1000, proceed)

    tk.focus_force()
    tk.mainloop()


rsize = 7
csize = 7

rsize = int(rsize / 2)
csize = int(csize / 2)

maze = []
mazeMap = []

input_flag = False
reward_flag = False

key = 0
reward = 0

posX = 1
posY = 0

destX = 0
destY = 0

tk = ''
canvas = ''

ball = ''
star = ''

done = False

make_maze()
generate()
