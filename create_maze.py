import random
import numpy as np
import tkinter
import time
import keyboard


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
    print('\n현재좌표: {0},{1}'.format(room.r, room.c))

    while True:
        if len(room.drct) == 0:
            break
        nr, nc = room.drct.pop()
        print('좌표체크: {0},{1}'.format(nr, nc))

        if nr >= 0 and nr < _rsize and nc >= 0 and nc < _csize:
            if not _maze[nr][nc].visit == 1:
                print('{0},{1} 좌표로 진행'.format(nr, nc))
                make(room, _maze[nr][nc], _maze, _rsize, _csize)
            else:
                print('방문기록있음')
        else:
            print('진행불가')


rsize = 7
rsize = 3
csize = 15
csize = 3

maze = []
mazeMap = []


def make_maze():
    global maze, mazeMap
    global rsize, csize

    maze = [[Room(r, c) for c in range(csize)] for r in range(rsize)]
    mazeMap = [[1 for c in range(csize * 2 + 1)] for r in range(rsize * 2 + 1)]
    print(np.shape(maze))
    print(np.shape(mazeMap))

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

    tk.destroy()

    posX = 1
    posY = 0

    maze = []
    maze = []

    make_maze()
    generate()


make_maze()

key = 0
posX = 1
posY = 0

tk = ''
canvas = ''


def move():
    global posX, posY
    global tk, canvas
    global mazeMap

    while True:
        dir = random.randint(0, 3)
        print(dir)
        if dir == 0:
            if posY - 1 >= 0:
                if mazeMap[posY-1][posX] != 1:
                    posY -= 1
        if dir == 1:
            if posY + 1 < 15:
                if mazeMap[posY+1][posX] != 1:
                    posY += 1
        if dir == 2:
            if posX + 1 < 31:
                if mazeMap[posY][posX+1] != 1:
                    posX += 1
        if dir == 3:
            if posX - 1 > 0:
                if mazeMap[posY][posX-1] != 1:
                    posX -= 1

        print(posX, posY)

        canvas.coords('player', posX * 50 + 25, posY * 50 + 25)
        canvas.pack()

        tk.update()

        print('move')

        time.sleep(0.05)

        if mazeMap[posY][posX] == 2:
            print('Goal!')
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
                canvas.create_oval(j * 50 + 5, i * 50 + 5, j * 50 + 50 - 5, i * 50 + 50 - 5, fill='#C3B0EA', outline='#242C2E', width='5')

    img = tkinter.PhotoImage(file='player.png').subsample(5)
    img.zoom(50, 50)

    canvas.create_image(posX * 50 + 25, posY * 50 + 25, image=img, tag='player')

    canvas.pack()

    tk.focus_force()

    tk.geometry('%dx%d+%d+%d' % ((csize * 2 + 1) * 50, (rsize * 2 + 1) * 50, 700, 300))
    tk.after(1000, move)
    tk.mainloop()


generate()
