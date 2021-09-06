import random
import numpy as np
import tkinter


class Room:
    def __init__(self, r, c):
        self.r, self.c = r, c
        self.visit = 0
        self.prev = None
        self.drct = [(r + 1, c), (r, c + 1),
                     (r - 1, c), (r, c - 1)]
        random.shuffle(self.drct)


class Player:
    def __init__(self):
        self.r = 0
        self.c = 1


def make(prev, room, _maze, _rsize, _csize):
    room.prev = prev
    if room.prev is None:
        mazeMap[0][1] = '↓'
    else:
        r = prev.r - room.r
        c = prev.c - room.c
        mazeMap[(room.r + 1) * 2 - 1 + r][(room.c + 1) * 2 - 1 + c] = '  '
    
    room.visit = 1
    mazeMap[(room.r+1)*2-1][(room.c+1)*2-1] = '  '
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


def keyClick(e):
    global key
    global posX, posY

    key = e.keysym
    print(key)

    if key == 'Up':
        if posY > 0 and posY <= 15:
            posY -= 1
    if key == 'Down':
        if posY >= 0 and posY < 15:
            posY += 1
    if key == 'Right':
        if posX >= 0 and posX < 31:
            posX += 1
    if key == 'Left':
        if posX > 0 and posX <= 31:
            posX -= 1

    print(posX, posY)

    canvas.coords('player', posX*50+25, posY*50+25)


rsize = 7
csize = 15

maze = [[Room(r, c) for c in range(csize)] for r in range(rsize)]
mazeMap = [['■' for c in range(csize*2+1)] for r in range(rsize*2+1)]
print(np.shape(maze))
print(np.shape(mazeMap))

make(None, maze[0][0], maze, rsize, csize)

while True:
    r = random.randint(1, rsize*2)
    if mazeMap[r][-2] == '■':
        continue
    mazeMap[r][-1] = '→'
    break

file = open('maze.txt', 'w')
for r in mazeMap:
    for c in r:
        file.write(c)
    file.write('\n')
file.close()

key = 0
posX = 1
posY = 0

tk = tkinter.Tk()
tk.title('Maze Map')
canvas = tkinter.Canvas(width=(csize*2+1)*50, height=(rsize*2+1)*50, bg='white')

for i, r in enumerate(mazeMap):
    for j, c in enumerate(r):
        if mazeMap[i][j] == '■':
            canvas.create_rectangle(j*50, i*50, j*50+50, i*50+50, fill='grey', outline='black', width='5')
        elif mazeMap[i][j] == '↓':
            canvas.create_oval(j*50+5, i*50+5, j*50+50-5, i*50+50-5, fill='green', outline='white', width='5')
        elif mazeMap[i][j] == '→':
            canvas.create_oval(j*50+5, i*50+5, j*50+50-5, i*50+50-5, fill='blue', outline='white', width='5')

tk.bind('<Key>', keyClick)
img = tkinter.PhotoImage(file='player.png').subsample(5)

player = Player()
canvas.create_image(posX*50+25, posY*50+25, image=img, tag='player')

canvas.pack()
tk.mainloop()
