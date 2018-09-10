import collections
import sys
import time

class MazeRunner:

    def __init__(self, file_path):
        self.maze = self.load_maze(file_path)
        self.Point = collections.namedtuple('Point', 'row col')
        self.char_wheel = ord('a')
        self.paths = []

    def load_maze(self, file_path):
        with open(file_path,'r') as f:
            maze = [list(line.rstrip()) for line in f.readlines()]
        return maze

    def show_maze(self):
        for point in self.paths:
            self.maze[point.row][point.col] = chr(self.char_wheel)
            self.char_wheel += 1

            if self.char_wheel > ord('z'):
                self.char_wheel = ord('a')

        for row in self.maze:
            print(''.join(row))

    def mark_startstop(self):
        #Mark starting point with '+'
        col_ = self.maze[0].index('_')
        self.maze[0][col_] = '+'

        #Mark stopping point with '-'
        col_ = self.maze[-1].index('_')
        self.maze[-1][col_] = '-'

    def adjacent(self, current):
        yield self.Point(current.row-1, current.col)
        yield self.Point(current.row+1, current.col)
        yield self.Point(current.row, current.col-1)
        yield self.Point(current.row, current.col+1)

    def neighbors(self, current):
        for pos in self.adjacent(current):
            if pos.row in range(0,len(self.maze)) and pos.col in range(0,len(self.maze[0])):
                if self.maze[pos.row][pos.col] in ['_','-']:
                    yield pos

    def solve(self, current):

        # print('\nPathing --> ',self.paths)
        # print('Attempting solve('+str(current)+')\n')

        #Base cases
        if len(self.paths) < 1:
            print('Unsuccessful')
            return False

        elif self.maze[current.row][current.col] == '-':
            print('Maze completed!')
            return True

        #Recursive case
        else:
            for point in self.neighbors(current):
                # print('\tChecking out '+str(point)+' - neighbor of '+str(current))
                if  self.maze[point.row][point.col] in ['_','-'] and point not in self.paths:
                    self.paths.append(point)
                    return self.solve(self.paths[-1])
                else:
                    pass
                    # print('\tSkipping '+str(point))


        #Handle Dead-end back-tracking
        self.maze[current.row][current.col] = '!'
        temp = self.paths.pop()
        # print('\tReached Dead-end. Back-tracking. Popped '+str(temp))
        return self.solve(self.paths[-1])


    def find_path(self):

        self.mark_startstop()
        start_col = self.maze[0].index('+')
        start_row = 0
        self.paths.append(self.Point(start_row, start_col))
        t0 = time.clock()
        self.solve(self.paths[-1])
        t1 = time.clock()
        print("Solution found in %.2fms" % ((t1-t0)*1000,))
def main():  

    # parser = ArgumentParser('Maze Runner', description="This program finds solution to mazes given as input")
    # parser.add_argument('--path', default='maze.txt', help='Path of the file containing the maze')

    # args = parser.parse_args()

    # path = args.path
    path = "maze.txt"
    solver = MazeRunner(path)

    print('Maze loaded.')
    solver.show_maze()

    solver.find_path()
    solver.show_maze()  

sys.setrecursionlimit(10000)
main()