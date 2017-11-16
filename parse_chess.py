import chess, chess.pgn
import tensorflow
import numpy
import random
import os
import h5py

#Reads the games from a PGN file
def readGames(fn_in):
    file = open(fn_in)

    while True:
        try:
            g = chess.pgn.read_game(file)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not g:
            break

        yield g

#Convert the board to byte array
def bb2array(board, flip = False):
    x = numpy.zeros(768 ,dtype=numpy.int8)
    #Creates 12 boards for each piece type, 64 for each board: 64 * 12 = 768
    for pos in range(64):
        if board.piece_at(pos) == None:
            continue
        character = str(board.piece_at(pos))
        if character == 'p':
            x[pos] = 1
        if character == 'n':
            x[pos + (64 * 1)] = 1
        if character == 'b':
            x[pos + (64 * 2)] = 1
        if character == 'r':
            x[pos + (64 * 3)] = 1
        if character == 'q':
            x[pos + (64 * 4)] = 1
        if character == 'k':
            x[pos + (64 * 5)] = 1
        if character == 'P':
            x[pos + (64 * 6)] = 1
        if character == 'N':
            x[pos + (64 * 7)] = 1
        if character == 'B':
            x[pos + (64 * 8)] = 1
        if character == 'R':
            x[pos + (64 * 9)] = 1
        if character == 'Q':
            x[pos + (64 * 10)] = 1
        if character == 'K':
            x[pos + (64 * 11)] = 1
    return x



#Parsing the information from a PGN game
def parse_game(game):
    #Checking the final result of the Game
    rm = {'1-0': 1, '0-1':-1,'1/2-1/2':0}
    r = game.headers['Result']
    if r not in rm:
        return None
    y = rm[r]

    #Generate board all the boards in the game
    #Get last game state and iterate backwards to the beginning of the game
    gState = game.end()
    if not gState.board().is_game_over():
        return None
    gStateList = []
    moves_left = 0
    #Iterates while there are moves
    while gState:
        gStateList.append((moves_left, gState, gState.board().turn == 0))
        #Going back a game state and increasing amount of moves left in game
        gState = gState.parent
        moves_left += 1

    print(len(gStateList))
    if len(gStateList) < 10:
        print(game.end())
    gStateList.pop()
    # Removes the first position
    moves_left, gState, flip = random.choice(gStateList)
    board = gState.board()
    x = bb2array(board, flip = flip)
    board_parent = gState.parent.board()
    x_parent = bb2array(board_parent,flip = (not flip))
    if flip:
        y = -y
    #Generate a random board
    moves = list(board_parent.legal_moves)
    move = random.choice(moves)
    board_parent.push(move)
    x_random = bb2array(board_parent,flip=flip)

    if moves_left < 3:
        print(moves_left)
        print("winner: ", y)
        print(game.headers)
        print(board)
        print("Checkmate: ",game.end().board().is_checkmate())

    return (x,x_parent,x_random,moves_left,y)

#Coverts the PNG file to a HDF5
def convertFile(filename):
    #Getting location of current directory and swapping the extention to hdf5
    files = []
    dir = 'games'
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    fDir = os.path.join(__location__, dir)
    for fn_in in os.listdir(fDir):
        if not fn_in.endswith('.pgn'):
            continue
        fn_in = os.path.join(fDir, fn_in)
        fn_out = fn_in.replace('.pgn', '.hdf5')
        #if not os.path.exists(fn_out):
        files.append((fn_in,fn_out))
    print(fn_in)
    return  files

def readhdf5(fn_in, fn_out):

    g = h5py.File(fn_out, 'w')
    #Creating databases
    X, Xr, Xp = [g.create_dataset(d, (0,768), dtype='b', maxshape = (None, 768), chunks =True) for d in ['x','xr', 'xp']]
    Y, M = [g.create_dataset(d, (0,), dtype='b', maxshape=(None,), chunks=True) for d in ['y', 'm']]
    size = 0
    line = 0
    #Opening files in game
    for game in readGames(fn_in):
        game = parse_game(game)
        if game == None:
            continue
        x, x_parent, x_random, moves_left,y = game

        if line + 1 >= size:
            g.flush()
            size = 2 * size + 1
            print('resizing to ', size)
            [d.resize(size = size, axis = 0) for d in (X,Xr,Xp,Y,M)]
        X[line] = x
        Xr[line] = x_random
        Xp[line] = x_parent
        Y[line] = y
        M[line] = moves_left

        line += 1

    [d.resize(size=line, axis=0) for d in (X, Xr, Xp, Y, M)]
    g.close()

#f = open(os.path.join(__location__, fn));
#print(f.read())
#board = chess.Board()
#print(board.legal_moves)
#print(board)
#print(board.variation_san([chess.Move.from_uci("e2e4")]))
#print(board)

def main():
    filename = 'ficsgame.pgn'
    files = convertFile(filename)
    splitData = list(files[0])
    readhdf5(splitData[0],splitData[1])


if __name__ == "__main__":
    main()