import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
import os

nL = int(os.environ["NUM_STOPS"])

class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are nL designated locations in the grid world indicated by the first nL members of the set: A, B, C, D, E, F, G, H
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the nL specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    State counts:
    if nL == 2:
        There are 125 discrete states since there are 25 taxi positions, 3 possible locations of the passenger, and 2 destination locations.
    if nL == 4:
        There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger, and 4 destination locations.
    if nL == 5:
        There are 750 discrete states since there are 25 taxi positions, 6 possible locations of the passenger, and 24 destination locations.
    if nL == 6:
        There are 1050 discrete states since there are 25 taxi positions, 7 possible locations of the passenger, and 6 destination locations.
    if nL == 8:
        There are 1800 discrete states since there are 25 taxi positions, 9 possible locations of the passenger, and 8 destination locations.
    
    Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        if nL == 2:
            MAP = [
                "+---------+",
                "|A: | : : |",
                "| : : : : |",
                "| : : : : |",
                "| | : | : |",
                "| | : |B: |",
                "+---------+",
            ]
            self.locs = locs = [(0,0), (4,3)]
        elif nL == 4:
            MAP = [
                "+---------+",
                "|A: | : :B|",
                "| : : : : |",
                "| : : : : |",
                "| | : | : |",
                "|C| : |D: |",
                "+---------+",
            ]
            self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]
        elif nL == 5:
            MAP = [
                "+---------+",
                "|A: | : :B|",
                "| : : : : |",
                "| : : : :C|",
                "| | : | : |",
                "|D| : |E: |",
                "+---------+",
            ]
            self.locs = locs = [(0,0), (0,4), (2,4), (4,0), (4,3)]
        elif nL == 6:
            MAP = [
                "+---------+",
                "|A: | : :B|",
                "| : : : : |",
                "| :C:D: : |",
                "| | : | : |",
                "|E| : |F: |",
                "+---------+",
            ]
            self.locs = locs = [(0,0), (0,4), (2,1), (2,2), (4,0), (4,3)]
        elif nL == 8:
            MAP = [
                "+---------+",
                "|A: |B: :C|",
                "| :D: : : |",
                "| : : :E:F|",
                "| | : | : |",
                "|G| : |H: |",
                "+---------+",
            ]
            self.locs = locs = [(0,0), (0,2), (1,1), (2,3), (2,4), (0,4), (4,0), (4,3)]


        self.desc = np.asarray(MAP,dtype='c')

        nR = 5 # rows
        nC = 5 # columns
        # states
        if nL == 2:
            nS = nR * nC * nL * (nL+1) * 2
        else:
            nS = nR * nC * nL * (nL+1)
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 6 # actions
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for row in range(5):
            for col in range(5):
                for passidx in range(nL+1):
                    for destidx in range(nL):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < nL and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a==0:
                                newrow = min(row+1, maxR)
                            elif a==1:
                                newrow = max(row-1, 0)
                            if a==2 and self.desc[1+row,2*col+2]==b":":
                                newcol = min(col+1, maxC)
                            elif a==3 and self.desc[1+row,2*col]==b":":
                                newcol = max(col-1, 0)
                            elif a==4: # pickup
                                if (passidx < nL and taxiloc == locs[passidx]):
                                    newpassidx = nL
                                else:
                                    reward = -10
                            elif a==5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx==nL:
                                    done = True
                                    reward = 20
                                elif (taxiloc in locs) and passidx==nL:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            #print("row:", row, "col:", col, "passidx:", passidx, "destidx:", destidx, "a:", a)
                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, destidx):
        # (5) 5, 5, 4
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= nL
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % nL)
        i = i // nL
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if passidx < nL:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
