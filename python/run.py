import numpy as np
import argparse
from collections import namedtuple

def transition_prop(start, end):
    # from absorbing states
    if start.attackers == 0 or start.defenders == 0:
        return (start.attackers == end.attackers and 
               start.defenders == end.defenders)

    # from transient states
    i = 3 if start.attackers >= 3 else start.attackers
    j = 2 if start.defenders >= 2 else start.defenders
    k = start.defenders - end.defenders # how many units defender loses
    l = start.attackers - end.attackers # how many units attacker loses
    if i == 1 and j == 1 and l == 0 and k == 1:
        return 15/36.0
    if i == 1 and j == 1 and l == 1 and k == 0:
        return 21/36.0
    if i == 1 and j == 2 and l == 0 and k == 1:
        return 55/216.0
    if i == 1 and j == 2 and l == 1 and k == 0:
        return 161/216.0
    if i == 2 and j == 1 and l == 0 and k == 1:
        return 125/216.0
    if i == 2 and j == 1 and l == 1 and k == 0:
        return 91/216.0
    if i == 2 and j == 2 and l == 0 and k == 2:
        return 295/1296.0
    if i == 2 and j == 2 and l == 1 and k == 1:
        return 420/1296.0
    if i == 2 and j == 2 and l == 2 and k == 0:
        return 581/1296.0
    if i == 3 and j == 1 and l == 0 and k == 1:
        return 855/1296.0
    if i == 3 and j == 1 and l == 1 and k == 0:
        return 441/1296.0
    if i == 3 and j == 2 and l == 0 and k == 2:
        return 2890/7776.0
    if i == 3 and j == 2 and l == 1 and k == 1:
        return 2611/7776.0
    if i == 3 and j == 2 and l == 2 and k == 0:
        return 2275/7776.0
    return 0

def calc_states(A, D):
    State = namedtuple('State', ['attackers', 'defenders'])
    states = []
    # transient states
    for a in range(1, A+1):
        for d in range(1, D+1):
            states.append(State(a,d))
    # absorbing states
    for d in range(1, D+1):
        states.append(State(0, d))
    for a in range(1, A+1):
        states.append(State(a, 0))

    return states

def calc_P(A, D):
    states = calc_states(A, D)
    n = len(states)
    P = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            p = transition_prop(states[i], states[j])
            P[i,j] = p
    return P

def slice_Q_R(P, A, D):
    return P[:A*D, :A*D], P[:A*D:, A*D:]

def calc_F(Q, R):
    I = np.identity(Q.shape[0])
    inv = np.linalg.inv(I-Q)
    return np.dot(inv, R)

def calc_winning_prob(F, A, D):
    WinProb = namedtuple('WinProb', ['attacker', 'defender'])
    pa, pd = 0,0
    for j in range(D, D+A):
        pa += F[A*D-1, j]
    for j in range(0, D):
        pd += F[A*D-1, j]
    
    return WinProb(pa,pd)

def calc_attacker_wins(A, D):
    P = calc_P(A, D)
    Q, R = slice_Q_R(P, A, D)
    F = calc_F(Q, R)

    winning_prob = calc_winning_prob(F, A, D)
    return winning_prob.attacker

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('a', help="nr of attackers", type=int)
    parser.add_argument('b', help="nr of defenders", type=int)
    args = parser.parse_args()
    A = args.a 
    D = args.b
    print("{} A vs {} D = {}".format(A, D, calc_attacker_wins(A,D)))
