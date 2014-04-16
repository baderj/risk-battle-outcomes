import numpy as np
from collections import namedtuple
A = 3 # nr of attackers
D = 1 # nr of defenders

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

    if i == 1 and j == 1 and l == 0 and k == 0:
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
            p = round(p*10)/10
            P[i,j] = p
    return P

def slice_components(P):
    Components = namedtuple('Components', ['Q', 'R', 'I', 'Z'])
    return Components(P[:A*D, :A*D], P[:A*D:, A*D:], P[A*D:, A*D:], P[A*D:, :A*D])

def calc_F(Q, R):
    I = np.identity(Q.shape[0])

P = calc_P(A, D)
comp = slice_components(P)
F = calc_F(comp.Q, comp.R)


