import numpy as np
import textwrap
import argparse
import json
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

def init_states(A, D):
    State = namedtuple('state', ['attackers', 'defenders'])
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
    states = init_states(A, D)
    n = len(states)
    P = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            p = transition_prop(states[i], states[j])
            P[i,j] = p
    return P

def calc_F(A, D):
    P = calc_P(A, D)
    Q, R = P[:A*D, :A*D], P[:A*D:, A*D:]
    I = np.identity(Q.shape[0])
    inv = np.linalg.inv(I-Q)
    return np.dot(inv, R)

def calc_winning_prob(A, D):
    F = calc_F(A, D)
    WinProb = namedtuple('win_probility', ['attacker', 'defender'])
    pa, pd = 0,0
    for j in range(D, D+A):
        pa += F[A*D-1, j]
    for j in range(0, D):
        pd += F[A*D-1, j]
    
    return WinProb(pa,pd)

def dist_mean(dist):
    res = 0
    for i, v in enumerate(dist):
        res += i*v
    return res

def calc_loss(A, D):
    F = calc_F(A, D)
    ExpectedLoss = namedtuple('expected_loss', ['attacker', 'defender'])
    LossDist = namedtuple('loss_distribution', ['attacker', 'defender'])

    # probabilities of losing some
    al, dl = (A+1)*[0], (D+1)*[0] 
    for k in range(1, D+1):
        dl[D-k] = F[A*D-1, k-1]
    for k in range(1, A+1):
        al[A-k] = F[A*D-1, D+k - 1]

    # probability of losing all
    winning_prob = calc_winning_prob(A, D)
    al[A] = winning_prob.defender
    dl[D] = winning_prob.attacker
    eal = dist_mean(al)
    edl = dist_mean(dl)
    return ExpectedLoss(eal, edl), LossDist(al, dl)

def _print_loss_info(exp, dist, who):
    print("The {} will lose the following number of units:".format(who))
    for i, v in enumerate(dist):
        print("{:>2}: {:>3.1f}%".format(i, v*100))
    print("(the expected loss is {:.2f})".format(exp))
    
def _perc(c, r):
    return round(c*100*pow(10,r))/ pow(10,r)
    
def _round(c, r):
    return round(c*pow(10,r))/ pow(10,r)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', help="create matrix", action="store_true")
    parser.add_argument('attackers', help="(maximum) nr of attackers", type=int)
    parser.add_argument('defenders', help="(maximum) nr of defenders", type=int)
    args = parser.parse_args()
    if not args.full:
        A = args.attackers
        D = args.defenders
        txt = "{0} attackers have a {2:.1f}% winning chance vs {1} defenders"
        print(txt.format(A, D, calc_winning_prob(A,D).attacker*100))
        exploss, lossdist = calc_loss(A, D)
        _print_loss_info(exploss.attacker, lossdist.attacker, "attacker")
        _print_loss_info(exploss.defender, lossdist.defender, "defender")
    else:
        res = {}
        for A in range(1, args.attackers+1):        
            for D in range(1, args.defenders+1):
                print(A,D)
                wp = calc_winning_prob(A,D)
                exploss, lossdist = calc_loss(A, D)
                res["{}v{}".format(A,D)] = {"wp":(_perc(wp.attacker,1), _perc(wp.defender,1)),
                    "da": [_perc(x,1) for x in lossdist.attacker],
                    "dd": [_perc(x,1) for x in lossdist.defender],
                    "ea": _round(exploss.attacker,2),
                    "ed": _round(exploss.attacker,2)}
        with open('out.txt', 'w') as w:
            dmp = str(json.dumps(res))
            w.write(textwrap.fill(dmp,80))
        
                
                
