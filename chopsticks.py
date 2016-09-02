import numpy as np
import random

indexHash = {}
partitionsHash = {}


class State:
    def __init__(p1, p2):
        self.state = (p1, p2)
        self.transitions = self.get_possible_transitions(p1, p2, True)
        self.indexHash = {x:y for x, y in zip(self.transitions, range(len(self.transitions)))}
        self.reward_matrix = np.zeros(1, len(self.transitions))
        self.reward_matrix[:] = np.nan

    def __str__(self):
        return self.state

def two_partitions(n):
    '''
    Get possible partitions of total 
    fingers on each hand
    '''
    if n <= 4:
        a, b = 0, n
    else:
        a, b = n-4, 4
    return zip(range(a, b+1), range(b, a-1, -1))

def get_indices():
    '''
    Get matrix indices hashed to partitions
    '''
    for i in xrange(9):
        partitions = two_partitions(i)
        partitionsHash[i] = partitions

def get_matrix():
    '''
    Compute all possible from states and to states 
    Construct a matrix of 576 x 624 (all NaN)
    Make a hash table of states to matrix indices
    '''
    from_states = {}
    to_states = {}
    from_index = 0
    to_index = 0
    for i in xrange(9):
        p1_partitions = two_partitions(i)
        for j in xrange(9):
            p2_partitions = two_partitions(j)
            for p1 in p1_partitions:
                for p2 in p2_partitions:
                    if (any(p2) ^ any(p1)):
                        to_states[(p1, p2)] = to_index
                        to_index += 1
                    if any(p2) and any(p1):
                        from_states[(p1, p2)] = from_index
                        to_states[(p1, p2)] = to_index
                        to_index += 1
                        from_index += 1
    state_matrix = np.zeros((len(from_states), len(to_states)))
    state_matrix[:] = np.nan
    return from_states, to_states, state_matrix

def get_possible_transitions(p1, p2, turn):
    '''
    Generate all possible transitions given a state
    1)Get hits from p1 to p2
    2)Get split states from p1
    3)Get self hits from p1
    4)Swap depending on turn 
    '''
    transitions = []
    if not any(p1) or not any(p2):
        return transitions
    if not turn:
        p1, p2 = p2, p1
    a,b,c,d = p1[0], p1[1], p2[0], p2[1]
    if a and d:
        transitions.append([p1, (c, a+d)]) if a+d < 5 else transitions.append([p1, (c, 0)])
    if b and d:
        transitions.append([p1, (c, b+d)]) if b+d < 5 else transitions.append([p1, (c, 0)])
    if a and c:
        transitions.append([p1, (a+c, d)]) if a+c < 5 else transitions.append([p1, (0, d)])
    if b and c:
        transitions.append([p1, (b+c, d)]) if b+c < 5 else transitions.append([p1, (0, d)])
    if p1.__contains__(0):
        if sum(p1) % 2 == 0:
            for partition in partitionsHash[sum(p1)]:
                if not partition.__contains__(0):
                    transitions.append([partition, p2])
    #else:
    #    for partition in partitionsHash[sum(p1)]:
    #        if not partition.__contains__(0) and partition != p1:
    #            transitions.append([partition, p2])
        #transitions.append([(a+b, b), p2]) if a+b < 5 else transitions.append([(0, b), p2])
        #transitions.append([(a, a+b), p2]) if a+b < 5 else transitions.append([(a, 0), p2])
    if not turn:
        for transition in transitions:
            transition[0], transition[1] = transition[1], transition[0]
    transitions = [tuple(transition) for transition in transitions]
    return list(set(transitions))


def play_game_better(from_states, to_states, state_matrix, turn=True):
    game = []
    p1 = (1, 1)
    p2 = (1, 1)
    game.append((p1, p2))
    while True:
        transition = get_best_move((p1, p2), from_states, to_states, state_matrix, turn)
        if transition:
            game.append(transition)
        else:
            transitions = get_possible_transitions(p1, p2, turn)
            transition = transitions[random.randint(0, (len(transitions) - 1))]
            game.append(transition)
        if len(game) > 30:
            return []
        turn = not turn
        p1, p2 = transition
        if not any(p1) or not any(p2):
            break
    return game

def play_game(turn=True):
    '''
    Start with [(1, 1), (1, 1)]
    Generate all possible transitions 
    Choose one at random 
    Repeat till game over
    '''
    game = []
    p1 = (1, 1)
    p2 = (1, 1)
    game.append((p1, p2))
    while True:
        transitions = get_possible_transitions(p1, p2, turn)
        transition = transitions[random.randint(0, (len(transitions) - 1))]
        game.append(transition)
        if len(game) > 30:
            return []
        turn = not turn
        p1, p2 = transition
        if not any(p1) or not any(p2):
            break
    return game

def condition_matrix(from_states, to_states, state_matrix):
    for state in from_states.keys():
        for transition in get_possible_transitions(state[0], state[1], True):
            if not any(transition[1]):
                state_matrix[from_states[state], to_states[transition]] = 1000000
            if not any(transition[0]):
                state_matrix[from_states[state], to_states[transition]] = -1000000
        for transition in get_possible_transitions(state[0], state[1], False):
            if not any(transition[1]):
                state_matrix[from_states[state], to_states[transition]] = -1000000
            if not any(transition[0]):
                state_matrix[from_states[state], to_states[transition]] = 1000000                  
    return state_matrix

def get_best_move(state, from_states, to_states, state_matrix, turn=True):
    '''
    Find out possible transitions
    Get the maximum reward for those
    return the best transition
    '''
    if not turn:
        state = (state[1], state[0])
    transitions = get_possible_transitions(state[0], state[1], True)
    indices = [to_states[i] for i in transitions]
    try:
        best_index = np.nanargmax(state_matrix[from_states[state]][indices])
    except:
        return None
    best_trans = None
    for trans in transitions:
        if to_states[trans] == indices[best_index]:
            best_trans = trans
    if not turn:
        best_trans = (best_trans[1], best_trans[0])
    return best_trans

def train_better(num_tries=10000):
    from_states, to_states, state_matrix = train(10000)
    get_indices() 
    win=1 
    reward = 50000
    discount = 0.8
    init_learning_rate = 0.9
    learning_decay = 1 
    learning_decay_step = 0.6/(num_tries)
    for i in xrange(num_tries):
        game = play_game_better(from_states, to_states, state_matrix, True)
        learning_rate = init_learning_rate/learning_decay  
        learning_decay += learning_decay_step
        if len(game) % 2 == 0:
            win = 1
        else:
            win = -1
        for index in range(0, len(game) - 1):
            row, col = from_states[game[index]], to_states[game[index+1]]
            nextQ = np.nan
            if index != len(game) - 2:
                if win == 1:
                    nextQ = np.nanmax(state_matrix[from_states[game[index+1]]])
                else:
                    nextQ = -1 * np.nanmax(state_matrix[from_states[game[index+1]]]) 
            else:
                if win == 1:
                    nextQ = 1000000
                else:
                    nextQ = -1000000               
            update_value = win*learning_rate*reward
            if not np.isnan(nextQ):
                update_value += update_value + discount*learning_rate*nextQ
            if not np.isnan(state_matrix[row, col]):
                update_value += (1 - learning_rate) * state_matrix[row, col]
            #print game[index], game[index+1], win, learning_rate, discount, nextQ,  reward, state_matrix[row][col], update_value
            state_matrix[row][col] = update_value
            win *= -1
        #print " "
    return from_states, to_states, state_matrix 

def train(num_tries=10000):
    '''
    Train with random games 
    p1 moves first for now 
    if win:
        update score positively for ever even pair 
        update score negatively for every odd pair
    if loss:
        do the opposite
    decrease learning rate and increase discount with time
    '''
    get_indices()
    win=1
    from_states, to_states, state_matrix = get_matrix()
    reward = 5000
    discount = 0.8
    init_learning_rate = 0.9
    state_matrix = condition_matrix(from_states, to_states, state_matrix)
    learning_decay = 1 
    learning_decay_step = 1/(num_tries)
    game_func = play_game
    for i in xrange(num_tries):
        game = play_game()
        learning_rate = init_learning_rate/learning_decay  
        learning_decay += learning_decay_step
        if len(game) % 2 == 0:
            win = 1
        else:
            win = -1
        for index in range(0, len(game) - 1):
            row, col = from_states[game[index]], to_states[game[index+1]]
            nextQ = np.nan
            if index != len(game) - 2:
                if win == 1:
                    nextQ = np.nanmax(state_matrix[from_states[game[index+1]]])
                else:
                    nextQ = -1 * np.nanmax(state_matrix[from_states[game[index+1]]]) 
            else:
                if win == 1:
                    nextQ = 1000000
                else:
                    nextQ = -1000000                 
            update_value = win*learning_rate*reward/(len(game) - index)
            if not np.isnan(nextQ):
                update_value += update_value + discount*learning_rate*nextQ
            if not np.isnan(state_matrix[row, col]):
                update_value += (1 - learning_rate) * state_matrix[row, col]
            #print game[index], game[index+1], win, learning_rate, discount, nextQ,  reward, state_matrix[row][col], update_value
            state_matrix[row][col] = update_value
            win *= -1
        #print " "
    return from_states, to_states, state_matrix