"""

RANDOM.

"""

import random


def coin_toss(n: int):
    n = int(n)
    coin = ['head', 'tail']
    coin_counter = 0
    for i in range(n):
        toss = random.choice(coin)
        if toss == 'head':
            coin_counter += 1
    return coin_counter / n


def dice_roll(n: int, dicesize=6):
    """
    THIS FUNCTION REQUIRES TO IMPORT 'random' BEFORE TO CALL.\n
    Rolls a dice with given trial n and make a list with each probability percentage. Initialized by 6(Hexagonal).\n
    Note that 'n and dicesize' will deprecise input into int type.
    and n(trials) must be greater than 1, dicesize 2.\n\n
    Use like dice_roll(10000), dice_roll(100000, dicesize=2), dice_roll(203222,5)\n
    :param n: Trials. must be positive integer. positive float will be automatically rescaled as int type.
    :param dicesize: Initialised by 6. greater than 2. Same as n.
    :return: a list of probability percentage sorted by dicesize 1, 2, ... input. For example, dice_roll(10000,2) -> [50.25, 49.75]
    """
    n = int(n)
    dicesize = int(dicesize)
    if n > 0 and dicesize > 1:  # function restrictions.
        dice = [i + 1 for i in range(dicesize)]  # make a dice eyes of 1, 2, ... n.
        counter_list = [0 for i in
                        range(len(dice))]  # make a counter for each eye 1, 2, ... n. (initiallised by [0, 0, ... 0]
        for i in range(n):
            roll = random.choice(dice)  # roll a dice.
            counter_list[roll - 1] += 1  # and reflects counter to the list.
    else:
        raise ValueError('invalid input')
    return [100 * counter_list[i] / n for i in range(len(dice))]


# [counter_list[i]/n for i in range(len(dice))]
# {i: counter_list[j]/n for i, j in zip(dice, counter_list)}

"""
Teacher's Solution

cases = []  # store result from cointoss
for _ range(t):
    case = [] # store each experiment.
    for _ range(n):
        random_coin = randcom.choice(type)
        case.append(random_coin)
    cases.append(tuple(case))
return cases


note that Counter function allows only if it is hashable(since lists can be modified anytime). In short, lists and dictionaries cannot be counted, tuples can.


Let H=1, T=0 be. then coin = [ 'H', 'T' ] -> [1,0].

coin_n = [1, 0]
cases = []
for i in range(trials):
    numberofhead = 0
    for j in range(2):
        if 1 == random.choice(coin_n):
            numberofhead += 1
    cases.append(numberofhead)
coin_event_counts = Counter(cases)
"""

if __name__ == '__main__':
    print(coin_toss(900000))
    print(dice_roll(900000, 6))
    print(dice_roll(10000, pow(2, 4)))

# roll 2 dices, probability of 8. (2 6), (3 5), (4 4), (5 3), (6 2) -> 5/36.


# roll 2 dices, probability of at least one is even (<-> both of them are odd). (1 1), (3 3), (5 5), (1 3), (3 1), (1 5), (5 1), (3 5), (5 3) -> 27/36





"""

"""