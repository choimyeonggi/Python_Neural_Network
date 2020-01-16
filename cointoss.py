import random


def coin_toss(n: int):
    if n > 0:
        n = int(n)
    else:
        raise ValueError('Invalid Input')
    coin = [1, 0]
    counter = 0
    for i in range(n):
        toss = random.choice(coin)
        if toss:
            counter += 1
    return counter / n


if __name__ == '__main__':
    print(coin_toss(100))
    