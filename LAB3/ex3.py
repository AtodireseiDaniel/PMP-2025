import numpy as np


def simulate_game():

    starter = np.random.choice([0, 1])


    n = np.random.randint(1, 7)

    num_flips = 2 * n

    if starter == 0:
        p_heads = 4 / 7
        m = np.random.binomial(num_flips, p_heads)

        winner = 0 if n >= m else 1
    else:
        p_heads = 0.5
        m = np.random.binomial(num_flips, p_heads)

        winner = 1 if n >= m else 0

    return winner


num_simulations = 10000
wins = {0: 0, 1: 0}

for _ in range(num_simulations):
    winner = simulate_game()
    wins[winner] += 1

print(f"Simulation Results ({num_simulations} games):")
print(f"Player P0 wins: {wins[0]} ({wins[0] / num_simulations:.2%})")
print(f"Player P1 wins: {wins[1]} ({wins[1] / num_simulations:.2%})")

if wins[1] > wins[0]:
    print("\nConclusion: Player P1 (the dishonest player) has a higher chance of winning.")
else:
    print("\nConclusion: Player P0 has a higher chance of winning.")