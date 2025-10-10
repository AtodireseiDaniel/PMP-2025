import random


def functie():
    urna = ["RED","RED","RED","BLUE","BLUE","BLUE","BLUE","BLACK","BLACK"]
    zar = random.randint(1,6)

    if zar in [1,3,5]:
        urna.append("BLACK")
    elif zar == 6:
        urna.append("RED")
    else:
        urna.append("BLUE")
    random.shuffle(urna)
    return random.choice(urna)
count =0

n  = int(input())
for i in range(n):
    if functie() == "RED":
        count+=1
probabilitate = (2/6) * (3/10) + (1/6) * (4/10) + (1/2) * 3/10

estimated = count / n;
print(f"probabilitatea teoretica e {probabilitate:.3f} ,iar cea estimata {estimated:.3f}")
