from pgmpy.models import MarkovNetwork
import networkx as nx
import matplotlib.pyplot as plt
import math
import itertools as it
model = MarkovNetwork()
model.add_nodes_from(['A1','A2','A3','A4','A5'])
model.add_edges_from([('A1','A2'),('A1','A3'),('A2','A4'),('A2','A5'),('A3','A4'),('A4','A5')])

pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()


cliques=nx.find_cliques(model)
for i,clique in enumerate(cliques):
    print(f"{i}. {clique}")

dic = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5}
def joint_distribution(clique,assigment):
    for v in clique:
        s=sum(dic[v]*assigment[v])
    return math.exp(s)

assigment = list(it.product([-1,1],repeat=5))
vars=['A1','A2','A3','A4','A5']

joint=[]
for a in assigment:
    n = dict(zip(vars, a))
    prod = 1
    for clique in cliques:
        prod *= joint_distribution(clique, n)
    joint.append((n, prod))

Z= sum(val for _,val in joint)
prob=[(asg,val/Z) for asg,val  in joint]

best = max(prob,key=lambda x: x[1])
print("cel mai bun e ",best[0])
print("prob este ",best[1])


