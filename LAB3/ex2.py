from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


model = BayesianNetwork([
    ('Die', 'AddedBall'),
    ('AddedBall', 'DrawnBall')
])


cpd_die = TabularCPD(
    variable='Die',
    variable_card=6,
    values=[[1/6]*6],
    state_names={'Die': [1, 2, 3, 4, 5, 6]}
)


cpd_added = TabularCPD(
    variable='AddedBall',
    variable_card=3,
    values=[
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0]
    ],
    evidence=['Die'],
    evidence_card=[6],
    state_names={
        'AddedBall': ['Red', 'Blue', 'Black'],
        'Die': [1, 2, 3, 4, 5, 6]
    }
)


cpd_drawn = TabularCPD(
    variable='DrawnBall',
    variable_card=3,
    values=[
        [4/10, 3/10, 3/10],
        [4/10, 5/10, 4/10],
        [2/10, 2/10, 3/10]
    ],
    evidence=['AddedBall'],
    evidence_card=[3],
    state_names={
        'DrawnBall': ['Red', 'Blue', 'Black'],
        'AddedBall': ['Red', 'Blue', 'Black']
    }
)


model.add_cpds(cpd_die, cpd_added, cpd_drawn)


assert model.check_model()


infer = VariableElimination(model)
result = infer.query(variables=['DrawnBall'])


print(result)
print("\nP(DrawnBall = Red) =", result.values[0])