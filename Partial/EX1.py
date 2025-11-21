from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model= DiscreteBayesianNetwork([("O","H"),("O","W"),("H","R"),("W","R"),("H","E"),("R","C")])


cpd_o= TabularCPD(variable='O',variable_card=2,values=[[0.7],[0.3]])

cpd_h= TabularCPD(variable='H',variable_card=2,values=[[0.8,0.1],[0.2,0.9]],evidence=['O'],evidence_card=[2])

cpd_w = TabularCPD(variable='W',variable_card=2,values=[[0.4,0.9],[0.6,0.1]],evidence=['O'],evidence_card=[2])

cpd_e = TabularCPD(variable='E',variable_card=2,values=[[0.8,0.2],[0.2,0.8]],evidence=['H'],evidence_card=[2])

cpd_r = TabularCPD(variable = 'R',variable_card=2,values=[[0.6, 0.9, 0.3, 0.5],[0.4, 0.1, 0.7, 0.5]],evidence=['H','W'],evidence_card=[2,2])

cpd_c = TabularCPD(variable= 'C',variable_card=2,values=[[0.15,0.6],[0.85,0.4]],evidence=['R'],evidence_card=[2])

model.add_cpds(cpd_o,cpd_h,cpd_w,cpd_e,cpd_r)
model.check_model()

print(model.get_independencies())

inference = VariableElimination(model)
print(model.check_model())
plt.figure(figsize=(10, 8))

plt.title("graph")
plt.show()
