from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

email = DiscreteBayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])

cpd_s=TabularCPD(variable='S',variable_card=2,values=[[0.6],[0.4]])

cpd_o=TabularCPD(variable='O',variable_card=2,values=[[0.9,0.3],[0.1,0.7]],evidence=['S'],evidence_card=[2])

cpd_l=TabularCPD(variable='L',variable_card=2,values=[[0.7,0.2],[0.3,0.8]],evidence=['S'],evidence_card=[2])

cpd_m=TabularCPD(variable='M',variable_card=2,values=[[0.8,0.4,0.5,0.1],[0.2,0.6,0.5,0.9]],evidence=['S','L'],evidence_card=[2,2])


email.add_cpds(cpd_s,cpd_o,cpd_l,cpd_m)

email.check_model()

print(email.get_independencies())

infer=VariableElimination(email)

posterior_p=infer.query(['S'],evidence={'O':1,'L':1,'M':1})
print(posterior_p)