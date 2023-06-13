import re

from mip import *
import numpy as np

class ModelBuilder:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vars = 0
        self.rests = 0
        self.list_of_coef_obj = []
        self.list_of_coef_rest = []

    def read_txt(self):
        with open(self.file_path) as file:
            values = [int(val) for line in file for val in re.findall(r'\d+', line)]
        self.vars, self.rests, *coefs = values
        self.list_of_coef_obj, *self.list_of_coef_rest = np.split(coefs, range(self.vars, len(coefs), self.vars+1))

    def create_model(self):
        self.read_txt()
        model = Model(sense=MAXIMIZE)
        x = [model.add_var(var_type="CONTINUOUS", lb=0, ub=1, name="x_" + str(i)) for i in range(self.vars + 1)]
        model.objective = xsum(self.list_of_coef_obj[i]*x[i] for i in range(self.vars))
        for i in range(self.rests):
            model += xsum(self.list_of_coef_rest[i][j]*x[j] for j in range(self.vars)) <= self.list_of_coef_rest[i][-1]
        return model
