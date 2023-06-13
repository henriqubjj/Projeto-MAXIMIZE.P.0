import numpy as np

class BranchAndBoundSolver:
    def __init__(self, model):
        self.model = model
        self.primal = 0
        self.optimal_model = None

    @staticmethod
    def solver(model):
        model.optimize()
        return {"objective": model.objective_value, "vars": model.vars}

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def bound(self, model):
        flag_solver = self.solver(model)
        count_int = 0
        if flag_solver["objective"] == None:
            return 'INVIABILIDADE'
        for i in flag_solver["vars"]:
            if i.x.is_integer():
                count_int += 1
        if count_int == len(flag_solver["vars"]):
            return 'INTEGRALIDADE'
        if flag_solver["objective"] <= self.primal:
            return 'LIMITE'
        return 'FRACIONÁRIO'

    def branch(self, model, values_solution):
        var_branch = values_solution[self.find_nearest([i.x for i in values_solution], 0.5)]
        model_0 = model.copy()
        model_0 += var_branch == 0
        model_1 = model.copy()
        model_1 += var_branch == 1
        return (model_0, model_1)

    def solve(self):
        nodes = [self.model]
        while nodes != []:
            model_solver = self.solver(nodes[0])
            flag_bound = self.bound(nodes[0])
            if flag_bound in {'INVIABILIDADE', 'LIMITE'}:
                nodes.pop(0)
            elif flag_bound == 'INTEGRALIDADE':
                if model_solver["objective"] > self.primal:
                    self.optimal_model = nodes[0]
                    self.primal = model_solver["objective"]
                nodes.pop(0)
            elif flag_bound == 'FRACIONÁRIO':
                flag_branch = self.branch(nodes[0], model_solver["vars"])
                nodes.append(flag_branch[0])
                nodes.append(flag_branch[1])
                nodes.pop(0)

        model_solved = self.solver(self.optimal_model)
        print("Variables:")
        for var in model_solved["vars"]:
            print(f'{var.name} = {var.x}')
        print("Objective function:")
        print(f'Z = {model_solved["objective"]}')
