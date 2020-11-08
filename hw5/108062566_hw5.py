import numpy as np

# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
from HomeworkFramework import Function

class CMA_ES_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally

        self.dim = self.f.dimension(target_func)
        self.lower = self.f.lower(target_func) * np.ones(self.dim)
        self.upper = self.f.upper(target_func) * np.ones(self.dim)
        self.target_func = target_func
        self.eval_times = 0
        self.generations = 0

        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        self.expected_length = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * np.square(self.dim)))

        self.set_parameters()

        self.hsig = 1
        self.y_weighted = np.zeros(self.dim)

    def set_parameters(self):
        self.path_sigma = np.zeros(self.dim)
        self.path_c = np.zeros(self.dim)

        self.B_matrix = np.identity(self.dim)
        self.diag = np.ones(self.dim)
        self.covariance_matrix = self.B_matrix @ np.diag(self.diag ** 2) @ self.B_matrix.T
        self.invsqrt_covariance_matrix = self.B_matrix @ np.diag(1 / self.diag) @ self.B_matrix.T

        self.mean = np.random.uniform(self.lower, self.upper, self.dim)
        self.mean_old = np.copy(self.mean)
        self.step_size = 0.3*np.max(self.upper-self.lower)

        self.offspring_size = 4 + int(np.floor(3 * np.log(self.dim)))
        self.parent_size = int(np.floor(self.offspring_size / 2))

        self.offsprings = np.full(self.offspring_size, None)
        self.objective_values = np.full(self.offspring_size, None)
        self.parents = np.full(self.parent_size, None)

        self.set_weights()
        self.mu_w = 1 / sum([np.square(i) for i in self.weights])
        self.alpha_cov = 2
        self.lr_sigma = (self.mu_w + 2) / (self.dim + self.mu_w + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_w - 1) / (self.dim + 1)) - 1) + self.lr_sigma
        self.lr_c = (4 + self.mu_w / self.dim) / (self.dim + 4 + 2 * self.mu_w / self.dim)
        self.lr_1 = self.alpha_cov / (np.square(self.dim + 1.3) + self.mu_w)
        self.lr_mu = min(1 - self.lr_1, self.alpha_cov * (
                    (self.mu_w - 2 + 1 / self.mu_w) / (np.square(self.dim + 2) + self.alpha_cov * self.mu_w / 2)))

    def set_weights(self):
        self.weights = [np.log((self.offspring_size + 1) / 2) - np.log(i) for i in range(1, self.parent_size + 1)]
        self.weights /= sum(self.weights)
        self.weights /= sum(self.weights)
        self.weights = np.array(self.weights)

    def sampling_and_select(self):
        for i in range(self.offspring_size):
            self.offsprings[i] = self.mean + self.step_size * self.B_matrix @ (self.diag * np.random.randn(self.dim))
            np.clip(self.offsprings[i], self.lower, self.upper, out=self.offsprings[i])
            #self.eval_times += 1

        #self.objective_values = self.target_func(self.offsprings)
        for i in range(self.offspring_size):
            #self.objective_values[i] = self.target_func(self.offsprings[i])
            objective_value = self.f.evaluate(self.target_func, self.offsprings[i])
            self.eval_times += 1
            if objective_value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
            self.objective_values[i] = objective_value
        indice = np.argsort(self.objective_values)
        if self.objective_values[indice[0]] < self.optimal_value:
            self.optimal_solution[:] = self.offsprings[indice[0]]
            self.optimal_value = self.objective_values[indice[0]]
        if objective_value == "ReachFunctionLimit":
            return 1
        self.parents[:] = self.offsprings[indice[:self.parent_size]]
        return 0

    def update_mean(self):
        self.mean_old[:] = self.mean
        m = np.zeros(self.dim)
        for i in range(len(self.parents)):
            m += self.parents[i] * self.weights[i]
        # np.add(m, self.parents[i]*self.weights[i], out=m, casting="unsafe")
        self.mean[:] = m

    def update_evolution_path(self):
        self.y_weighted = (self.mean - self.mean_old) / self.step_size

        self.path_sigma = (1 - self.lr_sigma) * self.path_sigma + np.sqrt(
            self.lr_sigma * (2 - self.lr_sigma) * self.mu_w) * \
                          self.invsqrt_covariance_matrix @ self.y_weighted

        if np.linalg.norm(self.path_sigma) / np.sqrt(1 - np.power((1 - self.lr_sigma),
                                                                  2 * (self.generations+1))) / self.expected_length < 1.4 + 2 / (
                self.dim + 1):
            self.hsig = 1
        else:
            self.hsig = 0
        self.path_c = (1 - self.lr_c) * self.path_c + self.hsig * np.sqrt(
            self.lr_c * (2 - self.lr_c) * self.mu_w) * self.y_weighted

    def update_covariance_matrix(self):
        x = np.empty((self.parent_size, self.dim))
        for i in range(len(x)):
            x[i] = self.parents[i]
        artmp = (1 / self.step_size) * (x - np.tile(self.mean_old, (self.parent_size, 1)))
        self.covariance_matrix = (1 - self.lr_1 - self.lr_mu) * self.covariance_matrix + self.lr_1 * (
                    np.outer(self.path_c, self.path_c) + (1 - self.hsig) * self.lr_c * (
                        2 - self.lr_c) * self.covariance_matrix) \
                                 + self.lr_mu * (artmp.T @ np.diag(self.weights) @ artmp)

    # print((self.covariance_matrix==self.covariance_matrix.T))

    def update_step_size(self):
        self.step_size = self.step_size * np.exp(
            self.lr_sigma / self.d_sigma * (np.linalg.norm(self.path_sigma) / self.expected_length - 1))
        print("--------------------step_size----------------------")
        print(self.step_size)

    def covariance_matrix_decomposition(self):
        self.covariance_matrix = np.triu(self.covariance_matrix) + np.triu(self.covariance_matrix, k=1).T
        # square_diag, self.B_matrix = np.linalg.eig(self.covariance_matrix)
        square_diag, self.B_matrix = np.linalg.eigh(self.covariance_matrix)
        self.diag = np.sqrt(square_diag)
        self.invsqrt_covariance_matrix = self.B_matrix @ np.diag(1 / self.diag) @ self.B_matrix.T

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        while self.eval_times < FES:
            print('======= CMA_ES Generation {', int(self.generations), '} =====================')
            print('==========================FE============================')
            print(self.eval_times)

            reach_limit = self.sampling_and_select()
            if reach_limit == 1:
                break
            self.update_mean()
            self.update_evolution_path()
            self.update_covariance_matrix()
            self.update_step_size()
            self.covariance_matrix_decomposition()

            print("optimal: {}\n".format(self.get_optimal()[1]))
            self.generations += 1

if __name__ == '__main__':
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # you should implement your optimizer
        op = CMA_ES_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1

