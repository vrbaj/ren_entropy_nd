import numpy as np
import padasip as pa
from scipy.stats import chi2, beta
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import fsolve
from scipy.optimize import fmin
import math

# data generation
experiment_len = 50000
inputs_number = 2
filter_len = 3
coeff = np.array([0.5, 0.5, 0.5])
parameter_change_idx = [10000, 15000, 20000, 23000,30000, 37000, 40000]
coeffs = [3.5 * coeff, 1.7 * coeff, 0.9 * coeff,1.5 * coeff,1.1 * coeff,1.3 * coeff,1.4 * coeff]
noise_coeff = 1
noise_std = 0.001

# input data vector
x = 2 * np.random.rand(experiment_len, inputs_number) - 1
# initialize filter data
desired_output = np.zeros([experiment_len, ])
filter_data = np.zeros([experiment_len, 3])
coeff_history = []
# experiment data

honu_filter = pa.filters.FilterGNGD(filter_len, mu=1, w=np.array(coeff))
honu_filter.w = np.array(coeff)
for idx in range(experiment_len):
    filter_data[idx, 0] = x[idx, 0]
    filter_data[idx, 1] = x[idx, 1]
    filter_data[idx, 2] = x[idx, 0] * x[idx, 1]
    for coeff_idx, change_idx in enumerate(parameter_change_idx):
        if idx > change_idx:
            coeff = coeffs[coeff_idx]
    coeff_history.append(coeff[0])
    desired_output[idx] = coeff[0] * x[idx, 0] + coeff[1] * x[idx, 1] + coeff[2] * x[idx, 0] * x[idx, 1] + noise_coeff * np.random.normal(0, noise_std, 1)




y, e, w = honu_filter.run(desired_output, filter_data)


dw = np.copy(w)
dw[1:] = np.diff(dw, n=1, axis=0)
dw_count = int(dw.shape[0])

# batch creation
batch_size = experiment_len
batches = {}
batch_n = 0
for k, weights in enumerate(dw):
    if k % batch_size == 0:
        batch_n += 1
        batches["batch_" + str(batch_n)] = []
    batches["batch_" + str(batch_n)].append(weights)

for batch in batches:
    batches[batch] = np.stack(batches[batch])


# avg weight in batch estimation
batch_t = {}
for batch in batches:
    batch_t[batch] = np.average(batches[batch], axis=0)
print("batch_t> ", batch_t)

# # correlation estimation
# C = {}
# for batch in batches:
#     C[batch] = np.corrcoef(batches[batch])
# print(C)
C = np.zeros((filter_len))
for item in dw:
    trans = np.array(item - batch_t["batch_1"], ndmin=2)

    no_trans = trans.T

    C = np.matmul(no_trans, trans) + C

C = 1 / (batch_size - 1) * C
print("C> ", C)

# decorrelation
xk = []
C_pom = C ** -0.5
for item in dw:
    xk.append(np.matmul(C_pom, np.array(item - batch_t["batch_1"], ndmin=2).T))
chi_k = []
beta_k = []
# alpha = filter_len / (filter_len + 2)
rho_k = []
for xk_item in xk:
    #chi_k.append([1 - chi2.cdf(np.linalg.norm(xk) ** 2, filter_len)])
    # beta_k.append(1 - beta.cdf(xk, filter_len / 2, 1/(1-alpha) - filter_len/2))
    rho_k.append(np.linalg.norm(xk_item))
rho = 1 / batch_size * np.sum(rho_k)
print("rho: ", rho)


# solve Er - rho = 0
alpha_history = []
p_value = []
initial_guess = filter_len / (filter_len + 2) + 0.00001
func = lambda alpha: (((2 * alpha / (1 - alpha) - filter_len) ** 0.5) * (gamma((filter_len + 1)/2) * gamma(1/(1 - alpha) - (filter_len + 1)/2)) / (gamma(filter_len / 2) * gamma(1/(1-alpha) - filter_len / 2)) - rho) ** 2
alpha_solution = fmin(func, initial_guess)
if alpha_solution[0] == initial_guess:
    alpha_solution[0] = 1

print("alpha solution: ", alpha_solution[0])
if filter_len / (filter_len + 2) < alpha_solution[0] < 1:
    print("Feasible alpha")
    s_star = (2 * alpha_solution[0] / (1 - alpha_solution[0]) - filter_len) ** 0.5
else:
    print("chi square")


for xk in batches["batch_1"]:
    if filter_len / (filter_len + 2) < alpha_solution[0] < 1:
        wk = (np.linalg.norm(xk) ** 2) / (np.linalg.norm(xk) ** 2 + s_star ** 2)
        # wk = np.linalg.norm(xk) ** 2
        p_value.append(1-beta.cdf(wk, filter_len / 2, 1/(1-alpha_solution) - filter_len/2))
    else:
        p_value.append(1 - chi2.cdf(np.linalg.norm(xk), filter_len))


plt.plot(p_value[1:])
plt.xlabel("k [-]")
plt.ylabel("pvalue")
print("pmin", min(p_value[1:]))
#plt.ylim([0, 1.1])

helpy_sum = sum(1 / i for i in range(1, batch_size + 1))
results = []
k_max = 0
try:

    p_value = np.concatenate(p_value).ravel()
except ValueError:
    pass
    print("Value error")

orig_p = np.copy(p_value)
new_p_value = np.copy(p_value)
new_p_value.sort()
for k, value in enumerate(new_p_value[1:]):
    if value <= (k + 1) * alpha_solution[0] / ((batch_size) * helpy_sum):
        k_max = k + 1
        #break
print("alpha fdr: ", new_p_value[k_max], " k*: ", k_max)
print("original index: ", np.where(orig_p == new_p_value[k_max+1])[0][0])
plt.figure()
plt.title("new_p_value")
plt.plot(new_p_value)
plt.figure()
plt.title("dw")
plt.plot(dw[20:])

plt.figure()
plt.plot(coeff_history)
plt.title("historyc")

plt.figure()
plt.title("y")
plt.plot(y)

plt.figure()
plt.title("beta")
rvs = beta.rvs(filter_len / 2, 1/(1-alpha_solution) - filter_len/2, size=1000000)
plt.hist(rvs, density=True, bins="auto")

plt.show()
