#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:45:25 2021

@author: sami
"""

from utils.fonctions import *

p = 100
m = 2
t = 2

beta = np.linspace(0,1,10)
n_t = [[1000,1000], [50, 50]]
n_t_test = [[5000, 5000]]
nt = sum(n_t_test[0])
emp_rate, th_rate, var = [], [], []
emp_rate_n, var_n = [], []
emp_rate_s = []
th_rate2 = []
n_trial = 1

n=0
for i in range(len(n_t)):
    n+=sum(n_t[i])

ni = [sum(n_t[0]), sum(n_t[1])]

task_target = 1
for b in beta:
    err = []
    err_n = []
    err_s = []
    
    M = mean_matrix(p, b, t, constant=1, starting=0)
    c = estimate_c(n_t, n, t, m)
    c0 = p/n
    Dc = np.diag(c)
    M_true = true_mean(M, p, t, m)
    MM_true = M_true.T@M_true
    correlation_matrix_true = compute_M_cal(n, p, Dc, MM_true, display=False)
    y_true = label_evaluation(t,m,Dc,MM_true, c0)
    m_t_true = create_mt(t, m, y_true, Dc, correlation_matrix_true, c0)
    
    erreur_th = error_rate(t, m,  Dc, MM_true, c0)[0][0]
    th_rate2.append(erreur_th)
    
    for l in range(n_trial):
        X, y_n = gaussian_synthetic_data(n, p, m, t, n_t, M)
        X_test, y_test = gaussian_synthetic_data(nt, p, m, 1, n_t_test, [M[task_target]])
        MM = []
        diag = []        
        
        for i in range(t):
            MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], halves=0)
            MM.append(MM1)
            diag.append(diag1)
            
        V, y, correlation_matrix, Dc, c0, MM_gathered, y_n, V_n, y_s, V_s = merging_center(MM, diag, t, m, p, n, n_t, task_target, naive=True, single=True)


        m_t = create_mt(t, m, y, Dc, correlation_matrix, c0)
        cor_mat_n = compute_M_cal(n,p,Dc,MM_gathered)
        cor_mat_s = compute_M_cal(n,p,Dc,MM_gathered)
        m_t_n = create_mt(t, m, y_n, Dc, cor_mat_n, c0)
        m_t_s = create_mt(t, m, y_s, Dc, cor_mat_s, c0)
        J = create_J(2, t, n, n_t)
        err_n.append(compute_error_rate(X_test, V_n, m_t_n, m, n_t_test, Dc, c0, 1, average=False))
        err_s.append(compute_error_rate(X_test, V_s, m_t_s, m, n_t_test, Dc, c0, 1, average=False))
        X_test_aggregated = aggregate_array(X_test, p, nt, 1, m)
        erreur_empirique = compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0, 1, average=False)
        err.append(erreur_empirique)
        
    emp_rate.append(np.mean(err))
    emp_rate_n.append(np.mean(err_n))
    emp_rate_s.append(np.mean(err_s))
    
    var.append(np.std(err))
    var_n.append(np.std(err_n))

# with open("log_task", "a") as log:
#     log.write(f"n = {n}, p ={p}\n")
#     log.write(f"n_trial = {n_trial}\n")
#     log.write("\nEmp rate MTL-SPCA: \n")
#     for i in range(len(emp_rate)):
#         log.write(f"({beta[i]}, {emp_rate[i]})")
#     log.write("\nEmp rate N-SPCA: \n")
#     for i in range(len(emp_rate)):
#         log.write(f"({beta[i]}, {emp_rate_n[i]})")
#     log.write("\nEmp rate ST-SPCA: \n")
#     for i in range(len(emp_rate)):
#         log.write(f"({beta[i]}, {emp_rate_s[i]})")
#     log.write("\nVariance: \n")
#     for i in range(len(emp_rate)):
#         log.write(f"({beta[i]}, {emp_rate[i]}) +- ({beta[i]},  {var[i]})")
#     log.write("\nTh Rate: \n")
#     for i in range(len(th_rate2)):
#         log.write(f"({beta[i]}, {th_rate2[i]})")


lower = np.array(emp_rate) - np.array(var)
upper = np.array(emp_rate) + np.array(var)

plt.plot(beta, emp_rate, '-o', label='empirical rate')
plt.plot(beta, th_rate2, '-v', label='theoritical rate')
plt.plot(beta, emp_rate_n, '-^', label='N-SPCA')
plt.plot(beta, emp_rate_s, '-x', label='ST-SPCA')
plt.fill_between(beta, lower, upper, alpha=0.2, label="variance")

plt.legend()
plt.title(f"2-class Gaussian mixture transfer error rate for n={n} and p={p}")
plt.xlabel("Task relatedness (beta)")
plt.ylabel("error rate")
plt.grid()
plt.show()