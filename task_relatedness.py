#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:45:25 2021

@author: sami
"""

from fonctions import *

p = 100
m = 2
t = 2

beta = np.linspace(0,1,5)
n_t = [[1000,1000], [50, 50]]
n_t_test = [[500, 500]]
nt = sum(n_t_test[0])
emp_rate, th_rate, var = [], [], []
emp_rate_n, var_n = [], []
emp_rate_s = []
th_rate2 = []
n_trial = 10

n=0
for i in range(len(n_t)):
    n+=sum(n_t[i])

ni = [sum(n_t[0]), sum(n_t[1])]

task_target = 1
for b in beta:
    err = []
    err_n = []
    err_s = []
    # on crée les données synthétiques
    M = mean_matrix(p, b, t, constant=1, starting=0)
    c = estimate_c(n_t, n, t, m)
    c0 = p/n
    Dc = np.diag(c)
    M_true = true_mean(M, p, t, m)
    MM_true = M_true.T@M_true
    print(MM_true, b**2)
    correlation_matrix_true = compute_M_cal(n, p, Dc, MM_true, display=False)
    y_true = label_evaluation(t,m,Dc,MM_true, c0)
    m_t_true = create_mt(t, m, y_true, Dc, correlation_matrix_true, c0)
    xx = m_t_true[1][1]-m_t_true[1][0]
    rho1 = n_t[1][0]/sum(n_t[1])
    rho2 = n_t[1][1]/sum(n_t[1])
    # erreur théorique optimale
    erreur_th = error_rate(t, m,  Dc, MM_true, c0)[0][0]
    th_rate2.append(erreur_th)
    th_rate.append(optimal_rate(xx, rho1, rho2))
    for l in range(n_trial):
        X, y_n = gaussian_synthetic_data(n, p, m, t, n_t, M)
        X_test, y_test = gaussian_synthetic_data(nt, p, m, 1, n_t_test, [M[task_target]])
        MM = []
        diag = []        
        # On calcule les moyennes empiriques sur les données locales
        # diag1 = [diag1[0], diag1[1]]
        for i in range(t):
            MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], 0)
            MM.append(MM1)
            diag.append(diag1)

    # CENTRAL SERVER
        # sending empirical means to central server
        # y est un vecteur de vecteurs de labels optimaux
        V, y, correlation_matrix, Dc, c0, MM_gathered, y_n, V_n, y_s, V_s = merging_center(MM, diag, t, m, p, n, n_t, task_target, naive=True, single=True)

    # END CENTRAL SERVER

        # Sending back optimal labels to clients
#         aggregated = []
#         for i in range(t):
#             aggregated.append(aggregate_array([X[i]], p, ni[i], 1, m))
        m_t = create_mt(t, m, y, Dc, correlation_matrix, c0)
        cor_mat_n = compute_M_cal(n,p,Dc,MM_gathered)
        m_t_n = create_mt(t, m, y_n, Dc, cor_mat_n, c0)
        m_t_s = create_mt(t, m, y_s, Dc, cor_mat_n, c0)
        J = create_J(2, t, n, n_t)
        err_n.append(compute_error_rate(X_test, V_n, m_t_n, m, n_t_test, Dc, c0, 1, rho1, rho2, False, average=False))
        err_s.append(compute_error_rate(X_test, V_s, m_t_s, m, n_t_test, Dc, c0, 1, rho1, rho2, False, average=False))
        X_test_aggregated = aggregate_array(X_test, p, nt, 1, m)
        erreur_empirique = compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0, 1, rho1, rho2, False, average=False)
        err.append(erreur_empirique)
#         epsilon1.append(eps1)
#         epsilon2.append(eps2)
        
    # x = np.linspace(-5,5, 500)
    # plt.plot(x, norm.pdf(x, m_t_true[1][0], 1), label=r"true m_{t1}")
    # plt.plot(x, norm.pdf(x, m_t_true[1][1], 1), label=r"true m_{t2}")
    # plt.plot(x, norm.pdf(x, m_t[1][0], 1), label=r"m_{t1}")
    # plt.plot(x, norm.pdf(x, m_t[1][1], 1), label=r"m_{t2}")    
    # debug_histogram(V, X_test_aggregated, n_t_test)
    emp_rate.append(np.mean(err))
    emp_rate_n.append(np.mean(err_n))
    emp_rate_s.append(np.mean(err_s))
#     e1.append(np.mean(epsilon1))
#     e2.append(np.mean(epsilon2))
    var.append(np.std(err))
    var_n.append(np.std(err_n))


lower = np.array(emp_rate) - np.array(var)
upper = np.array(emp_rate) + np.array(var)
# lower_n = np.array(emp_rate_n) - np.array(var_n)
# upper_n = np.array(emp_rate_n) + np.array(var_n)
plt.plot(beta, emp_rate, '-o', label='empirical rate')
plt.plot(beta, th_rate2, '-v', label='theoritical rate')
plt.plot(beta, emp_rate_n, '-^', label='N-SPCA')
plt.plot(beta, emp_rate_s, '-x', label='ST-SPCA')
plt.fill_between(beta, lower, upper, alpha=0.2, label="variance")
# plt.fill_between(beta, lower_n, upper_n, alpha=0.2)
plt.legend()
plt.title(f"2-class Gaussian mixture transfer error rate for n={n} and p={p}")
plt.xlabel("Task relatedness (beta)")
plt.ylabel("error rate")
plt.grid()
plt.show()