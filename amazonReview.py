#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:18:31 2021

@author: sami
"""

from utils.fonctions import *

m=2
p=400
n11, n12, n21, n22 = 200, 2*100, 50, 90
var, emp_rate, emp_rate_naive, emp_rate_s, th_rate = [], [], [], [], []
k = 3
mat = loadmat("data/AmazonReview/books_400.mat")
mat_test = loadmat("data/AmazonReview/kitchen_400.mat")

X_test_aggregated, y_test = mat_test['fts'], mat_test['labels']
X_test_aggregated = preprocess(X_test_aggregated,p)
X_test, n_t_test = divide_array(X_test_aggregated, y_test, 1)
X1_aggregated, y = mat["fts"], mat["labels"]
X1_aggregated = preprocess(X1_aggregated, p)

X, n_t = divide_array(X1_aggregated, y, 1)
X = [[X[0][0].T[:n11].T, X[0][1].T[:n12].T]]
n_t = [[n11, n12]]
X.append([X_test[:][0][0].T[:n21].T, X_test[:][0][1].T[:n22].T])
n_t.append([n21, n22])
task_target = 1
    
for t in range(2, k+2):
    
    if t>2:
        if t==3:
            mat = loadmat("data/AmazonReview/dvd_400.mat")
            X2_aggregated, y2 = mat["fts"], mat["labels"] 
            X2_aggregated = preprocess(X2_aggregated, p)
            X_tmp, n_t_tmp = divide_array(X2_aggregated, y2, 1)
            X_tmp = [[X_tmp[0][0].T[:n11].T, X_tmp[0][1].T[:n12].T]]
            X.append(X_tmp[0][:])
            n_t.append([n11, n12])
        elif t==4:
            mat = loadmat("data/AmazonReview/elec_400.mat")
            X3_aggregated, y3 = mat["fts"], mat["labels"] 
            X3_aggregated = preprocess(X3_aggregated, p)
            X_tmp, n_t_tmp = divide_array(X3_aggregated, y3, 1)
            X_tmp = [[X_tmp[0][0].T[:n21].T, X_tmp[0][1].T[:n22].T]]
            X.append(X_tmp[0][:])
            n_t.append([n21, n22])
    n = sum(list(map(sum, (n_t[i] for i in range(t)))))
    print('n : ', n)
    
    MM = []
    diag = []
    for i in range(t):
        MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], halves=False)
        MM.append(MM1)
        diag.append(diag1)
        
    V, y_opt, correlation_matrix, Dc, c0, MM_gathered, y_n, V_naive, y_s, V_s = merging_center(MM, diag, t, m, p, n, n_t, task_target, naive=True, single=True)
    m_t = create_mt(t, m, y_opt, Dc, correlation_matrix, c0)
    m_t_naive = create_mt(t, m, y_n, Dc, correlation_matrix, c0)
    m_t_s = create_mt(t, m, y_s, Dc, correlation_matrix, c0)
    
    emp_rate.append(compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0, average=0))
    emp_rate_naive.append(compute_error_rate(X_test, V_naive, m_t_naive, m, n_t_test, Dc, c0, task_target=1, average=0))
    emp_rate_s.append(compute_error_rate(X_test, V_s, m_t_s, m, n_t_test, Dc, c0, task_target=1, average=0))

    
plt.plot(range(len(emp_rate)), emp_rate, '-o', label='MTL-SPCA with optimized labels')
plt.plot(range(len(emp_rate)), emp_rate_naive, '-x', label=r"N-SPCA with $y=\pm1$")
plt.plot(range(len(emp_rate)), emp_rate_s, '-o', label='ST-SPCA')
ticks = ["Books", "DVDs", "Electronics"]
plt.xticks(range(len(ticks)), ticks, size='larger')
plt.xlabel("Added tasks")
plt.ylabel("Empirical error rate")
plt.legend()
plt.grid()
plt.title("Real data")
plt.show()