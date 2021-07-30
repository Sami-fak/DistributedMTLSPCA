#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:18:31 2021

@author: sami
"""
from fonctions import *

dataset = "amazon"
m=2
p=400
n11, n12, n21, n22 = 200, 2*100, 50, 90
var, emp_rate, th_rate = [], [], []
if dataset == "amazon":
    k = 3
    mat = loadmat("books_400.mat")
    mat_test = loadmat("kitchen_400.mat")

    X_test_aggregated, y_test = mat_test['fts'], mat_test['labels']
    print(y_test)
    X_test_aggregated = preprocess(X_test_aggregated,p)
    X_test, n_t_test = divide_array(X_test_aggregated, y_test, 1)
    # X1 contient toute la premiere tâche
    X1_aggregated, y = mat["fts"], mat["labels"]
    X1_aggregated = preprocess(X1_aggregated, p)
#     X1_aggregated = zscore(X1_aggregated, axis=None)
    
    X, n_t = divide_array(X1_aggregated, y, 1)
    X = [[X[0][0].T[:n11].T, X[0][1].T[:n12].T]]
    n_t = [[n11, n12]]
#     X = normalisation(X, p, True)
    X.append([X_test[:][0][0].T[:n21].T, X_test[:][0][1].T[:n22].T])
#     X_test = [[X_test[0][0].T[n21:n21+500].T, X_test[0][1].T[n22:n22+50].T]]
    print(X_test[0][0].shape, X_test[0][1].shape)
    print(n_t_test)
#     n_t_test = [[500, 500]]
    n_t.append([n21, n22])
    task_target = 1
    
for t in range(2, k+2):
#     nt = sum(n_t_test[0])
    # add tasks
    if t>2:
        if t==3:
            mat = loadmat("dvd_400.mat")
            X2_aggregated, y2 = mat["fts"], mat["labels"] 
#             X2_aggregated = zscore(X2_aggregated, axis=None)
            X2_aggregated = preprocess(X2_aggregated, p)
            X_tmp, n_t_tmp = divide_array(X2_aggregated, y2, 1)
            X_tmp = [[X_tmp[0][0].T[:n11].T, X_tmp[0][1].T[:n12].T]]
#             X_tmp = normalisation(X_tmp, p)
#             X_tmp = normalisation(X_tmp, p, True)
            X.append(X_tmp[0][:])
            n_t.append([n11, n12])
        elif t==4:
            mat = loadmat("elec_400.mat")
            X3_aggregated, y3 = mat["fts"], mat["labels"] 
#             X3_aggregated = zscore(X3_aggregated, axis=None)
            X3_aggregated = preprocess(X3_aggregated, p)
            X_tmp, n_t_tmp = divide_array(X3_aggregated, y3, 1)
            X_tmp = [[X_tmp[0][0].T[:n21].T, X_tmp[0][1].T[:n22].T]]
#             X_tmp = normalisation(X_tmp[:], p, True)
#             X_tmp = normalisation(X_tmp, p)
            X.append(X_tmp[0][:])
            n_t.append([n21, n22])
    print(n_t)
    n = sum(list(map(sum, (n_t[i] for i in range(t)))))
    print("n : ", n)
    MM = []
    diag = []
    for i in range(t):
        MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], halves=False)
        # chaque moyenne empirique calculée est un vecteur de taille p
#         sent+=1
#         t_MM.append(time()-t0)
        MM.append(MM1)
        diag.append(diag1)
    print(f"lenMM : {len(MM)}")
    # CENTRAL SERVER
#     t0 = time()
    # sending empirical means to central server
    V, y_opt, correlation_matrix, Dc, c0, MM_gathered = merging_center(MM, diag, t, m, p, n, n_t, task_target)
    matprint(Dc)
    # END CENTRAL SERVER
#     VTX = V.T@X_test_aggregated.T
#     var.append(np.var(VTX))
    m_t = create_mt(t, m, y_opt, Dc, correlation_matrix, c0)
    x = np.linspace(-5,5, 500)
    plt.plot(x, norm.pdf(x, m_t[1][0], 1))
    plt.xlim(-5,5)
    plt.axvline(x=m_t[1][0],ls='--')
    plt.axvline(x=m_t[1][1],ls='--')
    plt.plot(x, norm.pdf(x, m_t[1][1], 1))
    debug_histogram(V, X_test_aggregated.T, n_t_test)
    
#     erreur_theorique = error_rate(t, m,  Dc, MM_true, c0)[0][0]
    emp_rate.append(compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0, average=0))

print(emp_rate)
    
plt.plot(list(range(k)), emp_rate, '-o')
# lower = np.array(emp_rate) - np.array(var)
# upper = np.array(emp_rate) + np.array(var)
# plt.fill_between(list(range(k)), lower, upper, alpha=0.2, label="variance")
ticks = ["Books", "DVDs", "Electronics"]
plt.xticks(range(len(ticks)), ticks, size='larger')
plt.xlabel("Added tasks")
# plt.ylim(0.18,0.24)
plt.ylabel("Empirical error rate")
# plt.legend()
plt.grid()
plt.title("Real data")
plt.show()