#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:19:44 2021

@author: sami
"""

from fonctions import *

p=800
m=2
n_d, n_w, n_a, n_c = 157, 295, 958, 1123
dslr = loadmat("dslr_surf_10.mat")
webcam = loadmat("webcam_surf_10.mat")
amazon = loadmat("amazon_surf_10.mat")
caltech = loadmat("caltech_surf_10.mat")
dslr_fts, y_d = dslr['feas'], dslr['label']
webcam_fts, y_w = webcam['feas'], webcam['label']
amazon_fts, y_a = amazon['feas'], amazon['label']
caltech_fts, y_c = caltech['feas'], caltech['label']

y_d = np.reshape(y_d, n_d)
y_w = np.reshape(y_w, n_w)
y_a = np.reshape(y_a, n_a)
y_c = np.reshape(y_c, n_c)

n11, n12, n21, n22 = 2*100, 2*100, 50, 50
var, emp_rate, th_rate = [], [], []

n_t_test = [[100,90]]

X_test_aggregated, y_test = np.concatenate((caltech_fts[(y_c==1)][20:120],caltech_fts[(y_c==2)][20:110])), np.concatenate((y_c[:6],y_c[12:18]))
X_test_aggregated = preprocess(X_test_aggregated,p)
X_test = [
    [(X_test_aggregated[:n_t_test[0][0]]).T,
     (X_test_aggregated[n_t_test[0][0]:n_t_test[0][0]+n_t_test[0][1]]).T]
]

X1_aggregated = np.concatenate((amazon_fts[(y_a==1)][:50],amazon_fts[(y_a==2)][:50]))
X1_aggregated = preprocess(X1_aggregated,p)
X = [
    [(X1_aggregated[:50]).T, (X1_aggregated[:50]).T],
    [(caltech_fts[(y_c==1)][:20]).T, (caltech_fts[(y_c==2)][:20]).T]
]

X2 = [
    [(webcam_fts[(y_w==1)][:20]).T, (webcam_fts[(y_w==2)][:20]).T]
]

X3 = [
    [(dslr_fts[(y_d==1)][:6]).T, (dslr_fts[(y_d==2)][:6]).T]
]

n_t = [[50, 50], [20,20]]
task_target = 1

# X1_aggregated, y = mat["fts"], mat["labels"]
# X1_aggregated = preprocess(X1_aggregated, p)

# X, n_t = divide_array(X1_aggregated, y, 1)
# X = [[X[0][0].T[:n11].T, X[0][1].T[:n12].T]]
# n_t = [[n11, n12]]
# X.append([X_test[:][0][0].T[:n21].T, X_test[:][0][1].T[:n22].T])
# print(X_test[0][0].shape, X_test[0][1].shape)
# print(n_t_test)
# n_t.append([n21, n22])



for t in range(2,5):
    MM = []
    diag = []
    print(t)
    if t==3:
        X.append(X2[0])
        n_t.append([20,20])
    if t==4:
        X.append(X3[0])
        n_t.append([6,6])
    n=0
    for i in range(len(n_t)):
        n+=sum(n_t[i])
    print("n : ", n)
    for i in range(t):
        MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]])
        # chaque moyenne empirique calculée est un vecteur de taille p
    #         sent+=1
    #         t_MM.append(time()-t0)
        MM.append(MM1)
        diag.append(diag1)
    # CENTRAL SERVER
    #     t0 = time()
    # sending empirical means to central server
    V, y_opt, correlation_matrix, Dc, c0 = merging_center(MM, diag, t, m, p, n, n_t, task_target)
    matprint(Dc)
    # END CENTRAL SERVER
    #     VTX = V.T@X_test_aggregated.T
    #     var.append(np.var(VTX))
    m_t = create_mt(t, m, y_opt, Dc, correlation_matrix, c0)
    x = np.linspace(-10,10, 500)
    plt.plot(x, norm.pdf(x, m_t[1][0], 1))
    plt.xlim(-10,10)
    plt.axvline(x=m_t[1][0],ls='--')
    plt.axvline(x=m_t[1][1],ls='--')
    plt.plot(x, norm.pdf(x, m_t[1][1], 1))
    debug_histogram(V, X_test_aggregated.T, n_t_test)

#     erreur_theorique = error_rate(t, m,  Dc, MM_true, c0)[0][0]
    emp_rate.append(compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0))
    print(emp_rate)  

plt.plot(range(len(emp_rate)), emp_rate, '-o')
# lower = np.array(emp_rate) - np.array(var)
# upper = np.array(emp_rate) + np.array(var)
# plt.fill_between(list(range(k)), lower, upper, alpha=0.2, label="variance")
ticks = ["Amazon", "Webcam", "DSLR"]
plt.xticks(range(len(ticks)), ticks, size='larger')
plt.xlabel("Added tasks")
# plt.ylim(0.18,0.24)
plt.ylabel("Empirical error rate")
# plt.legend()
plt.grid()
plt.title("Office Caltech SURF Data")
plt.show()