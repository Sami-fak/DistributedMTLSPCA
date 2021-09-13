#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 08:58:48 2021

@author: sami
"""
from fonctions import *
from time import time
t0=time()

multiple=2

p = 200
m = 2

# t = [7]
t = list(range(2,15))
# t = [2**i for i in range(2,7)]
n_t_test = [[1000, 1000]]
nt = sum(n_t_test[0])

n_trial = 10

# np.random.seed(0)
emp_rate, th_rate, var, relative_error_rate = [], [], [], []
emp_rate2 = []
task_target = 1
# betat = np.random.uniform(0,1,size=(t[-1]))
betat = 0.6*np.ones((t[-1]))
beta = 0
X = []
X_test = []
M = []
n_t = []
n=0
# print(f"beta = {betat[task_target]}")
mean = mean_matrix(p, beta=betat[task_target], k=2, starting=1, constant=1)
# print("mean : ", mean)
M.append(mean[0])
# print("M beggining", M)
to_add = [multiple*50, multiple*50]
n_t.append(to_add)
# on garde moy pour la tâche target
moy = mean[1][:]
# on crée les données de test
X_test, y_test = gaussian_synthetic_data(nt, p, m, 1, n_t_test, [moy], True)
X_test_aggregated = aggregate_array(X_test, p, nt, 1, m)

theory_only = False
log = False
added=False

for idx,b in enumerate(t):
    print(idx)
    if idx==0:
        boucle = b-1
    else:
        boucle = b-(t[idx-1])
    # print(f"boucle = {boucle}")
    # to_add correspond au nombre de data à ajouter en dehors de la tache target
    
    for i in range(1,boucle+1):
        # print("boucle ", i)
        print(f"beta = {betat[beta]}")
        mean = mean_matrix(p, beta=betat[beta], k=1, starting=0, constant=1)
        # print("mean", mean)
        if i==task_target and not added:
            M.append(moy)
            # pour la tache target
            n_t.append([multiple*20, multiple*20])
            added=True
        else:
            n_t.append(to_add)
            M.append(mean[0])
        beta+=1
    err=[]
    err2=[]
    # print(b)
    n=0
    # print(n_t)
    for i in range(len(n_t)):
        n += sum(n_t[i])
    print("n = ", n)
    # on crée les données synthétiques 
    c = estimate_c(n_t, n, b, m)
    c0 = p/n
    Dc = np.diag(c)
    # calcul des vraies moyennes et des labels optimaux
    M_true = true_mean(M, p, b, m)
    MM_true = M_true.T@M_true
    correlation_matrix_true = compute_M_cal(n, p, Dc, MM_true, display=False)
    y_true = label_evaluation(b,m,Dc,MM_true, c0, task_target)
    m_t_true = create_mt(b, m, y_true, Dc, correlation_matrix_true, c0)
    xx = m_t_true[task_target][1]-m_t_true[task_target][0]
    rho1 = n_t[1][0]/sum(n_t[1])
    rho2 = n_t[1][1]/sum(n_t[1])
    J = create_J(m, b, n, n_t)
    if not theory_only:
        for l in range(n_trial):
            X, y_bs = gaussian_synthetic_data(n, p, m, b, n_t, M, True)
            MM = []
            diag = []
            
            # calcul des moyennes empiriques pour chaque client
            for i in range(b):
                MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], halves=0)
    #             print(f"task 1 empirical mean = {np.mean(MM1[0])}")
                MM.append(MM1)
                diag.append(diag1)
            
            # if display = True, affiche M'M, la matrice M gothique et les labels optimaux
            V, y, correlation_matrix, Dc, c0, MM_gathered = merging_center(MM, diag, b, m, p, n, n_t, task_target, display=False) 
            X_train_aggregated = aggregate_array(X, p, n, b, m)
            m_t = create_mt(b, m, y, Dc, correlation_matrix, c0)
            V_true = compute_V_old(y_true, X_train_aggregated, J)
            V_true = np.reshape(V_true, p)
            
            # Remplacer V par V_true et m_t par m_t_true pour tracer l'erreur empirique avec les vraies moyennes
            erreur_empirique = compute_error_rate(X_test,V, m_t, m, n_t_test, Dc, c0, 1, rho1, rho2, False, average=False)
            erreur_empirique2 = compute_error_rate(X_test, V_true, m_t_true, m, n_t_test, Dc, c0, 1, rho1, rho2, False, average=False)
            err.append(erreur_empirique)
            err2.append(erreur_empirique2)
#     print(MM_true)
    if not theory_only:
        pass
        # x = np.linspace(-5,5, 500)
        # plt.plot(x, norm.pdf(x, m_t_true[task_target][0], 1),label=r'$\mathcal{N}(m_1,1)$')
        # plt.plot(x, norm.pdf(x, m_t_true[task_target][1], 1),label=r'$\mathcal{N}(m_2,1)$')
        # plt.axvline(x=1/2*(m_t_true[task_target][0]+m_t_true[task_target][1]))
        # plt.axvline(m_t_true[task_target][0], ls='--', color='g')
        # plt.axvline(m_t_true[task_target][1], ls='--', color='g')
        # debug_histogram(V_true, X_test_aggregated, n_t_test)
        # x = np.linspace(-5,5, 500)
        # tick = [-2, m_t[1][0],0, m_t[1][1],2]
        # labels = [-2, r'$\hat{m}_1$', 0, r'$\hat{m}_2$', 2]
        # plt.xticks(tick, labels)
        # plt.plot(x, norm.pdf(x, m_t[task_target][0], 1))
        # plt.plot(x, norm.pdf(x, m_t[task_target][1], 1))
        # plt.axvline(x=0)
        # plt.axvline(m_t[1][0], ls='--')
        # plt.axvline(m_t[1][1], ls='--')
        # debug_histogram(V, X_test_aggregated, n_t_test)

    erreur_th = optimal_rate(xx, rho1, rho2)
    # erreur_th2 = error_rate(b,m,Dc,MM_true,c0,1)[0][0]
    if not theory_only:
        emp_rate.append(np.mean(err))
        emp_rate2.append(np.mean(err2))
        var.append(np.std(err))
    th_rate.append(erreur_th)


if not theory_only:
    lower = np.array(emp_rate) - np.array(var)
    upper = np.array(emp_rate) + np.array(var)
    plt.fill_between(t, lower, upper, alpha=0.2, label="variance")
    plt.plot(t, emp_rate, "-o", label='empirical rate')
    plt.plot(t, emp_rate2)
    
if log:
    with open("log.txt", "a") as f:
        f.write(f"\nbeta = {betat[0]}\n")
        f.write("------------\n")
        for i in range(len(emp_rate)):
            f.write(f"({i+2}, {emp_rate[i]})")
        f.write("\n\n")
        for i in range(len(th_rate)):
            f.write(f"({i+2}, {th_rate[i]})")
        f.write("\n\n")
        for i in range(len(emp_rate)):
            f.write(f"({i+2}, {emp_rate[i]}) +- ({i+2}, {abs(upper[i]-emp_rate[i])})")
        f.write("\n------------\n")
    
plt.plot(t, th_rate, '-v', label='theoritical rate')
plt.xlabel("Nombre de tâches")
plt.ylabel("Taux d'erreur")
plt.title(f"Taux d'erreur empirique et théorique p={p}, n={n}, beta={betat[0]}")
plt.legend()
plt.grid()
plt.show()