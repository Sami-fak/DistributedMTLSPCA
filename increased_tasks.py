#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 08:58:48 2021

@author: sami
"""
from fonctions import *

p = 100
m = 2

# t = [7]
t = list(range(2,10))
# t = [2**i for i in range(2,7)]
n_t_test = [[10000, 10000]]
nt = sum(n_t_test[0])

n_trial = 10

# np.random.seed(1)
emp_rate, th_rate, var, relative_error_rate = [], [], [], []
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
mean = mean_matrix(p, beta=betat[task_target], k=2)
M.append(mean[0])
# on garde moy pour la tâche target
moy = mean[1][:]
# on crée les données de test
X_test, y_test = gaussian_synthetic_data(nt, p, m, 1, n_t_test, [moy])
X_test_aggregated = aggregate_array(X_test, p, nt, 1, m)

theory_only = False

added=False
for idx,b in enumerate(t):
    if idx==0:
        boucle = b
    else:
        boucle = b-t[idx-1]
    print(f"boucle = {boucle}")
    # to_add correspond au nombre de data à ajouter en dehors de la tache target
    to_add = [2*50, 2*50]
    for i in range(boucle):
#         print(f"beta = {betat[beta]}")
        mean = mean_matrix(p, beta=betat[beta], k=1, starting=0)
        if i==task_target and not added:
            M.append(moy)
            # pour la tache target
            n_t.append([2*6,2*6])
            added=True
        else:
            n_t.append(to_add)
            M.append(mean[0])
        beta+=1
    err=[]
    print(b)
    n=0
    print(n_t)
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
            X, y_bs = gaussian_synthetic_data(n, p, m, b, n_t, M)
            MM = []
            diag = []
            
            # calcul des moyennes empiriques pour chaque client
            for i in range(b):
                MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]])
    #             print(f"task 1 empirical mean = {np.mean(MM1[0])}")
                MM.append(MM1)
                diag.append(diag1)
            
            # if display = True, affiche M'M, la matrice M gothique et les labels optimaux
            V, y, correlation_matrix, Dc, c0 = merging_center(MM, diag, b, m, p, n, n_t, task_target, display=False, normalization=False) 
            X_train_aggregated = aggregate_array(X, p, n, b, m)
            
            # ici le code pour comparer avec le non distribué
    #         MM = empirical_mean(b,m,X,p,n_t)
    #         print("MM Sami 1 : ")
    #         ns = [50, 50, 6, 6]+[50, 50]*(b-2)
    #         print(ns)
    #         ns=np.array(ns,dtype=int)
    #         MMM=np.zeros((2*b,2*b));
    #         for i in range(2*b):
    #             for j in range(2*b):
    #                 if i==j:
    #                     X_int1=X_train_aggregated[:,sum(ns[:i]):sum(ns[:i])+ns[i]//2];
    #                     X_int2=X_train_aggregated[:,sum(ns[:i])+ns[i]//2:sum(ns[:i+1])];
    #                     MMM[i,j]=4*np.ones((ns[i]//2,)).T@X_int1.T@X_int2@np.ones((ns[j]//2,))/(ns[i]**2);
    #                 else:
    #                     X1=X_train_aggregated[:,sum(ns[:i]):sum(ns[:i+1])];
    #                     X2=X_train_aggregated[:,sum(ns[:j]):sum(ns[:j+1])];
    #                     MMM[i,j]=np.ones((ns[i],)).T@X1.T@X2@np.ones((ns[j],))/(ns[i]*ns[j]);
    #         print("MM == MMM : ", MM==MMM)
    #         matprint(MM)
            
    #         y = label_evaluation(b, m, np.diag(c), MM,c0,1)
    #         correlation_matrix = compute_M_cal(n,p,np.diag(c),MM)
    #         X_test_aggregated = aggregate_array(X_test, p, nt, 1, m)
            
    #         V = compute_V_old(y, X_train_aggregated, J)
    #         V = np.reshape(V, p)
            m_t = create_mt(b, m, y, Dc, correlation_matrix, c0)
            V_true = compute_V_old(y_true, X_train_aggregated, J)
            V_true = np.reshape(V_true, p)
            
            # Remplacer V par V_true et m_t par m_t_true pour tracer l'erreur empirique avec les vraies moyennes
            erreur_empirique = compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0, 1, rho1, rho2, False, average=True)
            err.append(erreur_empirique)
#     print(MM_true)
    if not theory_only:
        x = np.linspace(-5,5, 500)
        plt.plot(x, norm.pdf(x, m_t_true[task_target][0], 1))
        plt.plot(x, norm.pdf(x, m_t_true[task_target][1], 1))
        plt.axvline(x=1/2*(m_t_true[task_target][0]+m_t_true[task_target][1]))
        plt.axvline(m_t_true[task_target][0], ls='--')
        plt.axvline(m_t_true[task_target][1], ls='--')
        debug_histogram(V_true, X_test_aggregated, n_t_test)
        x = np.linspace(-5,5, 500)
        plt.plot(x, norm.pdf(x, m_t[task_target][0], 1))
        plt.plot(x, norm.pdf(x, m_t[task_target][1], 1))
        plt.axvline(x=1/2*(m_t[task_target][0]+m_t[task_target][1]))
        plt.axvline(m_t[1][0], ls='--')
        plt.axvline(m_t[1][1], ls='--')
        debug_histogram(V, X_test_aggregated, n_t_test)
    
    erreur_th = optimal_rate(xx, rho1, rho2)
    # erreur_th = error_rate(b,m,Dc,MM_true,c0,1)[0][0]
    if not theory_only:
        emp_rate.append(np.mean(err))
        var.append(np.std(err))
    th_rate.append(erreur_th)


if not theory_only:
    lower = np.array(emp_rate) - np.array(var)
    upper = np.array(emp_rate) + np.array(var)
    plt.fill_between(t, lower, upper, alpha=0.2, label="variance")
    plt.plot(t, emp_rate, "-o", label='empirical rate')
plt.plot(t, th_rate, '-v', label='theoritical rate')
plt.xlabel("Nombre de tâches")
plt.ylabel("Taux d'erreur")
plt.title(r"Taux d'erreur empirique et théorique")
plt.legend()
plt.grid()
plt.show()