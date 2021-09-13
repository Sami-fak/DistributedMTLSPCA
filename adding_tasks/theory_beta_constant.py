#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:46:46 2021

@author: sami
"""
#!/usr/bin/env python3
from fonctions import *

multiple=1

p = 100
m = 2

t = list(range(2,15))
# t = [2**i for i in range(2,7)]
constant = list(range(0,12,2))

# np.random.seed(0)
for beta_constant in constant:
    th_rate = []
    task_target = 1
    # betat = np.random.uniform(0,1,size=(t[-1]))
    print(beta_constant)
    betat = beta_constant/10*np.ones((t[-1]))
    beta = 0
    X = []
    M = []
    n_t = []
    n=0
    mean = mean_matrix(p, beta=betat[task_target], k=2, starting=1, constant=1)
    M.append(mean[0])
    to_add = [multiple*50, multiple*50]
    n_t.append(to_add)
    # on garde moy pour la tâche target
    moy = mean[1][:]
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
        print("xx = ", xx)
        rho1 = n_t[1][0]/sum(n_t[1])
        rho2 = n_t[1][1]/sum(n_t[1])
        # erreur_th = optimal_rate(xx, rho1, rho2)
        erreur_th = error_rate(b,m,Dc, MM_true,c0)[0][0]
        th_rate.append(erreur_th)
        
    plt.plot(t, th_rate, label=f'beta={beta_constant/10}')
plt.xlabel("Nombre de tâches")
plt.ylabel("Taux d'erreur")
plt.title(f"Taux d'erreur théorique p={p}, n={n}")
plt.legend()
plt.grid()
plt.show()