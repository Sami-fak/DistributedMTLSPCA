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
M = []
mean = mean_matrix(p, beta=betat[task_target], k=2, starting=0, constant=1)
M.append(mean[0])

t = list(range(2,4))
# t = [2**i for i in range(2,7)]
constant = [0.2, 0.6, 0.9]

# np.random.seed(0)
for beta_constant in constant:
    th_rate = []
    task_target = 1
    # betat = np.random.uniform(0,1,size=(t[-1]))
    print(beta_constant)
    betat = beta_constant*np.ones((t[-1]))
    beta = beta_constant
    X = []
    n_t = []
    n=0
    to_add = [multiple*50, multiple*50]
    n_t.append(to_add)
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
            # print("mean", mean)
            if i==task_target and not added:
                # pour la tache target
                n_t.append([multiple*6, multiple*6])
                added=True
            else:
                n_t.append(to_add)
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
        # MM_true = (1-beta**2)*np.identity(2*b)+beta**2*np.ones((2*b,1))@np.ones((2*b,1)).T
        
        
        # reprendre le calcul de MM_true, there might be something wronf with it
        if b==2:
            print(MM_true)
        erreur_th = error_rate(b,m,Dc, MM_true,c0)[0][0]
        th_rate.append(erreur_th)
    with open("log", "a") as log:
        log.write(f"beta = {beta}\n")
        for i, j in enumerate(th_rate):
            log.write(f"({i+2}, {j})")
        
        log.write("\n\n")
        
    plt.plot(t, th_rate, label=f'beta={beta_constant}')
plt.xlabel("Nombre de tâches")
plt.ylabel("Taux d'erreur")
plt.title(f"Taux d'erreur théorique p={p}, n={n}")
plt.legend()
plt.grid()
plt.show()