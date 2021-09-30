#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:46:46 2021

@author: sami
"""
#!/usr/bin/env python3

from utils.fonctions import *

multiple=1
p = 100
m = 2

t = list(range(2,100))
constant = [0.2,0.6,0.8,0.9]
task_target = 1

compare_with_bayes_optimal = False

for beta in constant:
    th_rate = []
    opt_rate = []
    R1=[]
    R2=[]
    print(beta)
    betat = beta*np.ones((t[-1]))
    n_t = []
    n=0
    to_add = [multiple*50, multiple*50]
    n_t.append(to_add)
    added=False
    for idx,b in enumerate(t):
        if idx==0:
            boucle = b-1
        else:
            boucle = b-(t[idx-1])
        
        for i in range(1,boucle+1):
            if i==task_target and not added:
                n_t.append([multiple*20, multiple*20])
                added=True
            else:
                n_t.append(to_add)
        n=0
        for i in range(len(n_t)):
            n += sum(n_t[i])
        print("n = ", n)
        
        c = estimate_c(n_t, n, b, m)
        c0 = p/n
        Dc = np.diag(c)
                
        ones = np.ones(2*b-1)
        for i in range(len(ones)):
            if i%2==0:
                ones[i]-=beta**2
            else:
                ones[i]-=1
            
        diag_sup = np.diag(ones, 1)
        diag_inf = np.diag(ones, -1)
        MM_true = (1-beta**2)*np.identity(2*b)+beta**2*np.ones((2*b,1))@np.ones((2*b,1)).T + diag_inf + diag_sup
        
        neg=1
        for i in range(len(MM_true)):
            neg^=1
            for j in range(len(MM_true)):
                MM_true[i][j]*=(-1)**(neg)
                neg^=1
        
        
        if b==2:
            print(MM_true)
        erreur_th = error_rate(b,m,Dc, MM_true,c0)[0][0]
        th_rate.append(erreur_th)
        
        if compare_with_bayes_optimal:
            r1, r2 = bayes_optimal(n_t, MM_true, p, b)
            R1.append(r1)
            R2.append(r2)
        
    plt.plot(t, th_rate, '-', label=f'beta={beta}')
    if compare_with_bayes_optimal:
        plt.plot(t, R2, linewidth=5, linestyle=((0, (1, 3))), color='red')
plt.xlabel("Tasks number")
plt.ylabel("Error rate")
plt.title(f"Theoretical rate p={p}, n={n}")
plt.legend()
plt.grid()
plt.show()
