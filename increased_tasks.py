#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 08:58:48 2021

@author: sami
"""
from fonctions import *
multiple=1
p = 100
m = 2

t = list(range(2,4))
n_t_test = [[1000, 1000]]
nt = sum(n_t_test[0])

n_trial = 1

emp_rate, th_rate, var, relative_error_rate = [], [], [], []
emp_rate2 = []
th_rate2 = []
task_target = 1
bc = 0.9
betat = bc*np.ones((t[-1]))
beta = 0
X = []
X_test = []
M = []
n_t = []
n=0
mean = mean_matrix(p, beta=betat[task_target], k=2, starting=0, constant=1)
M.append(mean[0])
to_add = [multiple*50, multiple*50]
n_t.append(to_add)
moy = mean[1][:]
X_test, y_test = gaussian_synthetic_data(nt, p, m, 1, n_t_test, [moy], True)
X_test_aggregated = aggregate_array(X_test, p, nt, 1, m)

log = False
added=False

for idx,b in enumerate(t):
    print(idx)
    if idx==0:
        boucle = b-1
    else:
        boucle = b-(t[idx-1])
    
    for i in range(1,boucle+1):
        print(f"beta = {betat[beta]}")
        mean = mean_matrix(p, beta=betat[beta], k=1, starting=0, constant=1)
        
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
    ones = np.ones(2*b-1)
    for i in range(len(ones)):
        if i%2==0:
            ones[i]-=bc**2
        else:
            ones[i]-=1
    
    diag_sup = np.diag(ones, 1)
    diag_inf = np.diag(ones, -1)
    MM_true = (1-bc**2)*np.identity(2*b)+bc**2*np.ones((2*b,1))@np.ones((2*b,1)).T + diag_inf + diag_sup
    
    neg=1
    for i in range(len(MM_true)):
        neg^=1
        for j in range(len(MM_true)):
            MM_true[i][j]*=(-1)**(neg)
            neg^=1
    if b==2:
        print(MM_true)
    c = estimate_c(n_t, n, b, m)
    c0 = p/n
    Dc = np.diag(c)
    
    erreur_th = error_rate(b,m,Dc, MM_true,c0)[0][0]
    th_rate.append(erreur_th)
    
    
    
    for l in range(n_trial):
        X, y_bs = gaussian_synthetic_data(n, p, m, b, n_t, M, True)
        MM = []
        diag = []
        
        for i in range(b):
            MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], halves=0)
            MM.append(MM1)
            diag.append(diag1)
            
        V, y, correlation_matrix, Dc, c0, MM_gathered = merging_center(MM, diag, b, m, p, n, n_t, task_target, display=False) 
        X_train_aggregated = aggregate_array(X, p, n, b, m)
        m_t = create_mt(b, m, y, Dc, correlation_matrix, c0)
        erreur_empirique = compute_error_rate(X_test,V, m_t, m, n_t_test, Dc, c0, 1, average=False)
        err.append(erreur_empirique)
        
    emp_rate.append(np.mean(err))
    var.append(np.std(err))
    # M_true = true_mean(M, p, b, m)
    # MM_true = M_true.T@M_true
    # th_rate2.append(error_rate(b,m,Dc, MM_true,c0)[0][0])

# lower = np.array(emp_rate) - np.array(var)
# upper = np.array(emp_rate) + np.array(var)
# plt.fill_between(t, lower, upper, alpha=0.2, label="variance")
# plt.plot(t, emp_rate, "-o", label='empirical rate')
    
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
    
plt.plot(t, th_rate, '-v', label='optimal rate')
# plt.plot(t, th_rate2, '-v', label='theoritical rate')
plt.xlabel("Nombre de tâches")
plt.ylabel("Taux d'erreur")
plt.title(f"Taux d'erreur empirique et théorique p={p}, n={n}, beta={betat[0]}")
plt.legend()
plt.grid()
plt.show()