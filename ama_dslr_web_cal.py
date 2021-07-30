#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:00:55 2021

Amazon-Caltech-DSLR-Webcam database

@author: sami
"""

from fonctions import *
from sklearn.decomposition import PCA

pca=False
shuffle=True

m=2
if pca:
    p=150
else:
    p=800
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
c1,c2=1,2

if pca:
    dslr_fts = PCA(n_components=150).fit(dslr_fts).transform(dslr_fts)
    webcam_fts = PCA(n_components=150).fit(webcam_fts).transform(webcam_fts)
    amazon_fts = PCA(n_components=150).fit(amazon_fts).transform(amazon_fts)
    caltech_fts = PCA(n_components=150).fit(caltech_fts).transform(caltech_fts)

X11 = caltech_fts[(y_c==c1)]; X12=caltech_fts[(y_c==c2)]
X21 = dslr_fts[(y_d==c1)];X22 = dslr_fts[(y_d==c2)]
X31 = webcam_fts[(y_w==c1)];X32 = webcam_fts[(y_w==c2)]
X41 = amazon_fts[(y_a==c1)]; X42 =amazon_fts[(y_a==c2)]

test1= X11[50:150][:];test2=X12[60:110][:]

if shuffle:
    #shuffling classes
    np.random.shuffle(X11[:50])
    np.random.shuffle(X21)
    np.random.shuffle(X31)
    np.random.shuffle(X41)
    
    np.random.shuffle(X12[:60])
    np.random.shuffle(X22)
    np.random.shuffle(X32)
    np.random.shuffle(X42)
    

var, emp_rate, emp_rate_naive, emp_rate_s, th_rate = [], [], [], [],[]

n_t_test = [[100,50]]

X_test_aggregated = np.concatenate((test1,test2))
print(X_test_aggregated.shape)
X_test_aggregated = preprocess(X_test_aggregated, p, True, 0, minmax=True, norm=0)
X_test = [
    [X_test_aggregated[:n_t_test[0][0]].T,
     X_test_aggregated[n_t_test[0][0]:].T]
]

X1_aggregated = np.concatenate((X11[:50],X12[:60]))
X1_aggregated = preprocess(X1_aggregated, p, True, 0, minmax=True, norm=0)

# X2_aggregated = np.concatenate((caltech_fts[(y_c==c1)][:50],caltech_fts[(y_c==c2)][:60]))
# X2_aggregated = preprocess(X2_aggregated, p, True, 0,1)
X = [
    [X1_aggregated[:50].T, X1_aggregated[50:].T]
]

#,
#[X2_aggregated[:50].T, X2_aggregated[50:].T]

n_t = [[50, 60]]
task_target = 0

# X1_aggregated, y = mat["fts"], mat["labels"]
# X1_aggregated = preprocess(X1_aggregated, p)

# X, n_t = divide_array(X1_aggregated, y, 1)
# X = [[X[0][0].T[:n11].T, X[0][1].T[:n12].T]]
# n_t = [[n11, n12]]
# X.append([X_test[:][0][0].T[:n21].T, X_test[:][0][1].T[:n22].T])
# print(X_test[0][0].shape, X_test[0][1].shape)
# print(n_t_test)
# n_t.append([n21, n22])
t=2
n = sum(list(map(sum, (n_t[i] for i in range(t-1)))))
print("n : ", n)


for b in range(1, 5):
    MM = []
    diag = []

    if b==3:
        X2_aggregated = np.concatenate((X21[:12],X22[:12]))
        X2_aggregated = preprocess(X2_aggregated, p, True, 0, minmax=True, norm=0)
        X.append([X2_aggregated[:12].T, X2_aggregated[12:].T])
        n_t.append([12,12])
        n=0
        for i in n_t:
            n+=sum(i)
        print("n : ", n)  
    
    if b==4:
        X3_aggregated = np.concatenate((X31[:20],X32[:20]))
        X3_aggregated = preprocess(X3_aggregated, p, True, 0, minmax=True, norm=0)
        X.append([X3_aggregated[:20].T, X3_aggregated[20:].T])
        n_t.append([20, 20])
        n=0
        for i in n_t:
            n+=sum(i)
        print("n : ", n)    
    
    if b==2:
        X4_aggregated = np.concatenate((X41[:90],X42[:80]))
        X4_aggregated = preprocess(X4_aggregated, p, True, 0, minmax=True, norm=0)
        X.append([X4_aggregated[:90].T, X4_aggregated[90:].T])
        n_t.append([90, 80])
        n=0
        for i in n_t:
            n+=sum(i)
        print("n : ", n)
    
    for i in range(b):
        MM1, diag1 = empirical_mean_old(1, m, [X[i]], p, [n_t[i]], 0)
        # chaque moyenne empirique calculée est un vecteur de taille p
    #         sent+=1
    #         t_MM.append(time()-t0)
        MM.append(MM1)
        diag.append(diag1)
    V, y_opt, correlation_matrix, Dc, c0, MM_gathered, y_n, V_naive, y_s, V_s = merging_center(MM, diag, b, m, p, n, n_t, task_target, naive=True, single=True)
    # matprint(Dc)
    # END CENTRAL SERVER
    #     VTX = V.T@X_test_aggregated.T
    #     var.append(np.var(VTX))
    m_t = create_mt(b, m, y_opt, Dc, correlation_matrix, c0)
    m_t_naive = create_mt(b, m, y_n, Dc, correlation_matrix, c0)
    m_t_s = create_mt(b, m, y_s, Dc, correlation_matrix, c0)
    x = np.linspace(-10,10, 500)
    
    plt.plot(x, norm.pdf(x, m_t[task_target][0], 1))
    plt.xlim(-3+m_t[task_target][1],m_t[task_target][0]+3)
    plt.axvline(x=m_t[task_target][0],ls='--')
    plt.axvline(x=m_t[task_target][1],ls='--')
    plt.plot(x, norm.pdf(x, m_t[task_target][1], np.var(X_test_aggregated[:n_t_test[0][0]])))
    debug_histogram(V, X_test_aggregated.T, n_t_test)

#     erreur_theorique = error_rate(t, m,  Dc, MM_true, c0)[0][0]
    emp_rate.append(compute_error_rate(X_test, V, m_t, m, n_t_test, Dc, c0, task_target=0, average=1))
    emp_rate_naive.append(compute_error_rate(X_test, V_naive, m_t_naive, m, n_t_test, Dc, c0, task_target=0, average=1))
    emp_rate_s.append(compute_error_rate(X_test, V_s, m_t_s, m, n_t_test, Dc, c0, task_target=0, average=1))
print(emp_rate, emp_rate_s, emp_rate_naive)

plt.plot(range(len(emp_rate)), emp_rate, '-o', label='MTL-SPCA with optimized labels')
plt.plot(range(len(emp_rate)), emp_rate_naive, '-x', label=r"N-SPCA with $y=\pm1$")
plt.plot(range(len(emp_rate)), emp_rate_s, '-o', label='ST-SPCA')
# lower = np.array(emp_rate) - np.array(var)
# upper = np.array(emp_rate) + np.array(var)
# plt.fill_between(list(range(k)), lower, upper, alpha=0.2, label="variance")
ticks = ["Caltech", "Amazon", "Webcam", "DSLR"]
plt.xticks(range(len(ticks)), ticks, size='larger')
plt.xlabel("Added tasks")
# plt.ylim(0.18,0.24)
plt.ylabel("Empirical error rate")
plt.legend()
plt.grid()
plt.title("Real data")
plt.show()

