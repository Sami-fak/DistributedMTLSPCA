#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:00:01 2021

@author: sami
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp
from scipy.stats import norm 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat

plt.style.use('seaborn-dark-palette')

def matprint(mat, fmt="g"):
    """
    Pour une un print plus clair de la matrice
    https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def mean_matrix(p, beta=None, k=2, m=2,starting=1):
    """
    Crée des vecteurs de moyennes en respectant les conditions de non trivialité 
    retourne un tableau contenant k tableaux de m moyennes.
    beta est le paramètre de task relatedness. Si beta n'est pas précisé, il est tiré au hasard uniformément sur [0,1] pour chaque t.
    k=2, m=2 par défaut
    if starting==1 la premiere moyenne sera les vecteurs canoniques e1 et ep
    """
    mu = np.zeros((p,1))
    mu[0]= 1
    mu_ortho = np.zeros((p,1))
    mu_ortho[1] = 1
    
    M = []
    classes = []
    if starting==1:
        for l in range(m):
            classes.append((-1)**l*mu)
        M.append(classes)
    
    for t in range(starting,k):          
        mu_t = beta*mu+np.sqrt(1-beta**2)*mu_ortho
        classes = []
        for l in range(m):
            classes.append((-1)**l*mu_t)
        M.append(classes)
    return M

def gaussian_synthetic_data(n, p, m, t, n_t, M, centered=False):
    """
    Renvoie un tableau de données synthétiques gaussiennes. X[0] accède aux données de la premiere tache.
    X[0][1] accede aux données de la deuxieme classe de la premiere tache.
    (vecteurs gaussiens de taille n_j * p tq sum(n_j for j) = n)
    à partir du nombre d'échantillons n de taille p et du nombre de classe m.
    t est le nombre de tâches
    n_t est un vecteur comprenant les différentes valeurs n_j pour chaque task
    M est la matrice des moyennes de chaque composante 
    de chaque vecteur aléatoire
    """
    
    X = []
    tmp = []
    y_test = []
    
    for task in range(t):
        # pour une tache on a m classes
        tmp = []
        for k in range(m):
            mean = np.reshape(M[task][k], p)
            X_k = np.random.multivariate_normal(mean, np.identity(p), size=(n_t[task][k]))
            y_test.append(k)
            if centered:
                X_k = preprocess(X_k, p, False, 1)
            X_k = np.transpose(X_k)
            tmp.append(X_k)
        X.append(tmp)
            
    return X, y_test

def true_mean(M, p, nb_tasks, nb_classes):
    """
    Retourne un ndarray contenant les vraies moyennes avec lequel on peut travailler
    """
    true_M = np.empty((nb_tasks*nb_classes, p))
    for t in range(nb_tasks):
        for l in range(nb_classes):
            mean = np.reshape(M[t][l], (p,))
            true_M[t*nb_classes+l] = mean

    return np.transpose(true_M)

def power_diagonal_matrix(D, exponent):
    """
    Pour travailler avec des exposants négatifs
    """
    diag = np.zeros(len(D))
    for i in range(len(D)):
        diag[i] = D[i][i]**exponent
    
    return np.diag(diag)

def estimate_c(n_t, n, nb_tasks, nb_classes):
    """
    Estime le vecteur c en divisant n_t[nb_tasks][nb_classes]/n
    """
    c = np.empty(nb_tasks*nb_classes)
    for task in range(nb_tasks):
        for m in range(nb_classes):
            c[task*nb_classes+m]=n_t[task][m]/n
            
    return c

def compute_M_cal(n,p,Dc,MM, k=2, display=False):
    """
    renvoie la matrice M cursive estimée.
    O(2*k)
    """
    c0 = p/n
    correlation_matrix = 1/c0*np.power(Dc, 1/2)@MM@np.power(Dc, 1/2)
    
    return correlation_matrix

# a revoir ?

def label_evaluation(nb_tasks, nb_classes, Dc, M_estimated, c0, task_target=None):
    """
    Evalue le label y pour une tache t pour 2 classes
    task_target=None par défaut, permet de choisir la tâche target en cas d'algo distribué.
    """
    e3_e4 = np.zeros((nb_tasks*nb_classes,1))
    if task_target is not None:
        e3_e4[2*task_target] = 1
        e3_e4[2*task_target+1] = -1
    else:
        e3_e4[-2] = 1
        e3_e4[-1] = -1
    tilde_y=np.linalg.solve((Dc+Dc@M_estimated@Dc*1/(nb_tasks*c0)),(Dc*1/(nb_tasks*c0)@M_estimated@(e3_e4)))
    return tilde_y

def asymptotic_mean(nb_tasks, nb_classes, y_tilde, Dc, correlation_matrix, t, j, c0=1/21):
    """
    compute asymptotic mean m_tj
    t current task
    j current class
    """
    y_transpose = np.transpose(y_tilde)
    etj = np.zeros((nb_tasks*nb_classes, 1))
    etj[t*nb_classes+j] = 1
    power_dc = power_diagonal_matrix(Dc, -1/2)
    m_tj = np.sqrt(c0)*y_transpose.dot(np.power(Dc, 1/2)).dot(correlation_matrix).dot(power_dc).dot(etj)
    m_tj /= np.sqrt(y_transpose.dot(np.power(Dc, 1/2).dot(correlation_matrix).dot(np.power(Dc, 1/2)) + Dc).dot(y_tilde))
    return m_tj[0][0]

def aggregate_array(X, p, n, nb_tasks, nb_classes):
    """
    Renvoie un ndarray de taille pxn à partir du tableau de données X = [[X11, X12], [X21, X22], ...]
    """
    X_aggregated = np.empty((p, n))
    class_1 = X[0][0]
    for t in range(nb_tasks):
        for l in range(nb_classes):
            if t==0 and l==0:
                continue
            class_1 = np.append(class_1, X[t][l], 1)
    X_aggregated = class_1
    return X_aggregated

def create_J(nb_classes, nb_tasks, n, n_t):
    """
    Crée le vecteur J
    """
    left = 0
    beg = 0
    for i in range(nb_tasks):
        left += int(sum(n_t[i]))
    J = np.zeros((left, nb_tasks*nb_classes))
    for t in range(nb_tasks):
        for j in range(nb_classes):
            for i in range(beg, beg+n_t[t][j]):
                J[i][t*nb_classes+j] = 1
            beg += n_t[t][j]
    return J

def compute_V_old(y_tilde, X, J):
    """
    Utilise la formule explicite de V pour la classification binaire
    """
    xy_product = X.dot(J).dot(y_tilde)
    return xy_product/np.linalg.norm(xy_product)

def compute_V(y, X, J, n):
    """
    Recalcule V (calcul vecteur propre)
    """
    eigenvalue, V = np.linalg.eig(X.dot(J).dot((y).dot(y.T)).dot(J.T).dot(X.T)/(n))
    maximum = eigenvalue[0]
    idx_larg = 0
    
    for idx, value in enumerate(eigenvalue):
        if value > maximum:
            maximum = value
            idx_larg = idx
    
    largest_eigenvalue = np.sort(eigenvalue.real)[-1]
    return V.T[idx_larg].real

def create_mt(t, m, y, Dc, correlation_matrix, c0):
    m_t = []
    for k in range(t):
        m_tj = []
        for l in range(m):
            m_tj.append(asymptotic_mean(t, m, y, Dc, correlation_matrix, k, l, c0))
        m_t.append(m_tj)

    return m_t

def compute_score(V, x, m_t, rho1=0.5, rho2=0.5, average=True):
    """
    x vecteur aléatoire que l'on veut classifier
    On compare V^Tx à la moyenne des moyennes estimées pour les deux classes de la tache t
    """
    x_projection = np.transpose(V).dot(x)
    if average:
        average_mean = 1/2*(m_t[0] + m_t[1]) 
        # - 1/(m_t[0]-m_t[1])*np.log(rho1/rho2)
    else:
        average_mean = 0
    return (1 if x_projection > average_mean else -1) 

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def error_rate(nb_tasks, nb_classes, Dc, M_cur, c0, task_target=1):
    """
    Calcule l'erreur théorique
    """
    e3 = np.zeros((nb_tasks*nb_classes, 1))
    e3[2*task_target] = 1
    power_dc = power_diagonal_matrix(Dc, -1/2)
    inv = np.linalg.inv(M_cur+np.identity(nb_tasks*nb_classes))
    # print("arg q-func : ")
    # print(e3.T.dot(M_cur).dot(Dc).dot(np.linalg.inv(Dc.dot(M_cur).dot(Dc)+c0*Dc)).dot(Dc).dot(M_cur).dot(e3))
    return qfunc(np.sqrt(e3.T.dot(M_cur).dot(Dc).dot(np.linalg.inv(Dc.dot(M_cur).dot(Dc)+c0*Dc)).dot(Dc).dot(M_cur).dot(e3)))

def optimal_rate(xx, rho1, rho2):
    """
    Renvoie l'optimum bayésien.
    """
    return 1-(rho1*qfunc(xx/2+1/xx*np.log(rho1/rho2))+rho2*qfunc(xx/2-1/xx*np.log(rho1/rho2)))

def compute_error_rate(X_test, V, m_t, nb_classes, n_t, Dc, c0, task_target=1, rho1=0.5, rho2=0.5, debug=False, average=True):
    """
    Compute empirical error rate
    """
#     print("mt1= \n", m_t[1])
    error = 0
    eps1,eps2=0,0
    ni = sum(n_t[0])
    for l in range(nb_classes):
        for i in range(n_t[0][l]):
            # on prend la transposée pour pouvoir travailler avec les colonnes

            score = compute_score(V, X_test[0][l].T[i].T, m_t[task_target], rho1, rho2, average)
            # misclassifaction of class 1 in class 2, 
            if (score == -1 and l == 0):
                error +=1
                eps1+=1
            #missclassification of class 2 in class 1
            elif (score == 1 and l == 1):
                error +=1
                eps2+=1
                    
    erreur_emp = error/ni
    
    if erreur_emp > 0.5:
        erreur_emp=1-erreur_emp
        
    if debug:
        print(eps1/n_t[0][0])
        print(eps2/n_t[0][1])
        print(erreur_emp)
        print(eps1/n_t[0][0]*rho1+eps2/n_t[0][1]*rho2)
        eps1=eps1/n_t[0][0]
        eps2=eps2/n_t[0][1]
        if eps1>0.5:
            eps1=1-eps1
        if eps2>0.5:
            eps2=1-eps2
        print(f"erreur empirirque = {erreur_emp}")
        return erreur_emp, eps1, eps2
    return erreur_emp

def debug_histogram(V, X_test, n_t):
    """
    Trace l'histogramme de V^T*x_1 et V^T*x_2.
    """
#     print(X1[0][0])
#     print(n_t[0][0])
    alpha = 0.5
    bins = 20
    plt.hist(V.T.dot(X_test.T[:n_t[0][0]].T), bins = bins, alpha=alpha, label=r"$C_1$", density=True)
    plt.hist(V.T.dot(X_test.T[n_t[0][0]:].T), bins = bins, alpha=alpha, label=r"$C_2$", density=True)
    plt.grid()
    plt.title(r"Histogramme des données de tests projetées sur $V$ : $V^Tx_j$")
    plt.legend()
    plt.show()
    
def empirical_mean(nb_tasks, nb_classes, X, p, n_t, display=False):
    """
    Retourne la matrice M avec les produits scalaires croisés
    cf. Remark 1
    """
    
    M = np.empty((nb_tasks*nb_classes, nb_tasks*nb_classes)) # ici 4x4
    for i in range(nb_tasks):
        for j in range(nb_classes):
            for k in range(nb_tasks):
                for l in range(nb_classes):
                    if i == k and j == l:
                        moitie = int(n_t[i][j]/2)
#                         print("DEBUG diagonal")
#                         print(f"i = {i}, j = {j}")
#                         print(i*nb_classes+j, i*nb_classes+j)
#                         print("moitie : ", moitie)
                        
                        M[i*nb_classes+j][i*nb_classes+j] = np.ones((moitie, 1)).T@X[i][j].T[:moitie]@X[i][j].T[moitie:].T@np.ones((moitie))
                        M[i*nb_classes+j][i*nb_classes+j] /= moitie**2
                    else:
#                         print(i*nb_classes+j, k*nb_classes+l)
#                         print(i, j, k, l)
                        M[i*nb_classes+j][k*nb_classes+l] = np.ones((n_t[i][j], 1)).T@X[i][j].T@X[k][l]@np.ones((n_t[k][l]))
                        M[i*nb_classes+j][k*nb_classes+l] /= n_t[i][j]*n_t[k][l]
    
    if display:
        for t in range(nb_tasks):
            for l in range(nb_classes):
                print(f"class {t*nb_classes+l} empirical mean = {np.mean(M[t*nb_classes+l])}")
                
    return M

def empirical_mean_old(nb_tasks, nb_classes, X, p, n_t, halves=True):
    """
    compute empirical mean for data 
    retourne la matrice M de taille px(2*k) et un vecteur contenant les coefficients diagonaux
    nb_classes=2 (toujours)
    """
    M = np.empty((nb_classes*nb_tasks, p))
    diag = []
    emp = 0
    for t in range(nb_tasks):
        # O(k)
        for l in range(nb_classes):
            # O(2)
            emp = X[t][l].dot(np.ones((n_t[t][l])))/n_t[t][l]
            M[t*nb_classes+l] = emp
            # O(1)
            
            # halves
            if halves:
                moitie = int(n_t[t][l]/2)
                mu1 = X[t][l].T[:moitie].T@np.ones((moitie))
                mu2 = X[t][l].T[moitie:].T@np.ones((moitie))
                mu1, mu2 = np.reshape(mu1, (p, 1)), np.reshape(mu2, (p, 1))
                diag.append(mu1.T@mu2/moitie**2)
            else:                
                diag.append(emp.T@emp-p/n_t[t][l])
    return M.T, diag


def gather_empirical_mean(nb_tasks, nb_classes, emp_means, diag_means, p, n_t):
    """
    emp_means est un vecteur contenant les moyennes empiriques de chaque tache de chaque classe.
    Chaque vecteur de moyennes et de taille px1
    Renvoie la matrice M des, produits scalaires entre moyennes empiriques de chaque client
    """
    # print("diag : ", diag_means)
    M = np.empty((nb_classes*nb_tasks, nb_classes*nb_tasks)) # ici 4x4
    for i in range(nb_tasks):
        # O(k)
        for j in range(nb_classes):
            # O(2)
            for k in range(nb_tasks):
                # O(k)
                for l in range(nb_classes):
                    # O(2)
                    if i == k and j == l:
                        # print(i*nb_classes+j,i*nb_classes+j)
                        M[i*nb_classes+j][i*nb_classes+j] = diag_means[i*nb_classes+j]
                    else:
                        M[i*nb_classes+j][k*nb_classes+l] = emp_means[i*nb_classes+j].T@emp_means[k*nb_classes+l]
                
    return M

def merging_center(MM, diag, t, m, p, n, n_t, task_target=None, display=False):
    """
    Recoit les moyennes empiriques des k clients, calcule la matrice de corrélation, les labels optimaux et renvoie le vecteur V
    Renvoie y un vecteur de labels optimaux adapté à chaque client. (à changer?)
    
    Paramètres
    -----------
    
    MM = [[MM11, MM12], [MM21, MM22], [MM31, MM32], ...], contient les moyennes empiriques des deux classes de chaque tâche
    diag = [diag1[0], diag1[1], etc.] contient les coefficients diagonaux des matrices MM pour chaque tâche
    t le nombre de tâche
    m le nombre de classes
    n_t = [[n11, n12], [n21, n22]]
    if display: print optimal labels, MM and M gothique
    """
    
    emp_means = []
    diagonal = []
    
    for i in range(len(MM)):
        for l in range(m):
            emp_means.append(np.reshape(MM[i].T[l], (p, 1)))
            diagonal.append(diag[i][l])
    #emp_means = [MM11, MM12, MM21, MM22, MM31, ...]
    MM_gathered = gather_empirical_mean(t, m, emp_means, diagonal, p, n_t)
    
    if display:
        print("MM_gathered : ")
        matprint(MM_gathered)
    
    c = estimate_c(n_t, n, t, m)
    c0 = p/n
    Dc = np.diag(c)
    correlation_matrix = compute_M_cal(n, p, Dc, MM_gathered, display=display)
    y = label_evaluation(t,m,Dc,MM_gathered, c0, task_target=task_target)
#     e3=np.zeros((m*t,1));e3[-2]=1;
#     e4=np.zeros((m*t,1));e4[-1]=1;
#     y = np.linalg.solve((np.diag(c)+np.diag(c)@MM_gathered@np.diag(c/c0)),(np.diag(c/c0)@MM_gathered@(e3-e4)))
#     tilde_y=np.linalg.solve((Dc+Dc@M_estimated@Dc*1/c0),(Dc*1/c0@M_estimated@(e3_e4)))
    if display:
        print("y : ")
        matprint(y)
    
    # le serveur calcule aussi V
    V = np.zeros((p,1))
    for i in range(t):
        for j in range(m):
            V += n_t[i][j]*emp_means[i*m+j]*y[i*m+j]
    V /= np.linalg.norm(V)
    V = np.reshape(V, (p))
    
    return V, y, correlation_matrix, Dc, c0, MM_gathered

def normalisation(X, p, z=False):
    """
    si z: z-score normalisation par tâche
    sinon: divsion par sqrt(p)
    """
    if not z:
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] *= 1/np.sqrt(p)
    else:
        for i in range(len(X)):
            task = np.concatenate((X[i][0], X[i][1]), axis=1)
            mean = np.mean(task, axis=1)
            mean = np.reshape(mean, (p, 1))
            X[i][0] = (X[i][0] - mean)/np.std(task)
            X[i][1] = (X[i][1] - mean)/np.std(task)
    return X

def divide_array(X,y,k,m=2):
    """
    Divide a data ndarray in an array of k arrays of m ndarrays each.
    Return the divided array and the number of elements in each classes of each tasks
    """
    X_data = []
    n_t = []
    for task in range(k):
        n_t.append([])
        X_data.append([])
        n_t1 = np.count_nonzero(y)
        n_t[task].append(n_t1)
        n_t[task].append(len(y) - n_t1)
        X_data[task].append(X[:n_t1].T)
        X_data[task].append(X[n_t1:].T)
        
    return X_data, n_t

def preprocess(X, p, std=True, axis=0, minmax=False, norm=False):
    """
    Centre et réduit les données X
    """
    if minmax:
        scaler = MinMaxScaler()
        scaler.fit_transform(X)
        return X
    if norm:
        X /= np.linalg.norm(X)
    # tiled = np.tile(np.reshape(np.sum(X, axis=0), (p, 1)), (1, X.shape[0])).T
    # X_t = np.true_divide(X, tiled, where=(tiled!=0))
    return preprocessing.scale(X, axis=axis, with_std=std)
    




