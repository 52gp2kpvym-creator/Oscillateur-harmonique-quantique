#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:07:22 2025

@author: baptisteviu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integr
import numpy.linalg as alg



#partie I


pi=np.pi
R=30
N0=3
N1=5
N2=8
N3=15
N4=60
aimage=1e-9

def kron(a,b):  #définition du symbole de kronecker
    if a==b:
        return 1
    else:
        return 0
    
#Question I/A/c
#définition du potentiel harmonique de l'énoncé pour 
#pouvoir le tracer pour différentes valeurs de R    
def vohimage(x):  
  return ((pi**2)/4)*(R**2)*(((x/aimage)-0.5)**2)


#tracé du potentiel harmonique 
Ximage=np.linspace(0,aimage,1000)
Yimage=[vohimage(x) for x in Ximage]
plt.plot(Ximage,Yimage,label='potentiel harmonique')
plt.xlabel('x')
plt.grid()
plt.legend()
plt.show()

def voh(x):  #définition du potentiel harmonique de l'énoncé
  return ((pi**2)/4)*(R**2)*((x-0.5)**2)
        
#Question I/B/1
def H(N):  #construction de l'hamiltonien
    M=[[0 for i in range(0,N)] for j in range(0,N)]
    for n in range (0,N):
        for m in range(0,N):
            def f(x):
                return (np.sin((n+1)*pi*x)*np.sin((m+1)*np.pi*x)*voh(x))
            M[n][m]=(n+1)*(n+1)*kron(n,m)+2*integr.quad(f,0,1)[0]
    return M

print(np.array(H(6))) #constuction de l'Hamiltonien pour N=6


#Question I/B/2

#ensemble des énergies (valeurs propres de l'hamiltonien) 
#classées par ordre croissant
def EnsembleE(N): 
    return alg.eigh(H(N))[0]



def epsilon(n):
    return EnsembleE(N4)[n-1]

#Tracé des énergies propres normalisées et 
#ajustement des courbes correspondantes

Y=[epsilon(n) for n in range(1,N4+1,1)]
X=[k for k in range(1,N4+1,1)]
X2=[k for k in range(0,N4,1)]
Y2=[R*(k+0.5) for k in range(0,N4,1)]#Tracé des énergies de l'OH
Y3=[195+k**2 for k in range(1,N4+1,1)]#Tracé des énergies du puit infini

plt.plot(X,Y,'+',label="énergies normalisées")
plt.plot(X2,Y2,label="OH normalisé")
plt.xlabel("valeurs de n et k")
plt.plot(X,Y3,label='epsilon(n) ajustée')

plt.legend()
plt.grid()
plt.show()


   
#Question I/B/3/

#construction des vecteurs propres pour l'état fondamental
def vecteurpropref(N):  
    return alg.eigh(H(N))[1][:,0]

#construction des vecteurs propres pour le premier état excité
def vecteurpropre1(N): 
    return alg.eigh(H(N))[1][:,1]

#construction des vecteurs propres pour le deuxième état excité
def vecteurpropre2(N):  
    return alg.eigh(H(N))[1][:,2]

def phi(n,x):
    return np.sqrt(2)*np.sin(n*np.pi*x)

#construction de la fonction d'onde pour l'état fondamental
def psif(N,x):
    S=0
    for m in range(0,N):
        S+=phi(m+1,x)*vecteurpropref(N)[m]
    return S

#construction de la fonction d'onde pour le premier état excité
def psi1(N,x):
    S=0
    for m in range(0,N):
        S+=phi(m+1,x)*vecteurpropre1(N)[m]
    return S

#construction de la fonction d'onde pour le deuxième état excité
def psi2(N,x):
    S=0
    for m in range(0,N):
        S+=phi(m+1,x)*vecteurpropre2(N)[m]
    return S

#fonction d'onde théorique du premier état excité
def psithexcite1(x): 
    return ((((R**3)*(pi**5)/2)**(1/4))*(x-0.5)
            *np.exp(-(pi**2)*(R/4)*((x-0.5)**2)))


#fonction d'onde théorique du deuxième état excité
def psithexcite2(x):
    return (((R*pi/8)**(1/4))*((pi**2)*R*((x-0.5)**2)-1)
            *np.exp(-(pi**2)*(R/4)*((x-0.5)**2)))

#fonction d'onde théorique de l'état fondamental
def psithfondamental(x):
    return (((R*pi/2)**(1/4))*np.exp(-(pi**2)*(R/4)*((x-0.5)**2)))


X=np.linspace(0,1,200)

#Tracé des fonctions d'onde numériques/théoriques 
#pour plusieurs valeurs de N (état fondamental)
Y0=[-psif(N0,x) for x in X]
Y1=[psif(N1,x) for x in X]
Y2=[-psif(N2,x) for x in X]
Y3=[psif(N3,x) for x in X]
Y4=[psithfondamental(x) for x in X]
plt.plot(X,Y0,label='N=3')
plt.plot(X,Y1,label='N=5')
plt.plot(X,Y2,label='N=8')
plt.plot(X,Y3,label='N=15')
plt.plot(X,Y4,label='courbe théorique')

#Tracé des fonctions d'onde numériques et théoriques 
#pour différentes valeurs de N (état excité 1)
Y5=[-psi1(N0,x) for x in X]
Y6=[psi1(N1,x) for x in X]
Y7=[psi1(N2,x) for x in X]
Y8=[-psi1(N3,x) for x in X]
Y9=[psithexcite1(x) for x in X]
plt.plot(X,Y5,label='N=3')
plt.plot(X,Y6,label='N=5')
plt.plot(X,Y7,label='N=8')
plt.plot(X,Y8,label='N=15')
plt.plot(X,Y9,label='courbe théorique')

#Tracé des fonctions d'onde numériques et théoriques 
#pour différentes valeurs de N (état excité 1)
Y11=[psi2(N0,x) for x in X]
Y12=[psi2(N1,x) for x in X]
Y13=[-psi2(N2,x) for x in X]
Y14=[psi2(N3,x) for x in X]
Y10=[psithexcite2(x) for x in X]
plt.plot(X,Y11,label='N=3')
plt.plot(X,Y12,label='N=5')
plt.plot(X,Y13,label='N=8')
plt.plot(X,Y14,label='N=15')
plt.plot(X,Y10,label='courbe théorique')


plt.xlabel('x')
plt.legend()
plt.grid()
plt.show()
    
#partie II
e=1.602176634e-19#C
V0=0.5#eV
a=1e-9 #m
b=5*a
m=9.1e-31 #kg
hbar=1.054571628e-34#J.s


#Question II/B/2/


#énergies propres d'une particule dans le puit de l'énoncé en eV
def E0(n):
    return ((n*hbar*pi/b)**2)/(8*m*e)

#fonction d'onde d'une particule dans le puit de l'énoncé
def phi2(n,x):
    return np.sqrt(1/b)*np.sin(n*pi*(x+b)/(2*b))


#Profondeur du puit
def Vfinal(x):
    if abs(x)<a:
        return -V0
    else:
        return 0
    
#construction de l'hamiltonien d'une particule dans le puit de l'énoncé
def H2(N):
    M=[[0 for k in range(0,N)]for k in range(0,N)]
    for n in range(0,N):
        for m in range(0,N):
          def f(x):
              return phi2(n+1,x)*phi2(m+1,x)*Vfinal(x)
          M[n][m]+=E0(n+1)*kron(n,m)+integr.quad(f,-b,b)[0]
    return M


#Ensemble des énergies propres (en eV) 
#d'une particule située dans le puit de l'énoncé
def EnsembleE2eV(N):
    L=alg.eigh(H2(N))[0]
    return L
    
#construction du vecteur propre pour 
#l'état fondamental
def vraisvecteurpropresfondamental(N):
    return alg.eigh(H2(N))[1][:,0]

#construction du vecteur propre pour 
#le premier état excité
def vraisvecteurpropres1(N):
    return alg.eigh(H2(N))[1][:,1]

#construction du vecteur propre 
#pour le deuxième état excité
def vraisvecteurpropres2(N):
    return alg.eigh(H2(N))[1][:,2]


#Construction des vecteurs popres pour N=150
vp=vraisvecteurpropresfondamental(150)
vp1=vraisvecteurpropres1(150)
vp2=vraisvecteurpropres2(150)

#Construction de la fonction d'onde d'une 
#particule située dans l'état fondamental
def psi2fondamental(N,x):
    S=0
    for m in range(0,N):
        S+=phi2(m+1,x)*vp[m]
    return S

#Construction de la fonction d'onde d'une particule 
#située dans le premier état excité
def psi2_1(N,x):
    S=0
    for m in range(0,N):
        S+=phi2(m+1,x)*vp1[m]
    return S

#Construction de la fonction d'onde d'une particule 
#située dans le deuxième état excité
def psi2_2(N,x):
    S=0
    for m in range(0,N):
        S+=phi2(m+1,x)*vp2[m]
    return S

#Tracé des fonctions d'onde pour les différents niveaux d'énergie
Xnew=np.linspace(-b,b,100)

Yfondamental=[psi2fondamental(150,x) for x in Xnew]
Y1=[psi2_1(150,x) for x in Xnew]
Y2=[psi2_2(150,x) for x in Xnew]
plt.plot(Xnew,Yfondamental,label='psi0')
plt.plot(Xnew,Y1,label='psi1')
plt.plot(Xnew,Y2,label='psi2')
plt.xlabel('x')

#défintion de l'allure du puit fini pour le tracé de la figure 1
def puit_fini(x):
    if abs(x)>a:
        return 20000
    else:
        return -150000

#Tracé figure semblable à la figure 1
offset1=EnsembleE2eV(150)[1]*10**(5.5)
offset0=EnsembleE2eV(150)[0]*10**(5.5)
offset2=EnsembleE2eV(150)[2]*10**(5.5)
Y1figure1=[offset1+psi2_1(150,x) for x in Xnew]
Yfondamentalfigure1=[offset0+psi2fondamental(150,x) for x in Xnew]
Y2figure1=[offset2-psi2_2(150,x) for x in Xnew]
Ypuit=[puit_fini(x) for x in Xnew]

#mise en exrgue des lignes de niveau
Ylignefondamental=[offset0 for x in Xnew]
Yligneexcite1=[offset1 for x in Xnew]
Yligneexcitee2=[offset2 for x in Xnew]
#Tracé des figures
plt.plot(Xnew,Y1figure1,label='psi1')
plt.plot(Xnew,Yfondamentalfigure1,label='psi0')
plt.plot(Xnew,Y2figure1,label='psi2')
plt.plot(Xnew,Ylignefondamental,linestyle='dashed',label='E0')
plt.plot(Xnew,Yligneexcite1,linestyle='dashed',label='E1')
plt.plot(Xnew,Yligneexcitee2,linestyle='dashed',label='E2')
plt.plot(Xnew,Ypuit,label='puit')
plt.xlabel('x')
plt.ylabel("Energie (pas à l'échelle)")
plt.grid()
plt.legend(loc=1)
plt.show()




    
    
