#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import quad,odeint,solve_ivp
from scipy import linalg
#from scipy.optimize import fsolve
from scipy.special import gamma, factorial,binom
from scipy import special
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# In[2]:


#Algorithm 1 Al-Kashi's method of calculating pi
import numpy as np                  # import numerical library 

m = 20                              # number of iterations

xa = np.sqrt(3.0)                   # initial guess for x
na = 6.0                            # number of sided on polygon
print( 0,'first guess',3*np.sqrt(4 - xa**2 ) ) 
for i in range(m):
    xb = np.sqrt( 2 + xa )
    nb = 2.0*na
    ab = nb/2.0*np.sqrt(4 - xb**2 ) 
    print( i+1, ab)
    xa = xb
    na = nb
    pass                             # end loop


# In[ ]:





# In[3]:


from mpmath import mp


# In[4]:


#Algorithm 1 Al-Kashi's method of calculating pi
import numpy as np                  # import numerical library 
mp.dps = 50
m = 20                              # number of iterations

xa=np.sqrt(3)
na=6
print( 0,'first guess',mp.mpf(3)*mp.sqrt(mp.mpf(4.0) - xa**2 ) ) 
for i in range(m):
    xb= mp.sqrt( mp.mpf(2.0) + xa )
    nb= mp.mpf(2.0)*na
    ab= nb/mp.mpf(2.0)*mp.sqrt(mp.mpf( 4.0) - xb**2 ) 
    print( i+1, ab )
    xa=xb
    na=nb
    pass                             # end loop


# In[5]:


np.pi


# In[6]:


# Algorithm 2 Heron's or Babylonian method to find a square root.
N = 231001                                      # find sqrt of N
a0= 1/2                                     # initial guess
m = 20                                       # number oif iterations
print('square root of ', N, 'initial guess', a0)
aa = a0 + (N - a0**2)/(2*a0)                # initial iteration eqn 6
print( 0, aa )                              # print 1st value
for i in range(m-1):                        # do iteration
    ab = aa +( N - aa**2 )/( 2*aa )         # equation 6
    print( i+1,  ab)
    aa = ab                                 # replace new value with old 
    pass


# In[7]:


np.sqrt(231001)


# In[ ]:





# In[8]:


from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import cm 

from scipy.spatial import Delaunay 
plt.rcParams.update({'font.size': 16})  # set font size for plots

fig=plt.figure(figsize=(8,8))


def Icosahedron():   # coords from golden rectangle  with 2 height 2g
    h = 0.5*(1+np.sqrt(5)) 
    p1 = np.array([[0,1,h],[0,1,-h],[0,-1,h],[0,-1,-h]]) 
    p2 = p1[:,[1,2,0]]    # rotate 1 left  
    p3 = p1[:,[2,0,1]]
    print(p1)
    print(p2)
    print(p3)
    return np.vstack((p1,p2,p3)) 

Ico = Icosahedron() 

CH = Delaunay(Ico).convex_hull 
x,y,z = Ico[:,0], Ico[:,1], Ico[:,2] 

ax = fig.add_subplot(111, projection='3d') 
S = ax.plot_trisurf(x,y,z,triangles = CH ,shade = False, cmap=cm.jet,alpha=1) 

ax.view_init(11, 120)
ax.set_xticklabels([]) 
ax.set_yticklabels([]) 
ax.set_zticklabels([]) 
ax.axis('off')

plt.tight_layout()

#plt.savefig('chapter1-fig6.png')
plt.show() 


# In[ ]:





# In[9]:


# Algorithm 3 Golden Search

f= lambda x:  x**3 - 5*x**2           # define a functio
a = 1.0                               # set limits a and b
b = 6.0
N = 20                                # number of iterations 
Lmt= 0.01                             # smallest b-a allowed 
g = (np.sqrt(5.0)-1.0)/2.0            # golden ratio
ka = g*a+(1-g)*b                      # define start points
kb = g*b+(1-g)*a

fa = f(ka)                            # function at start pos’ns
fb = f(kb)
print(fa,fb)
print('   i    ka    kb     fa       fb    b-a') 

i= 0
while abs(b-a)> Lmt and i < N-1 :
    i = i+1
    if fa < fb :                       # rule(i)
        b = kb
        kb= ka
        fb= fa 
        ka= g*a+(1-g)*b
        fa= f(ka)
    else:                              # rule(ii)
        a = ka
        ka= kb
        fa= fb
        kb= g*b+(1-g)*a
        fb= f(kb)
    print('{:4d} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format( i, ka, kb, fa, fb, b-a ) )
    pass                               # end while

print(f(2))


# In[10]:


# Algorithm 4 Series calculation of an exponetial exp(x).
p = 1.0 
s = 1.0
x = 2.0               # exponetial value to calculate
for i in range(1,20): 
    p = p*x/i 
    s = s + p 
    print(i,s)
    pass


# In[11]:


np.exp(2)


# In[12]:


plt.rcParams.update({'font.size': 16})  # set font size for plots

fig=plt.figure(figsize=(6,6))

x=np.linspace(-5,5,200)
xx=np.linspace(1e-6,5,200)

plt.plot(x,x,color='grey',linestyle='dashed')
plt.plot(x,np.exp(x),color='blue')
plt.plot(x,np.exp(-x),color='red')
plt.plot(xx,np.log(xx),color='blue')
#plt.plot(x,np.log(np.abs(x)),color='blue')
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.axhline(0,linewidth=1,color='grey')
plt.axvline(0,linewidth=1,color='grey')
plt.xlabel('x')
plt.ylabel('y')
plt.annotate(r'$e^x$',xy=(1.2,np.exp(1)))
plt.annotate(r'$e^{-x}$',xy=(-2.2,np.exp(1)))
plt.annotate(r'$\ln(x)$',xy=(2.5,np.log(4)))

#plt.savefig('chapter1-fig8.png')
plt.show()


# In[13]:


plt.rcParams.update({'font.size': 16})  # set font size for plots

fig=plt.figure(figsize=(6,4))

x=np.linspace(-5,5,400)

plt.plot(x,np.sin(2*np.pi*x),color='blue')
plt.plot(x,np.sin(2*np.pi*x+2/3),color='blue',linestyle='dashed')

plt.xlim([-2,2])
plt.ylim([-1,1])
plt.axhline(0,linewidth=1,color='grey')
plt.axvline(0,linewidth=1,color='grey')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

adict=dict(facecolor='black', shrink=0.01,width=1,headwidth=3)
plt.annotate(r'$\phi=0$', xy=(0.35,np.sin(2*np.pi*0.35)), xytext=(0.6, 0.9),
            arrowprops = adict,fontsize=10 )
plt.annotate(r'$\phi=2/3$', xy=(1.45,np.sin(2*np.pi*1.45 +2/3)), xytext=(0.85, -0.9),
            arrowprops = adict,fontsize=10 )
plt.tight_layout()
#plt.savefig('chapter1-fig12.png')

plt.show()


# In[14]:


plt.rcParams.update({'font.size': 16})  # set font size for plots

fig1=plt.figure(figsize=(13,5))
ax0 = fig1.add_subplot(1,2,1)
ax1 = fig1.add_subplot(1,2,2)

v1=5
v2=5.5

x=np.linspace(0,1.0,1000)
ax0.plot(x,np.sin(2*np.pi*x*v1),color='blue')
ax0.plot(x,np.sin(2*np.pi*x*v2),color='blue',linestyle='dashed')

ax0.set_xlim([0,1])
ax0.set_ylim([-2,2])
ax0.axhline(0,linewidth=1,color='grey')
ax0.axvline(0,linewidth=1,color='grey')
ax0.set_xlabel('time')
x=np.linspace(0,10,1000)

ax1.plot(x,np.sin(2*np.pi*x*v1) + np.sin(2*np.pi*x*v2),color='blue')
#ax1.plot(x,2*np.sin(2*np.pi*x*0.5/2 - np.pi/2),color='red',linestyle='dashed')

#ax1.plot(x, 2*np.sin(2*np.pi*(v1+v2)*x/2)*np.cos(2*np.pi*(v1-v2)*x/2),color='red'   )
ax1.plot(x, 2*np.cos(2*np.pi*(v1-v2)*x/2),color='red',linestyle='dashed'   )


ax1.set_xlabel('time')

ax1.set_xlim([0,5])
ax1.set_ylim([-2,2])

ax0.set_yticks([-2,-1,0,1,2])
ax1.set_yticks([-2,-1,0,1,2])

plt.tight_layout()
#plt.savefig('chapter1-fig13.png')

plt.show()


# In[15]:


plt.rcParams.update({'font.size': 16})  # set font size for plots

fig1=plt.figure(figsize=(6,6))

t=np.linspace(-2,2,500)

fx=lambda x: np.cosh(x)
fy=lambda x: np.sinh(x)
plt.plot(fx(t),fy(t),color='blue')
plt.plot(fx(t),-fy(t),color='blue')
plt.plot(-fx(t),fy(t),color='blue')
plt.plot(-fx(t),-fy(t),color='blue')
plt.scatter(fx(np.pi/4),fy(np.pi/4),color='red',zorder=10)

plt.plot(-np.cos(t),np.sin(t),color='blue')
plt.plot(np.cos(t),-np.sin(t),color='blue')
plt.scatter(np.cos(np.pi/4),np.sin(np.pi/4),s=50,color='red',zorder=10)
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.axhline(0,color='grey',linewidth=1)
plt.axvline(0,color='grey',linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
#plt.annotate(r'$(\cos(\theta),\sin(\theta))$', xy=(0.3,1),fontsize=12)

plt.annotate(r'$(\cos(\theta),\sin(\theta)$', xy=(np.cos(np.pi/4),np.sin(np.pi/4)), xytext=(0.05, 1.2),
            arrowprops = dict(facecolor='black', shrink=0.01,width=1,headwidth=3),fontsize=12 )
plt.annotate(r'$(\cosh(\theta),\sinh(\theta)$', xy=(np.cosh(np.pi/4),np.sinh(np.pi/4)), xytext=(0.5, 1.75),
            arrowprops = dict(facecolor='black', shrink=0.01,width=1,headwidth=3) ,fontsize=12)
plt.tight_layout()

#plt.savefig('chapter1-fig14.png')
plt.show()


# In[ ]:





# In[16]:


plt.rcParams.update({'font.size': 16})  # set font size for plots

fig1=plt.figure(figsize=(13,6))
ax0 = fig1.add_subplot(1,2,1)
ax1 = fig1.add_subplot(1,2,2)

x=np.linspace(-10,10,1000)

ax0.plot(x,np.cosh(x),color='blue')
ax0.plot(x,np.sinh(x),color='red')
ax0.plot(x,np.tanh(x),color='grey',linestyle='dashed')

ax0.set_xlim([-4,4])
ax0.set_ylim([-4,4])
ax0.axhline(0,linewidth=1,color='grey')
ax0.axvline(0,linewidth=1,color='grey')
ax0.set_xlabel('x')
ax0.annotate(r'$\sinh(x)$',xy=(-1.5,-3))
ax0.annotate(r'$\cosh(x)$',xy=(-2.25,1.1))
ax0.annotate(r'$\tanh(x)$',xy=(-3.4,-1.5))

ax1.plot(x,np.sin(x),color='blue')
ax1.plot(x,np.cos(x),color='red')
ax1.plot(x,np.tan(x),color='grey',linestyle='dashed')
ax1.set_xlabel('x')
ax1.axhline(0,linewidth=1,color='grey')
ax1.axvline(0,linewidth=1,color='grey')
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])

ax1.annotate(r'$\sin(x)$',xy=(2,1))
ax1.annotate(r'$\cos(x)$',xy=(-1.25,1.1))
ax1.annotate(r'$\tan(x)$',xy=(-3.4,2))


plt.tight_layout()

#plt.savefig('chapter1-fig15.png')
plt.show()


# In[17]:


plt.rcParams.update({'font.size': 16})  # set font size for plots

fig1=plt.figure(figsize=(13,6))
ax0 = fig1.add_subplot(1,3,1)
ax1 = fig1.add_subplot(1,3,2)
ax2 = fig1.add_subplot(1,3,3)


x=np.linspace(-2*np.pi,2*np.pi,1000)

ax0.plot(np.sin(x),x/np.pi,color='blue')
ax1.plot(np.cos(x),x/np.pi,color='red')
ax2.plot(np.tan(x),x/np.pi,color='grey')

ax0.set_xlim([-2,2])
ax0.set_ylim([-2,2])
ax1.set_xlim([-2,2])
ax1.set_ylim([-2,2])
ax2.set_xlim([-2,2])
ax2.set_ylim([-2,2])


ax0.axhline(0,linewidth=1,color='grey')
ax0.axvline(0,linewidth=1,color='grey')
ax0.set_xlabel('x')
ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax0.set_ylabel(r'$y/\pi$')
ax1.set_ylabel(r'$y/\pi$')
ax2.set_ylabel(r'$y/\pi$')

ax1.axhline(0,linewidth=1,color='grey')
ax1.axvline(0,linewidth=1,color='grey')

ax0.annotate(r'$\sin^{-1}(x)$',xy=(-1.7,1))
ax1.annotate(r'$\cos^{-1}(x)$',xy=(0.5,1.1))
ax2.annotate(r'$\tan^{-1}(x)$',xy=(-1.5,1.2))
ax2.axhline(0,linewidth=1,color='grey')
ax2.axvline(0,linewidth=1,color='grey')
plt.tight_layout()

#plt.savefig('chapter1-fig16.png')
plt.show()


# In[18]:


from scipy.special import gamma

def fact(n):
    if n <= 1 :
        return 1
    f=1.0
    for i in range(1,n+1):
        f = f*i
    return f
print(fact(15) )


plt.rcParams.update({'font.size': 20})  # set font size for plots

fig1=plt.figure(figsize=(13,6))
ax0 = fig1.add_subplot(1,2,1)
ax1 = fig1.add_subplot(1,2,2)

m = 20
x = np.zeros(m,dtype=int)
ff = np.zeros(m,dtype=int)
for i in range(m): 
    x[i]=i
    ff[i]= np.log( fact(i) )

ax0.scatter(x,ff,s=50,color='red')
ax0.set_ylabel(r'$\ln(n!)$')
ax0.set_xlabel(r'$n$')
ax0.set_yticks([0,10,20,30,40])
ax0.set_xticks([0,5, 10,15, 20])
ax0.set_xlim([0,20])

xx = np.linspace(-4,m,5000)

ax1.plot(xx,gamma(xx),color='blue' )
for i  in range(0,8): 
    ax1.scatter(i,fact(i-1),color='red',s=50, zorder=10 )
ax1.set_ylim([-10,30])
ax1.set_xlim([-4,6])
ax1.set_yticks([-10,0,10,20,30])
ax1.axhline(0,color='grey',linewidth=1)
ax1.set_xlabel(r'$x$')



plt.tight_layout()



#plt.savefig('chapter1-fig19.png')

plt.show()


# In[19]:


def fact(n):
    if n <= 1 :
        return 1
    f=1
    for i in range(1,n+1):
        f = f*i
    return f

j = 2
for i in range(26):
    if fact(i) > 10**j:
        print('**',i,j, fact(i))
        j=j+3


# In[20]:


def afact(n):
    if n <= 1 :
        return 1
    return n*afact(n-1)

print(afact(15))


# In[ ]:





# In[21]:


#x=symbols('x')
#f=lambda x: exp(-x)-log(x)
#findroot(f,1)
1.3098


# In[ ]:





# In[22]:



fig1=plt.figure(figsize=(13,6))
ax0 = fig1.add_subplot(2,3,1)
ax1 = fig1.add_subplot(2,3,2)
ax2 = fig1.add_subplot(2,3,3)
ax3 = fig1.add_subplot(2,3,4)
ax4 = fig1.add_subplot(2,3,5)
ax5 = fig1.add_subplot(2,3,6)


axx=['ax'+str(i) for i in range(6)]

x=np.linspace(-2*np.pi,2*np.pi,200)
f=lambda x: np.sin(x)

for i,ax in enumerate(axx):
    
    print(i,np.sin(i))
    
    qx = eval(ax)
    qx.axhline(0,color='grey',linewidth=1)
    qx.plot(x,np.sin((i-2)/2)*f(x),color='blue')
    qx.set_ylim([-1.1,1.1])
    qx.axis('off')

plt.show()


# In[23]:



# standing wave

fig1=plt.figure(figsize=(6,4))

x=np.linspace(-1*np.pi,1*np.pi,200)
f=lambda x: np.sin(x)
num = 4
cols = cm.brg_r( np.linspace(0, 1, num) )

for i in range(-num,num):
    plt.plot(x/np.pi,(i+1)/num *f(x),color=cols[i])
    
plt.plot(x/np.pi,-f(x),color='black')
plt.axhline(0,color='grey',linewidth=1)
bbox_props = dict(boxstyle="rarrow,pad=0.5",fc='None', ec="b", lw=2)
plt.text(-0.5, 0.65, "           ", ha="center", va="center", rotation=270,size=10,bbox=bbox_props)
plt.text(0.5, -0.65, "           ", ha="center", va="center", rotation=-270,size=10,bbox=bbox_props)
#plt.ylim([-1.1,1.1])
plt.xlim([-1,1])
plt.yticks([-1,-0.5,0,0.5,1])
#plt.axis('off')

#plt.savefig('chapter1-fig13a.png')

plt.show()


# In[ ]:





# In[24]:


x, n = symbols('x,n')  # hermite

def herm(n, x): 
    if n == 0:
        return 1 
    elif n == 1: 
        return 2*x 
    else: 
        return 2*x*herm(n - 1, x) - 2*(n- 1)*herm(n - 2, x)


# In[25]:


for i in range(6):
    print(simplify( herm(i,x) ))
    


# In[26]:


# Q13
x, n = symbols('x,n')  #  Legendre  chnage n-> n-1

def Legen(n, x): 
    if n == 0:
        return 1 
    elif n == 1: 
        return x 
    else: 
        return ( (2*(n-1)+1)*x*Legen(n - 1, x) - (n-1)*Legen(n - 2, x))/(n)

for i in range(6):
    print(i,expand( Legen(i,x) ))


# In[27]:


# Q13
x, n = symbols('x,n')  #  Legendre  chnage n-> n-1

def Cheb(n, x): 
    if n == 0:
        return 1 
    elif n == 1: 
        return x 
    else: 
        return  2*x*Cheb(n - 1, x) - Cheb(n - 2, x)

for i in range(6):
    print(i,expand( Cheb(i,x) ))


# In[ ]:





# In[28]:


def NA(days, num):      # birthdays 
    p = 1.0
    for i in range(num):
        p = p*(days-i)
        #print(p, days-i)
    return p

fig1=plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 16})  # set font size for plots
days = 365
num  = 80

x = np.linspace(1,num,num)
p = np.zeros(num)

print(1-NA(days,num)/days**num )

for anum in range(1,num):
    p[anum]=(1-NA(days,anum)/days**anum )

plt.scatter(x,p,color='blue',s=2)
plt.xlabel('number in group' )
plt.ylabel('probability')
plt.xlim([0,num])
plt.ylim([0,1])
plt.tight_layout()
#plt.savefig('chapter1-fig20a.png')
plt.show()


# In[29]:


def fact(n):
    if n == 1 or n==0 :
        return 1
    else:
        p = 1
        for i in range(1,n):
              p = p*(i+1)
    return p

print(fact(365)/fact(340)/365**25)


# In[30]:


fig1=plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 16})  # set font size for plots
Z = lambda E: 1+2*np.exp(-E)  # butane 


#x = [i for i in range(10)]
x=np.linspace(0,10,100)
#p = [1/Z(i) for i in range(10)]

#print(p)
plt.plot(x,1/Z(x),color='red')

plt.plot(x,1/Z(x)**10,color='blue')

plt.ylim([2e-5,2])
plt.xlim([0,10] )
plt.yscale('log')
plt.xlabel(r'$E_g/k_BT$')
plt.ylabel(r'$p(trans)$')

plt.tight_layout()
plt.annotate(r'$1/Z$',xy=(1,1))
plt.annotate(r'$1/Z^{10}$',xy=(2.5,0.1))

#plt.savefig('chapter1-fig23a.png')
plt.show()


# In[31]:


R,T,Z,Eg=symbols('R,T,Z,Eg')
Z =  1+2*exp(-Eg/(R*T)) # butane 
S=R*ln(Z)+R*T*diff(ln(Z),T)
S


# In[32]:


U=R*T**2*diff(log(Z),T)
U


# In[33]:



#entropy . 
fig1=plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 16})  # set font size for plots
S = lambda E: np.log(1+2*np.exp(-E) ) + E* 2*np.exp(-E)/(1+2*np.exp(-E) ) # butane 


#x = [i for i in range(10)]
x = np.linspace(0,10,100)
#p = [1/Z(i) for i in range(10)]

#print(p)
plt.plot(x,S(x),color='red')

plt.axhline(np.log(3))

#plt.ylim([2e-5,2])
#plt.xlim([0,10] )
#plt.yscale('log')
plt.xlabel(r'$E_g/k_BT$')
plt.ylabel(r'$S/R$')

plt.tight_layout()
#plt.annotate(r'$1/Z$',xy=(1,1))
#plt.annotate(r'$1/Z^{10}$',xy=(2.5,0.1))

#plt.savefig('chapter1-fig23a.png')
plt.show()


# In[34]:


# Q14  square root 

N = 23

k = 0
while k**2<= N:
    k = k+1
k = k-1
print(k)

r = np.zeros(20,dtype=float)
r[0]=0
r[1]=1
for i in range(2,20):
    r[i] = 2*k*r[i-1] + (N-k**2)*r[i-2] 
    print(i,r[i]/r[i-1]-k)
    
    pass
print(np.sqrt(N))


# In[35]:


#Q15 (a) fibonacci

f = np.zeros(20,dtype=int)
f[0]=1
f[1]=1
for i in range(2,20):
    f[i] = f[i-1] + f[i-2]
    print('{:4d} {:8d}   {:f}'.format( i, f[i], f[i]/f[i-1])  )
    
    pass
print((1+np.sqrt(5))/2 )


# In[ ]:





# In[36]:


#Q15

f = np.zeros(20,dtype=int)
f[0]=1
f[1]=1
for i in range(2,20):
    f[i] = 2*f[i-1] + f[i-2]
    print('{:d} {:10d} {:f}'.format( i, f[i], 1.0*f[i]/f[i-1])  )
    
    pass
print(1+np.sqrt(2))


# In[37]:



f = np.zeros(30,dtype=float)
f[0] = 1.0
f[1] = 1.0
for i in range(2,30):
    f[i] = f[i-1]+2.0*f[i-2]
    print(i,f[i], f[i]/f[i-1])
    pass


# In[38]:


# Q 16 . binomial recursion

def binom(n):
    b = 1
    for q in range(n+1):
        print(q,round(b))
        b = b*(n-q)/(q+1)
        pass
binom(12)


# In[ ]:





# In[39]:


10-39 % 10


# In[40]:


43 % 10


# In[41]:


fig1=plt.figure(figsize=(10,2))
x=np.linspace(-6100,-6000,100)

plt.plot([-6100,-6000],[0.0,0.0])

plt.scatter([-6051.78],[0],color='red')
plt.scatter([-6050],[0],color='black')

plt.plot([-6020,-6080],[0.3,0.3],color='red',linewidth=3)

plt.ylim([-0.1,0.5])
plt.xlim([-6100,-6000])
plt.yticks([])
plt.xticks([-6100,-6080,-6050,-6020,-6000])
plt.tight_layout()
#plt.savefig('chapter1-fig26.png')

plt.show()


# In[42]:



import scipy.constants
res = scipy.constants.physical_constants["Avogadro constant"]
print(res)


# In[43]:


#scipy.constants.find()


# In[44]:


x,D0,t = symbols('x D0 t',positive=True)
 
eqn= x**2*exp(-x**2/(4*D0*t))
ans = integrate(eqn, (x,-oo,oo))


# In[45]:


simplify(ans)


# In[46]:


expand( (1/(4*pi*D0*t)**(1/2))*4*sqrt(pi)*D0*t*sqrt(D0*t)   )


# In[47]:


def afact(n):
    if n==0:
        return 1
    if n==1:
        return 1
    return n*afact(n-1)


n = 60   # number of trials 60 atoms 
m = 0    # choose this number from n 
q = 0.01109 # chance of m occuring in n atoms, i.e prob of 13C in sample of C. 


# In[48]:


P=lambda n,m,q : afact(n)/(afact(m)*afact(n-m))*q**m*(1-q)**(n-m)


# In[49]:


for i in range(10):
    print(i,P(n,i,q) )


# In[50]:


from scipy.special import factorial
fig1=plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 16})  # set font size for plots

# poisson k in number , mu= mean 
pois = lambda k, mu : mu**k*np.exp(-mu)/ factorial(k)

k = np.linspace(0,20,200)
s = 0
for j in range(40):
    s = s + j*pois(j,5)
print('mean',s)

cols=['blue','red','grey','green','black']
for i,mu in enumerate( [0.5,2,6,9,12]):
    plt.plot(k,pois(k,mu),color=cols[i] )
    if mu ==6:
        for j in range(20):
            plt.scatter(j,pois(j,mu),color='black',zorder=10,s=20)

plt.xlim([0,20])
plt.ylim([0,0.65])
plt.xlabel(r'$k$')
plt.ylabel(r'$P(k,\mu)$')



mm=[2,9]
for m0 in mm:
    plt.plot([m0,m0],[0,pois(m0,m0)] ,color='black')
    sig = 1.05*np.sqrt(m0)
    plt.plot([m0,m0+sig],[pois(m0,m0)/2,pois(m0,m0)/2],'black' )


plt.text(2,0.3,r'$\mu=2 $')
plt.text(6,0.18,r'$\mu=6 $')
plt.text(9,0.15,r'$\mu=9 $')
plt.text(12,0.125,r'$\mu=12 $')

plt.text(10.5,0.042,r'$\sigma =3$',fontsize=12)

plt.tight_layout()
#plt.savefig('chapter1-fig20b.png')
plt.show()


# In[51]:


pois(0,2.93)


# In[52]:


g,N,epsilon,k,T=symbols('g,N,epsilon,k,T')

Z = (1-(g**N*exp(-N*epsilon/(k*T))))/(1-g*exp(-epsilon/(k*T))) 
avn = (1/epsilon)*simplify(( k*T**2*diff(Z,T)/Z))
avn


# In[53]:


fig1=plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})  # set font size for plots

avn   = lambda x: n/(1-(np.exp(x)/g)**n)  - 1/(1 - np.exp(x)/g)   #  x=E/kBT. blue

x = np.linspace(0,3,5000)  # x = E/k_BT
g = 4.0
n = 200.0

plt.axvline(np.log(g),color='grey',linewidth=1)
plt.plot(x,avn(x),color='blue',linewidth=1)
#plt.plot(x,f012(x),color='green',linewidth=1)

plt.ylim([0,n*1.05])
plt.xlim([0,3])

plt.ylabel(r'$\langle n\rangle$')
plt.xlabel(r'$\epsilon/k_BT$')
plt.text(1.2*np.log(g),0.89*n,'g = '+str(g)+', n = '+str(n),fontsize=14)
plt.xticks([0,1,2,3])
plt.yticks([0,50,100,150,200])

plt.tight_layout()

#plt.savefig('series-fig2c.png')
plt.show()


# In[54]:


g,N,epsilon,k,T,x=symbols('g,N,epsilon,k,T,x')

Z = (1-(g**N*exp(-N*epsilon/(k*T))))/(1-g*exp(-epsilon/(k*T))) 
U = simplify(( k*T**2*diff(Z,T)/Z))
U


# In[55]:


ans0=collect(collect( factor(simplify(diff(U,T) ) ), exp(N*epsilon/(k*T) ) ),exp(epsilon/(k*T) ) )
ans0


# In[56]:


ans=ans0.subs(epsilon/(T*k),x)


# In[57]:


print(ans)


# In[58]:


cv=lambda x:-x**2*((-g*np.exp(2*N*x) - g**(2*N + 1))*np.exp(x) + (N**2*g**N*np.exp(2*x) + N**2*g**(N + 2) + (-2*N**2*g**(N + 1) + 2*g**(N + 1))*np.exp(x))*np.exp(N*x))/((g - np.exp(x))**2*(-g**N + np.exp(N*x))**2)

epsilon=1

N=100
g=4

x=np.linspace(0,3,200)
plt.axvline(np.log(g),color='grey',linewidth=1)
plt.plot(x,cv(x))

plt.show()


# In[59]:


m0,p=symbols('m0,p')
factor(expand(m0*p*(1-p)**3 + m0*(1-p)**4 ) )


# In[60]:


factor(expand( 3*m0*p**2*(1-p)**2+3*m0*p*(1-p)**3  ) )


# In[61]:


factor(expand(3*m0*p**3*(1-p)+3*m0*p**2*(1-p)**2 ) )


# In[62]:


factor(expand(m0*p**4+m0*p**3*(1-p) ) )


# In[63]:


factor(expand(m0*p*(1-p)**2 + m0*(1-p)**3 ) )


# In[64]:


factor(expand(2*m0*p**2*(1-p) + 2*m0*p*(1-p)**2  ) )


# In[65]:


fig1=plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})  # set font size for plots


def afact(n):
    if n < 2:
        return 1
    else:
        return n*afact(n-1)*1.0
    
def lnfact(n):
    #return np.exp( n*np.log(n)  - n + 0.5*np.log(2*np.pi*n)+1/(12*n) )
    return np.exp( n*np.log(n)  - n +(1/6)*np.log(8*n**3+4*n**2+n+1/30) + 0.5*np.log( np.pi ) )# Ramanujan
    
aprob = lambda n,p,N :p*(n*p)**( N-1 )*np.exp(-n*p)/afact( N-1 )

def prob(n,p,N):  # do this way to reduce large numbers
    
    w = p*np.exp(-n*p)
    for i in range(N-1):
        w = w*n*p/(N-1-i) 
    return w

p = 0.1
N = 200
num=8000
x = np.linspace(0,num,400)

plt.plot(x,prob(x, p, N),color='blue')
plt.plot(x,prob(x, p*1.2, N),color='grey')


plt.plot(x,prob(x, p, N*3),color='blue')
plt.plot(x,prob(x, p*1.2, N*3),color='grey')
plt.text( 2000,0.003,r'$N=200$')
plt.text( 5000,0.003,r'$N=600$')
#plt.text( 1650,0.003,r'$p=$'+str(2*p))
plt.xlim([0,num])
#plt.axvline((3*N-1)/p,color='grey', linestyle='dashed',linewidth=1)
plt.yticks([0,0.001,0.002,0.003,0.004])
plt.ylim([0,0.004])
maxn = (3*N-1)/p



#plt.plot([ maxn,  maxn + np.sqrt(3*N)/p],[prob(maxn + np.sqrt(3*N)/p,p,3*N),prob(maxn + np.sqrt(3*N)/p,p,3*N)] )
#plt.ylim([0, 0.01])
plt.xlabel(r'$n$')
plt.ylabel(r'$P(n)$')
plt.tight_layout()

#plt.savefig('integration-fig17a.png')
plt.show()


# In[66]:


prob(N/p,p,N),N*p


# In[67]:


aprob(N/p,p,N),N*p


# In[116]:


n,p,N=symbols('n,p,N')

f01= p*(n*p)**(N-1)*exp(-n*p)
simplify(diff(f01,n) )


# 
# {\displaystyle \ln n!\approx n\ln n-n+{\tfrac {1}{6}}\ln(8n^{3}+4n^{2}+n+{\tfrac {1}{30}})+{\tfrac {1}{2}}\ln \pi .}

# In[39]:


n,k=symbols('n,k')
f01= (n**n*exp(n) )

f02=(k**k*exp(k) )
f03=( (n-k)**(n-k)*exp(n-k) )

ans=expand(f01/(f02*f03))
ans


# In[96]:


figbin = plt.figure(figsize=(6,6) )
plt.rcParams.update({'font.size': 16})  # set font size for plot


x = np.linspace(0,1,200)

#abinom = lambda n,k:  k**(-1.0*k)* n**(1.0*n) * (n-k)**(k-n)  # not good large n

def abinom(n,k):                # ok as integers not reals
    s=1
    for i in range(1,k+1,1):
        s = s*(n+1-i)//i
    return s


for n in [20,100,1000]:


    binm = special.binom(n,n/2)

    abin = special.binom(n,x*n)
    
    
    plt.plot( x, abin/binm,color='blue',linewidth=1)

#plt.annotate(r'$1000$', xy=(x[105],abin[105]/binm), xytext=(0.75,0.5),
#            arrowprops = dict(color='grey', shrink=0.01,width=1,headlength=15,headwidth=6) )    

for n in[20]:
    
    xx = np.linspace(0,1,n)
    
    binm = special.binom(n,n/2)

    abin = special.binom(n,xx*n)
    
    plt.scatter(xx,abin/binm,color='grey',s=10)

plt.scatter(xx,abin/binm,s=50,facecolor='none',edgecolor='black',zorder=10,linewidth=1)

pbig=[]
xbig=[]
for i in range(4800,5200,10):
    
    pbig.append((abinom(10000,i)/abinom(10000,10000//2) ))
    xbig.append(i/10000)
    #plt.scatter( i/10000,(abinom(10000,i)/abinom(10000,10000//2) ),s=2,color='black')

plt.plot(xbig,pbig,color='red',label=r'$n=10000$') 

#plt.annotate(r'$10000$', xy=(xbig[30],pbig[30]), xytext=(0.75,0.25),
#            arrowprops = dict(color='grey', shrink=0.01,width=1,headlength=15,headwidth=6) )    



plt.plot([0.5,0.5],[-0.05,0.05], color='black')

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel(r'$k/n$')
plt.ylabel(r'$p/p_{max}$')
plt.xticks([0,0.25,0.5,0.75,1])

plt.text(0.35,0.8,r'$n=20$',rotation =80)
plt.text(0.35,0.15,r'$100$',rotation=80)

plt.text(0.53,0.10,r'$1000$',rotation=-86)
#plt.legend()
plt.legend(fontsize=12)
plt.tight_layout()


#plt.savefig('chapter1-fig20b.png')

plt.show()


# In[46]:


def abinom(n,k):
    s=1
    for i in range(1,k+1,1):
        s = s*(n+1-i)//i
    return s
        
print( (abinom(100000,50500)/abinom(100000,100000//2) ) ) 


# In[75]:


special.binom(1000,1000//2)


# In[37]:


(np.sum(pbig))/1000


# In[98]:


n,z=symbols('z,n')
f01=z**z*exp(z)/(n**n*exp(n)*(z-n)**(z-n)*exp((z-n)))
simplify(f01 )


# In[100]:


10**(24)*np.log10(0.99)


# In[101]:


1e12*60*60*24*365


# In[ ]:




