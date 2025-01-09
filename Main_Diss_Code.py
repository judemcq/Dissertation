import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import cmath
#Have an input array of coordinates of lattice points, focussing on 2D array. 
#As well as number of steps wanting to iterate over total

#Functions for making the different movements throughout the brillouin zone, 
#as well as transforming real space coords to k space and then determining energies
def variableStepLength(posArray, N):
    stepsPerPoint = []
    length = []
    totalLength = 0

    for i in range(0,len(posArray)-1):
        length.append(np.sqrt( (posArray[i+1][0] - posArray[i][0])**2 + (posArray[i+1][1] - posArray[i][1])**2))
        totalLength = totalLength + length[i]

    stepsPerPoint = N * (length/totalLength)
    stepsPerPoint = np.around(stepsPerPoint)
    stepsPerPoint = stepsPerPoint.astype(np.int32)
    return stepsPerPoint

def reciprocalVectors(a1,a2, realSpaceCoords):
    Apc = (2*np.pi)/(a1[0]*a2[1] - a1[1]*a2[0]) #is actually 2pi/Apc but eh
    
    #These are the reciprocal lattice vectors given real lattice vectors
    b1 = [Apc*a2[1],-Apc*a2[0]]
    b2 = [-Apc* a1[1],Apc*a1[0]]
    B = np.column_stack((b1,b2))
    
    reciprocalSpaceCoords = realSpaceCoords @ B
    
    return B, reciprocalSpaceCoords

def kArray(kSpaceCoords, numberOfPoints):
    k = kSpaceCoords
    N = numberOfPoints
    stepsArray = variableStepLength(k,N)
    kxRange = []
    kyRange = []
    kxPoints = []
    kyPoints = []

    for i in range(0,len(kSpaceCoords)-1):
        xStep = k[i+1][0]-k[i][0]
        yStep = k[i+1][1]-k[i][1]

        for j in range(0,stepsArray[i]-1):
            kxRange.append(k[i][0] + (xStep/stepsArray[i])*j)
            kyRange.append(k[i][1] + (yStep/stepsArray[i])*j)
        
        kxPoints.append(kxRange)
        kyPoints.append(kyRange)


        kxRange = []
        kyRange = []
        
    return kxPoints, kyPoints, stepsArray

def DispersionFuncSquare(Kx, Ky, ax, ay, t):
    E = []
    steps = []
    E_temp = []
    for i in range(0,len(Kx)):
        for j in range(0,len(Kx[i])-1):
            E_temp.append(-2*t*(np.cos(Kx[i][j]*ax)+np.cos(Ky[i][j]*ay)))
        
        E.append(E_temp)
        E_temp = []
        if i==0:
            steps.append(np.array(range(1,len(Kx[i]))))
            max = len(Kx[i])
        else:
            steps.append(np.array(range(max,max+len(Kx[i])-1)))
            max = max + len(Kx[i])
    
    return E, steps

def dispersionFuncGraphene(Kx, Ky, a, gamma):
    E = []
    steps = []
    E_temp = []
    for i in range(0,len(Kx)):
        for j in range(0,len(Kx[i])-1):
            E_temp.append(gamma*np.sqrt(1+ 4*(np.cos((1/2)*Kx[i][j]*a)**2) + 4*np.cos((1/2)*Kx[i][j]*a)*np.cos((np.sqrt(3)/2)*Ky[i][j]*a)))
        
        E.append(E_temp)
        E_temp = []
        if i==0:
            steps.append(np.array(range(1,len(Kx[i]))))
            max = len(Kx[i])
        else:
            steps.append(np.array(range(max,max+len(Kx[i])-1)))
            max = max + len(Kx[i])
    
    return E, steps

def bandFunc(kx,E,t):
    inside_sqrt = +kx**3 + t*kx - E
    ky_pos = np.real(np.sqrt(inside_sqrt+0j))
    ky_neg = np.real(-np.sqrt(inside_sqrt+0j))
    ky_oth_pos = np.real(np.sqrt(kx**3 + t*kx + E+0j))
    ky_oth_neg = np.real(-np.sqrt(kx**3 + t*kx + E+0j))
    return ky_pos, ky_neg, ky_oth_pos, ky_oth_neg

def bandFuncMyFind(kx,E,t):
    inside_sqrt = E - t*kx - kx**3
    ky_pos = np.real(np.sqrt(inside_sqrt+0j))
    ky_neg = np.real(-np.sqrt(inside_sqrt+0j))
    ky_oth_pos = np.real(np.sqrt(-kx**3 - t*kx - E+0j))
    ky_oth_neg = np.real(-np.sqrt(-kx**3 - t*kx - E+0j))
    return ky_pos, ky_neg, ky_oth_pos, ky_oth_neg



def threeDBandForTwoAtomBasis(KX,KY,t1,t2,tprime): #Just does a 3D plot of kx,ky,E
    epsilon = []
    epsilon = 2*t1*np.cos(KX/2) - 2*tprime*np.sin(KX) + 2*t2*np.cos(KY)
    return epsilon

def threeDBandForGraphene(KX,KY,a,gamma, tprime): #Just does a 3D plot of kx,ky,E
    epsilon = []
    epsilon = gamma*np.sqrt(1+ 4*(np.cos((1/2)*KX*a)**2) + 4*np.cos((1/2)*KX*a)*np.cos((np.sqrt(3)/2)*KY*a))
    return epsilon

def threeDSquareLattice(KX,KY,a,t, tprime): #Just does a 3D plot of kx,ky,E
    epsilon = []
    epsilon = -2*t*np.cos(KX*a) -2*t*np.cos(KY*a)
    return epsilon
def dispersionFuncProj(Kx, Ky, t1, t2,tprime):
    E = []
    steps = []
    E_temp = []
    for i in range(0,len(Kx)):
        for j in range(0,len(Kx[i])-1):
            E_temp.append(2*t1*np.cos(Kx[i][j]/2) - 2*tprime*np.sin(Kx[i][j]) + 2*t2*np.cos(Ky[i][j]))
        
        E.append(E_temp)
        E_temp = []
        if i==0:
            steps.append(np.array(range(1,len(Kx[i]))))
            max = len(Kx[i])
        else:
            steps.append(np.array(range(max,max+len(Kx[i])-1)))
            max = max + len(Kx[i])
    
    return E, steps


def peierlsSquareLatticeThreeD(KX,KY,t,Bfield, a):
    epsilon = []
    epsilon = -2*t*np.cos(KX*a + Bfield*KY*a) - 2*t*np.cos(KY*a)
    return epsilon

def peierlsTwoAtomBasisLatticeThreeD(KX,KY,t1,t2,tprime):
    epsilon = []
    a=1
    phase = 0.7*a 
    
    epsilon = 2*t2*np.cos(KY*a - phase*KX) - 2*tprime*np.sin(KX*a - phase*KY) + 2*t1*np.exp(-(phase/4)*KX*KY)
    return epsilon


#Functions for plotting the different figures

#Animation plotting code
def AnimationForContourPlot(bandfuncFunction, t_values_ani, energy_levels):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Band Structure Animation")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")

    lines = []
    for _ in energy_levels:
        line_pos, = ax.plot([], [], color="red", lw=0.5)
        line_neg, = ax.plot([], [], color="red", lw=0.5)
        line_oth_pos, = ax.plot([], [], color="blue", lw=0.5)
        line_oth_neg, = ax.plot([], [], color="blue", lw=0.5)
        lines.append((line_pos, line_neg, line_oth_pos, line_oth_neg))

    t_text = ax.text(0.85, 0.95, '', transform=ax.transAxes, fontsize=10, color='black')
    def update(frame):
        t = t_values_ani[frame]
        for i, E in enumerate(energy_levels):
            ky_pos, ky_neg, ky_oth_pos, ky_oth_neg = bandfuncFunction(kx,E,t)
            lines[i][0].set_data(kx, ky_pos)  
            lines[i][1].set_data(kx, ky_neg)
            lines[i][2].set_data(kx, ky_oth_pos)  
            lines[i][3].set_data(kx, ky_oth_neg)  
            t_text.set_text(f't = {t:.3f}')
        ax.set_title(f"Band Structure Animation")
        return [line for pair in lines for line in pair] + [t_text]

    ani = FuncAnimation(fig, update, frames=len(t_values_ani), interval=100, blit=True)
    plt.show()

#Plot for contour with t = t_vals
def plotContourforT(bandfunc, t_vals, energy_levels, expansionAbout:str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Energy Contour for expansion about {expansionAbout}")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    for i in range(0,len(energy_levels)):
        kyPos, kyNeg, kyOthPos, kyOthNeg = bandfunc(kx,energy_levels[i], t_vals)
        plt.plot(kx, kyPos, color="red", linewidth=0.5)
        plt.plot(kx, kyNeg, color="red", linewidth=0.5)
        plt.plot(kx, kyOthPos, color="blue", linewidth=0.5)
        plt.plot(kx, kyOthNeg, color="blue", linewidth=0.5)
    plt.text(0.75, 0.9,f't={t_vals:.2f}', ha='left',va='top')
    plt.show()

#Line plot for movement throughout the Brillouin zone
def linePlotDispFunc(steps, E, myTitle:str): #Plots the line plot for the steps given in realPOIProj

    plt.figure()
    colorArray = ["red", "green", "blue", "pink", "purple", "orange", "black","violet"]
    ticks = [0]
    temp = 0
    for i in range(0,len(stepsArray)):
            if i==0:
                ticks.append(stepsArray[i])
            else:
                for j in range(0,i):
                    temp = temp + stepsArray[j]
                ticks.append(temp+stepsArray[i])
                temp = 0


    for i in range(0,len(steps)):
        plt.plot(steps[i][:],E[i][:],color=colorArray[i])
        plt.axvline(x=ticks[i], color='grey', linestyle='--', linewidth=1)
    plt.axvline(x=ticks[len(ticks)-1], color='grey', linestyle='--', linewidth=1)
    ksc = [[int(x), int(y)] for x, y in kSpaceCoords]
    def format_k_value(value):
        if value == 3:
            return r"$\pi$"
        elif value == -3:
            return r"$-\pi$"
        elif value == 6:
            return r"$2\pi$"
        elif value == -6:
            return r"$-2\pi$"
        elif value == 1:
            return r"$\frac{\pi}{2}$"
        elif value ==-1:
            return r"$\frac{-\pi}{2}$"
        else:
            return str(value)
    ticksarr = [f"({format_k_value(x)},{format_k_value(y)})" for x, y in ksc]

    plt.xticks(ticks, ticksarr)
    plt.xlabel("Movement through Brillouin Zone")
    plt.ylabel("Energy")
    plt.title(myTitle)
    plt.show()

#3D plot of the kx,ky,E dispersion 
def threeDDispersion(kxPoints, kyPoints, bandStructureFunc, t1, t2, tprime):
    kx = np.concatenate(kxPoints)
    ky = np.concatenate(kyPoints)
    KX, KY = np.meshgrid(kx, ky)

    simpleEpsilon = bandStructureFunc(KX,KY,t1,t2,tprime)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(KX, KY, simpleEpsilon, cmap="summer", linewidth=0, antialiased=False, alpha = 0.95)
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.title("Dispersion Relation Plot")
    fig.colorbar(surf, shrink=0.5, aspect=5)



#Defining the variables and executing functions that produce the Energy Contour Plots
energy_levels = np.linspace(0, 0.5, 25)
t_values_ani = np.linspace(-0.25, 0.25, 50) 
t_vals = 0
kx = np.linspace(-1,1,1000)


plotContourforT(bandFuncMyFind, t_vals, energy_levels, expansionAbout=r"($-\pi,\pi $)")

AnimationForContourPlot(bandFuncMyFind, t_values_ani=t_values_ani, energy_levels=energy_levels)

#Code that produces the 3D and movement plots
N = 5000
t1=0
t2=-0.2
tprime=0.1
a1Proj = [1,0]
a2Proj = [0,1]
realPOIProj = np.array([[-1/2,-1/2],[1/2,-1/2],[1/2,0],[-1/2,1/2],[0,0],[-1/2,-1/2],[0,-1/4]])

reciprocalVecs, kSpaceCoords = reciprocalVectors(a1Proj,a2Proj, realPOIProj)
kxPointsProj, kyPointsProj, stepsArray = kArray(kSpaceCoords, N)
E, steps = dispersionFuncProj(kxPointsProj, kyPointsProj, t1=t1, t2=t2, tprime=tprime)  


linePlotDispFunc(steps, E, f"Line plot for $t_1={t1}, t_2={t2},t'={tprime}$")
threeDDispersion(kxPointsProj, kyPointsProj, threeDBandForTwoAtomBasis, t1, t2, tprime)


a1Peierls = [1,0]
a2Peierls = [0,1]
N = 8000
t1=0.0
bField=1
t2 = 0.1
tprime = 0.15
realPOIPeierls = np.array([[-1,-1],[1,1]])
reciprocalVecs, kSpaceCoords = reciprocalVectors(a1Peierls,a2Peierls, realPOIPeierls)
kxPointsPeierls, kyPointsPeierls, stepsArray = kArray(kSpaceCoords, N)

threeDDispersion(kxPointsPeierls, kyPointsPeierls, threeDBandForTwoAtomBasis, t1, t2, tprime)

threeDDispersion(kxPointsPeierls, kyPointsPeierls, peierlsTwoAtomBasisLatticeThreeD, t1, t2, tprime)

plt.show()
#Doing for Original

energy_levels = np.linspace(0, 0.5, 25)
t_values_ani = np.linspace(-0.25, 0.25, 50) 
t_vals = 0
kx = np.linspace(-1,1,1000)

plotContourforT(bandFunc, t_vals, energy_levels, expansionAbout=r"($\pi,0 $)")
AnimationForContourPlot(bandFunc, t_values_ani=t_values_ani, energy_levels=energy_levels)

N = 5000
t1=0.0
t2=-0.2
tprime=0.1
a1Proj = [1,0]
a2Proj = [0,1]
realPOIProj = np.array([[-1,-1],[1/2,1/4],[1/2,-1/2],[0,0],[1/2,1/4],[-1/4,-1/4],[1,1]])

reciprocalVecs, kSpaceCoords = reciprocalVectors(a1Proj,a2Proj, realPOIProj)
kxPointsProj, kyPointsProj, stepsArray = kArray(kSpaceCoords, N)
#E, steps = dispersionFuncProj(kxPointsProj, kyPointsProj, t1=t1, t2=t2, tprime=tprime)  


#linePlotDispFunc(steps, E, myTitle = f"Line plot for $t_1={t1}, t_2={t2},t'={tprime}$")
threeDDispersion(kxPointsProj, kyPointsProj, t1, t2, tprime)
