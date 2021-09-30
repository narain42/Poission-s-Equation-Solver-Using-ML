import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
import math
import sys
sys.path.insert(0, '../modules')
from time import process_time

## program notes
## Matrix Inv for 2D , Gradient Descent and Conjugate Gradient for 2D,3D, and 4D problems with zero Dirichlet boundary conditions only.
## Jacobi and Gauss Siedel for polar, 2D, 3D and 4D problems with any kind of b.c.

t2_start = process_time()
method = input("Method to solve, 1 Matrx Inv, 2 Jacobi, 3 Guass Sidel, 4 Gradient Descent, 5 Conjgate Gradient :")
method = int(method)
N = input("Input number of points in the grid,like 41, modify code for different grid points :")
N = int (N)
# Grid parameters.
nx = N                   # number of points in the x direction
ny = N                 # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = 0.0, 1.0    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx - 1)        # grid spacing in the x direction
dy = ly / (ny - 1)        # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
p = np.empty((nx, ny))
submethod = 0
pi = np.pi

submethod = input("Dimensions to solve, 1 for polar, 2 Two-Dim, 3 Three-Dim, 4 four-Dim :")
submethod = int(submethod)

if submethod == 1:
    p0 = np.zeros((nx, ny))
    pxx = np.zeros((nx, ny))
    pyy = np.zeros((nx, ny))
    px = np.zeros((nx,ny))
    p = np.empty((nx, ny))
    
    xmin, xmax = 1.0, 60.0     # limits in the x direction
    ymin, ymax = -pi, pi    # limits in the y direction
    lx = xmax - xmin          # domain length in the x direction
    ly = ymax - ymin          # domain length in the y direction
    dx = lx / (nx - 1)        # grid spacing in the x direction
    dy = ly / (ny - 1)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
elif submethod == 2:
    p0 = np.zeros((nx, ny))
    pxx = np.zeros((nx, ny))
    pyy = np.zeros((nx, ny))
    py = np.zeros((nx, ny))
    p = np.empty((nx, ny))
    
elif submethod == 3:
    nz = N     
    zmin, zmax = 0.0, 1.0  
    lz = zmax - zmin          
    dz = lz / (nz - 1)       
    z = np.linspace(zmin, zmax, nz)
    p = np.empty((nx, ny,nz))
    p0 = np.zeros((nx, ny,nz))
    pxx = np.zeros((nx, ny,nz))
    pyy = np.zeros((nx, ny,nz))
    pzz = np.zeros((nx, ny,nz))
    p = np.empty((nx, ny,nz))
    X, Y, Z = np.meshgrid(x, y, z)
else:
    nz = N     
    zmin, zmax = 0.0, 1.0  
    lz = zmax - zmin          
    dz = lz / (nz - 1)
    z = np.linspace(zmin, zmax, nz)
    nt = N     
    tmin, tmax = 0.0, 1.0  
    lz1 = tmax - tmin          
    dt = lz1 / (nt - 1)
    t = np.linspace(tmin, tmax, nt)
    p0 = np.zeros((nx, ny,nz,nt))
    pxx = np.zeros((nx, ny,nz,nt))
    pyy = np.zeros((nx, ny,nz,nt))
    pzz = np.zeros((nx, ny,nz,nt))
    ptt = np.zeros((nx, ny,nz,nt))
    p = np.empty((nx, ny, nz, nt))
    X, Y, Z, T = np.meshgrid(x, y, z, t)

def plotSurface( XX, YY, ZZ, title, zLabel=r"$\phi$" ):
	"""
 	Plot a 3D surface.
	:param XX: Grid of x coordinates.
	:param YY: Grid of y coordinates.
	:param ZZ: Grid of z values.
	:param title: Figure title.
	:param zLabel: Label for z-axis.
	"""
	fig1 = plt.figure()
	ax = fig1.gca( projection='3d' )
	surf = ax.plot_surface( XX, YY, ZZ, cmap=cm.jet, linewidth=0, antialiased=False, rstride=1, cstride=1 )
	fig1.colorbar( surf, shrink=0.5, aspect=5 )
	ax.set_xlabel( r"$x_1$" )
	ax.set_ylabel( r"$x_2$" )
	ax.set_zlabel( zLabel )
	plt.title( title )
	plt.show()
	
def p_exact_2d(X, Y):
    """Computes the exact solution of the Poisson equation in the domain
    [0, 1]x[-0.5, 0.5] with rhs:
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
    np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))

    Parameters
    ----------
    X : numpy.ndarray
        array of x coordinates for all grid points
    Y : numpy.ndarray
        array of y coordinates for all grid points

    Returns
    -------
    sol : numpy.ndarray
        exact solution of the Poisson equation
    """
    if submethod == 2:
      
##    sol = (-1.0/(2.0*np.pi**2)*np.sin(np.pi*X)*np.cos(np.pi*Y)
##        - 1.0/(50.0*np.pi**2)*np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))
##      sol = np.exp(-X)*(X + Y **3)
      sol = np.sin(np.pi*X)*np.sin(np.pi*Y)
##      sol= (2.- (pi**2)*Y**2)*np.sin(pi*X)
##      sol=  Y**2*np.sin(pi*X)
    elif submethod == 3:
##        sol = np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
        sol = np.exp(-X)*(X + Y **3 + Z**3)
    elif submethod == 4:
##        sol = np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)*np.sin(np.pi*T)
        sol = np.exp(-X)*(X + Y **3 + Z**3 + T**3)
    else:
      sol = (X+1./X)*np.cos(Y)
    return sol

def rhs_2d(X, Y):
##    sol =  np.exp(-X)*(X -2. + Y **3 + 6.*Y)
##    sol= (2.- (pi**2)*Y**2)*np.sin(pi*X)
    sol =  -2.*(np.pi**2)*np.sin(np.pi*X)*np.sin(np.pi*Y)
    return sol

def rhs3d_func(X, Y, Z):
##    sol = -3.*(pi**2)*np.sin(pi*X)*np.sin(pi*Y)*np.sin(pi*Z)
    sol =  np.exp(-X)*(X -2. + Y **3 + 6.*Y + Z**3 + 6.*Z)
    return sol

def rhs4d_func(X, Y, Z,T):
##    sol = -4.*(pi**2)*np.sin(pi*X)*np.sin(pi*Y)*np.sin(pi*Z)*np.sin(pi*T)
    sol =  np.exp(-X)*(X -2. + Y **3 + 6.*Y + Z**3 + 6.*Z + T**3 + 6.*T)
    return sol



def A(v):
        """
        Computes the action of (-) the Poisson operator on any
        vector v_{ij} for the interior grid nodes
        
        Parameters
        ----------
        v : numpy.ndarray
            input vector
        dx : float
             grid spacing in the x direction
        dy : float
            grid spacing in the y direction
            

        Returns
        -------
        Av : numpy.ndarray
            action of A on v
        """
        if submethod == 2:
            Av = -((v[:-2, 1:-1]-2.0*v[1:-1, 1:-1]+v[2:, 1:-1])/dx**2 
               + (v[1:-1, :-2]-2.0*v[1:-1,1:-1]+v[1:-1, 2:])/dy**2)

        elif submethod == 3:
            Av = -((v[:-2,1:-1, 1:-1]-2.0*v[1:-1,1:-1, 1:-1]+v[2:,1:-1, 1:-1])/dx**2 \
                   + (v[1:-1, :-2, 1:-1]-2.0*v[1:-1,1:-1,1:-1]+v[1:-1, 2:,1:-1])/dy**2 \
                   + (v[1:-1, 1:-1, :-2 ]-2.0*v[1:-1,1:-1,1:-1]+v[1:-1, 1:-1, 2:])/dz**2 )
        elif submethod == 4:
            Av = -((v[:-2,1:-1, 1:-1,1:-1]-2.0*v[1:-1,1:-1, 1:-1, 1:-1]+v[2:,1:-1, 1:-1,1:-1])/dx**2 \
                   + (v[1:-1, :-2, 1:-1,1:-1]-2.0*v[1:-1,1:-1,1:-1,1:-1]+v[1:-1, 2:,1:-1,1:-1])/dy**2 \
                   + (v[1:-1, 1:-1, :-2,1:-1 ]-2.0*v[1:-1,1:-1,1:-1,1:-1]+v[1:-1, 1:-1, 2:,1:-1])/dz**2 \
                   + (v[1:-1, 1:-1, 1:-1, :-2 ]-2.0*v[1:-1,1:-1,1:-1,1:-1]+v[1:-1, 1:-1, 1:-1, 2:])/dt**2 )   
        return Av

X, Y = np.meshgrid(x, y, indexing='ij')
# Create the source term.
b = rhs_2d(X, Y)

# Compute the exact solution.
p_exact = p_exact_2d(X, Y)


def l2_diff(p1,p2):
  return np.sqrt(abs(np.sum(np.power(p1, 2) - np.power(p2 ,2 ) ) )) 

## Direct matrix inversion method 1
if method == 1:
  bflat = b[1:-1, 1:-1].flatten('F')
# Allocate array for the (full) solution, including boundary values
  p = np.empty((nx, ny))

  def d2_mat_dirichlet_2d(nx, ny, dx, dy):
    """
    Constructs the matrix for the centered second-order accurate
    second-order derivative for Dirichlet boundary conditions in 2D

    Parameters
    ----------
    nx : integer
        number of grid points in the x direction
    ny : integer
        number of grid points in the y direction
    dx : float
        grid spacing in the x direction
    dy : float
        grid spacing in the y direction

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions
    """
    a = 1.0 / dx**2
    g = 1.0 / dy**2
    c = -2.0*a - 2.0*g

    diag_a = a * np.ones((nx-2)*(ny-2)-1)
    diag_a[nx-3::nx-2] = 0.0
    diag_g = g * np.ones((nx-2)*(ny-3))
    diag_c = c * np.ones((nx-2)*(ny-2))

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-(nx-2), -1, 0, 1, nx-2]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    d2mat = diags(diagonals, offsets).toarray()

    # Return the final array
    return d2mat

  A = d2_mat_dirichlet_2d(nx, ny, dx, dy)
  Ainv = np.linalg.inv(A)
  # The numerical solution is obtained by performing
# the multiplication A^{-1}*b. This returns a vector
# in column-major ordering. To convert it back to a 2D array
# that is of the form p(x,y) we pass it immediately to
# the reshape function.
# For more info:
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
#
# Note that we have specified the array dimensions nx-2,
# ny-2 and passed 'F' as the value for the 'order' argument.
# This indicates that we are working with a vector in column-major order
# as standard in the Fortran programming language.
  pvec = np.reshape(np.dot(Ainv, bflat), (nx-2, ny-2), order='F')

# Construct the full solution and apply boundary conditions
  p[1:-1, 1:-1] = pvec
  p[0, :] = 0
  p[-1, :] = 0
  p[:, 0] = 0
  p[:, -1] = 0

  diff = l2_diff(p,p_exact)
  print(f'The l2 difference between the computed solution '
      f'and the exact solution is:\n{diff}')
  exact_norm = np.sqrt(abs(np.sum(p_exact)))
  percent_accuracy = 100.0 - (diff/exact_norm)*100.0
  print('percentage_accuracy', percent_accuracy)


elif method == 2:   ## jacobi iteration


    tolerance = 1e-10
    max_it = 100000
    
    it = 0 # iteration counter
    diff = 1.0
    tol_hist_jac = []
    pnew = p0.copy()
    
    while (diff > tolerance):
        if it > max_it:
            print('\nSolution did not converged within the maximum'
                  ' number of iterations'
                  f'\nLast l2_diff was: {diff:.5e}')
            break
        else:
            if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
        np.copyto(p, pnew)
        aa = 1.0
        
        if submethod == 1:
           
                rx2 = np.dot(x[1:-1],x[1:-1])
                rx1 = math.sqrt(rx2)
                denom = 2.*(dx**2 + dy**2*rx2)
                pyy[1:-1,1:-1] = ((p[1:-1,2:] + p[1:-1,:-2])*dx**2 )
                pxx[1:-1,1:-1] = ( (p[2:,1:-1] + p[:-2,1:-1])*dy**2*rx2 )
                px[1:-1,1:-1] = ( (p[2:,1:-1] - p[:-2, 1:-1])*dx**2*dy**2*rx2 )/ (2.*rx1*dx)
                pnew[1:-1,1:-1] = (pxx[1:-1,1:-1] + pyy[1:-1,1:-1] + px[1:-1,1:-1]) / denom

##                pnew[:,0] =  -(x+1./x)
##                pnew[:, ny-1] = -(x+1./x)
##                pnew[0, :] = 2.*np.cos(y)
##                pnew[nx-1, :] = (x[-1] +1./x[-1])*np.cos(y)

                pnew[:,0] = pnew[:, ny-2]  ## cyclic bc
                pnew[:,ny-1] = pnew[:,1]  ## cyclic bc
                pnew[0,:] = 2.*np.cos(y)  ##dirichlet bc
                pnew[nx-1,:] = pnew[nx-2,:]+ dx*np.cos(y)  ##Neumann bc
                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1
        elif submethod == 2:
                # We only modify interior nodes. The boundary nodes remain equal to
                # zero and the Dirichlet boundary conditions are therefore automatically
                # enforced.
                
                denom = 2*(dx**2 + dy**2)
                pxx[1:-1,1:-1] = ((p[1:-1,2:] + p[1:-1,:-2])*dy**2 )/ denom
                pyy[1:-1,1:-1] = ( (p[2:,1:-1] + p[:-2,1:-1])*dx**2)/ denom

                pnew[1:-1, 1:-1] = pxx[1:-1,1:-1] + pyy[1:-1,1:-1] - (b[1:-1,1:-1]*dx**2 * dy**2) / (denom)
##                pnew[:,0] =  x*np.exp(-x)
##                pnew[:, ny-1] = np.exp(-x)*(x + 1.)
##                pnew[0, :] = y**3 
##                pnew[nx-1, :] = (1. + y**3)*np.exp(-1.)
                pnew[:,0] =  aa*x*np.exp(-x)
                pnew[:, ny-1] =aa*np.exp(-x)*(x + 1.)
##                pnew[:, ny-1] = pnew[:, ny-2] + 2.* np.sin(pi*x)*dy  ## Neumann bc for ny-1
                pnew[0, :] = aa*y**3 
                pnew[nx-1, :] = aa*(1. + y**3)*np.exp(-1.)
              

                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1
        elif submethod == 3:
                rhs = rhs3d_func(X, Y, Z)
                pzz[1:-1,1:-1,1:-1] = ( (p[2:,1:-1,1:-1] + p[:-2,1:-1,1:-1])*dx**2*(dy**2))/ (2.*(dx**2*dy**2 + dy**2*dz**2+dz**2*dx**2))
                pyy[1:-1,1:-1,1:-1] = ((p[1:-1,2:,1:-1] + p[1:-1,:-2,1:-1])*dx**2*(dz**2) )/ (2.*(dx**2*dy**2 + dy**2*dz**2+dz**2*dx**2))
                pxx[1:-1,1:-1,1:-1] = ( (p[1:-1:,1:-1,2:] + p[1:-1,1:-1,:-2])*dy**2*(dz**2))/ (2.*(dx**2*dy**2 + dy**2*dz**2+dz**2*dx**2))
                pnew[1:-1,1:-1, 1:-1] = pxx[1:-1,1:-1,1:-1] + pyy[1:-1,1:-1,1:-1] + pzz[1:-1,1:-1,1:-1] \
                                     - (rhs[1:-1,1:-1,1:-1]*(dx**2 )* (dy**2)* (dz**2)) /(2.*(dx**2*dy**2 + dy**2*dz**2+dz**2*dx**2))
                #Boundary Conditions
                pnew[0,:,:] = aa*np.exp(-x)*(x + y**3)
                pnew[nz-1,:,:]= aa*np.exp(-x)*(x + 1. + y**3)

                pnew[:,0,:] = aa*np.exp(-x)*(x + z**3)
                pnew[:,ny-1,:] = aa*np.exp(-x)*(x + 1. + z**3)

                pnew[:,:,0] = aa*(y**3+z**3)
                pnew[:,:,nx-1] = aa*np.exp(-1)*(1. + y**3+z**3)

                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1

        else:
                rhs = rhs4d_func(X, Y, Z, T)
                denom = (dx**2)*(dy**2)*dz**2 + (dx**2)*(dy**2)*dt**2 +  (dx**2)*(dz**2)*dt**2 + (dy**2)*(dz**2)*dt**2 
                ptt[1:-1,1:-1,1:-1,1:-1] = ((p[2:,1:-1,1:-1,1:-1] + p[:-2,1:-1,1:-1,1:-1])*dx**2*(dy**2)*dz**2)/ (2.*denom)
                pzz[1:-1,1:-1,1:-1,1:-1] = ((p[1:-1,2:,1:-1,1:-1] + p[1:-1,:-2,1:-1,1:-1])*dx**2*(dy**2)*dt**2)/ (2.*denom)
                pyy[1:-1,1:-1,1:-1,1:-1] = ((p[1:-1,1:-1,2:,1:-1] + p[1:-1,1:-1,:-2,1:-1])*dx**2*(dz**2)*dt**2)/ (2.*denom)
                pxx[1:-1,1:-1,1:-1,1:-1] = ((p[1:-1:,1:-1,1:-1,2:]+ p[1:-1,1:-1,1:-1,:-2])*dy**2*(dz**2)*dt**2)/ (2.*denom)
                pnew[1:-1,1:-1, 1:-1,1:-1] = pxx[1:-1,1:-1,1:-1,1:-1] + pyy[1:-1,1:-1,1:-1,1:-1] \
                                        + pzz[1:-1,1:-1,1:-1,1:-1] + ptt[1:-1,1:-1,1:-1,1:-1] \
                                        - (rhs[1:-1,1:-1,1:-1,1:-1]*dx**2*(dy**2)*(dz**2)*dt**2) /(2.*denom)
                #Boundary Condition
                pnew[:,:,:,0] = aa*np.exp(-x)*(x + y**3+z**3)
                pnew[:,:,:,nt-1]= aa*np.exp(-x)*(x + y**3+z**3 + 1.)

                pnew[:,:,0,:] = aa*np.exp(-x)*(x + y**3+t**3)
                pnew[:,:,nz-1,:] = aa*np.exp(-x)*(x + y**3 + 1. +t**3)

                pnew[:,0,:,:] = aa*np.exp(-x)*(x + z**3+t**3)
                pnew[:,ny-1,:,:] = aa*np.exp(-x)*(x + 1. + z**3+t**3)
                
                pnew[0,:,:,:] = aa*np.exp(-x)*(y**3 + z**3+t**3)
                pnew[nx-1,:,:,:] = aa*np.exp(-1)*(1. +y**3 + z**3+t**3)
                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1
    else:
        print(f'\nThe solution converged after {it} iterations')
        p = pnew
    
    diff = l2_diff(pnew, p_exact)
    print(f'The l2 difference between the computed solution and '
            f'the exact solution is:\n{diff}')
    ex_norm = np.linalg.norm(p_exact)
    p_norm = np.linalg.norm(pnew)
    norm_diff = ex_norm-p_norm
    print(ex_norm, p_norm, norm_diff)
    percent_accuracy = 100.0 *(1. - (abs(ex_norm-p_norm)/ex_norm))
    print('linalg percentage_accuracy', percent_accuracy)

elif method == 3:   ## Gauss-Sidel iteration

    tolerance = 1e-10
    max_it = 10000

##    p0 = np.zeros((nx, ny))
    pnew = p0.copy()

    it = 0 # iteration counter
    diff = 1.0
    tol_hist_jac = []
    aa = 1.0
    while (diff > tolerance):
        if it > max_it:
            print('\nSolution did not converged within the maximum'
                  ' number of iterations'
                  f'\nLast l2_diff was: {diff:.5e}')
            break
        else:
            if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
        np.copyto(p, pnew)
        if submethod == 1:
## b = 0 for this problem           
                rx2 = np.dot(x[1:-1],x[1:-1])
                rx1 = math.sqrt(rx2)
                denom = 2.*(dx**2 + dy**2*rx2)
                for j in range(1, ny-1):
                  for i in range(1, nx-1):
                      pnew[i, j] = ((pnew[i-1, j]+p[i+1, j])*dy**2*rx2 + (pnew[i, j-1] + p[i, j+1])*dx**2  \
                                   +(pnew[i-1, j] - p[i+1, j])*(dy**2*rx2*dx**2)/(2.*rx1*dx))/denom \
                              -(0.0*b[i,j]*dx**2 * dy**2) / (denom)

                pnew[:,0] =  -(x+1./x)
                pnew[:, ny-1] = -(x+1./x)
                pnew[0, :] = 2.*np.cos(y)
                pnew[nx-1, :] = (x[-1] +1./x[-1])*np.cos(y)

##                pnew[:,0] = pnew[:, ny-2]  ## cyclic bc
##                pnew[:,ny-1] = pnew[:,1]  ## cyclic bc
##                pnew[0,:] = 2.*np.cos(y)  ##dirichlet bc
##                pnew[nx-1,:] = pnew[nx-2,:]+ dx*np.cos(y)  ##Neumann bc
                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1
        elif submethod == 2:
                # We only modify interior nodes. The boundary nodes remain equal to
                # zero and the Dirichlet boundary conditions are therefore automatically
                # enforced.
##                denom = 2*(dx**2 + dy**2)
##                for j in range(1, ny-1):
##                  for i in range(1, nx-1):
##                      pnew[i, j] = ((pnew[i-1, j]+p[i+1, j])*dy**2 + (pnew[i, j-1] + p[i, j+1])*dx**2 )/ denom \
##                              -(b[i,j]*dx**2 * dy**2) / (denom)
                denom = 2*(dx**2 + dy**2)
                pxx[1:-1,1:-1] = ((p[1:-1,2:] + pnew[1:-1,:-2])*dy**2 )/ denom
                pyy[1:-1,1:-1] = ( (p[2:,1:-1] + pnew[:-2,1:-1])*dx**2)/ denom
                pnew[1:-1, 1:-1] = pxx[1:-1,1:-1] + pyy[1:-1,1:-1] - (b[1:-1,1:-1]*dx**2 * dy**2) / (denom)

                pnew[:,0] =  aa*x*np.exp(-x)
                pnew[:, ny-1] = aa*np.exp(-x)*(x + 1.)
                pnew[0, :] = aa*y**3 
                pnew[nx-1, :] = aa*(1. + y**3)*np.exp(-1.)
                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1
        elif submethod == 3:
                rhs = rhs3d_func(X, Y, Z)
                denom = 2.*(dx**2*dy**2 + dy**2*dz**2+dz**2*dx**2)
##                for k in range(1, nz-1):
##                  for j in range(1, ny-1):
##                    for i in range(1, nx-1):
##                       pzz[i, j, k] = ( (pnew[i,j,k-1] + p[i,j,k+1])*dx**2*(dy**2))/ denom
##                       pyy[i,j,k] = ((pnew[i, j-1, k] + p[i, j+1,k])*dx**2*(dz**2) )/ denom
##                       pxx[i,j,k] = ( (pnew[i-1,j,k] + p[i+1,j,k])*dy**2*(dz**2))/ denom
##                       pnew[i,j,k] = pxx[i,j,k] + pyy[i,j,k] + pzz[i,j,k] \
##                                     - (rhs[i,j,k]*(dx**2 )* (dy**2)* (dz**2)) /denom

                pzz[1:-1,1:-1,1:-1] = ( (p[2:,1:-1,1:-1] + pnew[:-2,1:-1,1:-1])*dx**2*(dy**2))/ denom
                pyy[1:-1,1:-1,1:-1] = ((p[1:-1,2:,1:-1] + pnew[1:-1,:-2,1:-1])*dx**2*(dz**2) )/ denom
                pxx[1:-1,1:-1,1:-1] = ( (p[1:-1:,1:-1,2:] + pnew[1:-1,1:-1,:-2])*dy**2*(dz**2))/ denom
                pnew[1:-1,1:-1, 1:-1] = pxx[1:-1,1:-1,1:-1] + pyy[1:-1,1:-1,1:-1] + pzz[1:-1,1:-1,1:-1] \
                                     - (rhs[1:-1,1:-1,1:-1]*(dx**2 )* (dy**2)* (dz**2)) /denom

                #Boundary Conditions
##                pnew[0,:,:] = 0.
##                pnew[nz-1,:,:]= 0.
##
##                pnew[:,0,:] = 0.
##                pnew[:,ny-1,:] = 0.
##
##                pnew[:,:,0] = 0.
##                pnew[:,:,nx-1] = 0.

                pnew[0,:,:] = aa*np.exp(-x)*(x + y**3)
                pnew[nz-1,:,:]= aa*np.exp(-x)*(x + 1. + y**3)

                pnew[:,0,:] = aa*np.exp(-x)*(x + z**3)
                pnew[:,ny-1,:] = aa*np.exp(-x)*(x + 1. + z**3)

                pnew[:,:,0] = aa*(y**3+z**3)
                pnew[:,:,nx-1] = aa*np.exp(-1)*(1. + y**3+z**3)
                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1

        else:
                rhs = rhs4d_func(X, Y, Z, T)
                denom = (dx**2)*(dy**2)*dz**2 + (dx**2)*(dy**2)*dt**2 +  (dx**2)*(dz**2)*dt**2 + (dy**2)*(dz**2)*dt**2
##                for t in range(1, nt-1):
##                  for k in range(1, nz-1):
##                    for j in range(1, ny-1):
##                      for i in range(1, nx-1):
##                        ptt[i, j, k, t] = ((pnew[i,j,k,t-1]  + p[i,j,k,t+1])*dx**2*(dy**2)*dz**2)/ (2.*denom)
##                        pzz[i, j, k, t] = ((pnew[i,j,k-1,t]  + p[i,j,k+1,t])*dx**2*(dy**2)*dt**2)/ (2.*denom)
##                        pyy[i,j,k,t]    = ((pnew[i, j-1,k,t] + p[i,j+1,k,t])*dx**2*(dz**2)*dt**2)/ (2.*denom)
##                        pxx[i,j,k,t]    = ((pnew[i-1,j,k,t]  + p[i+1,j,k,t])*dy**2*(dz**2)*dt**2)/ (2.*denom)
##                        pnew[i,j,k,t] = pxx[i,j,k,t] + pyy[i,j,k,t] + pzz[i,j,k,t] + ptt[i, j, k, t] \
##                                     - (rhs[i,j,k,t]*(dx**2 )* (dy**2)* (dz**2)*dt**2) /(2.*denom) 

                denom = (dx**2)*(dy**2)*dz**2 + (dx**2)*(dy**2)*dt**2 +  (dx**2)*(dz**2)*dt**2 + (dy**2)*(dz**2)*dt**2 
                ptt[1:-1,1:-1,1:-1,1:-1] = ((p[2:,1:-1,1:-1,1:-1] + pnew[:-2,1:-1,1:-1,1:-1])*dx**2*(dy**2)*dz**2)/ (2.*denom)
                pzz[1:-1,1:-1,1:-1,1:-1] = ((p[1:-1,2:,1:-1,1:-1] + pnew[1:-1,:-2,1:-1,1:-1])*dx**2*(dy**2)*dt**2)/ (2.*denom)
                pyy[1:-1,1:-1,1:-1,1:-1] = ((p[1:-1,1:-1,2:,1:-1] + pnew[1:-1,1:-1,:-2,1:-1])*dx**2*(dz**2)*dt**2)/ (2.*denom)
                pxx[1:-1,1:-1,1:-1,1:-1] = ((p[1:-1:,1:-1,1:-1,2:]+ pnew[1:-1,1:-1,1:-1,:-2])*dy**2*(dz**2)*dt**2)/ (2.*denom)
                pnew[1:-1,1:-1, 1:-1,1:-1] = pxx[1:-1,1:-1,1:-1,1:-1] + pyy[1:-1,1:-1,1:-1,1:-1] \
                                        + pzz[1:-1,1:-1,1:-1,1:-1] + ptt[1:-1,1:-1,1:-1,1:-1] \
                                        - (rhs[1:-1,1:-1,1:-1,1:-1]*dx**2*(dy**2)*(dz**2)*dt**2) /(2.*denom)
                #Boundary Condition
                pnew[:,:,:,0] = aa*np.exp(-x)*(x + y**3+z**3)
                pnew[:,:,:,nt-1]= aa*np.exp(-x)*(x + y**3+z**3 + 1.)

                pnew[:,:,0,:] = aa*np.exp(-x)*(x + y**3+t**3)
                pnew[:,:,nz-1,:] = aa*np.exp(-x)*(x + y**3 + 1. +t**3)

                pnew[:,0,:,:] = aa*np.exp(-x)*(x + z**3+t**3)
                pnew[:,ny-1,:,:] = aa*np.exp(-x)*(x + 1. + z**3+t**3)
                
                pnew[0,:,:,:] = aa*np.exp(-x)*(y**3 + z**3+t**3)
                pnew[nx-1,:,:,:] = 1.*np.exp(-1)*(1. +y**3 + z**3+t**3)



                
                diff = l2_diff(pnew, p)
                tol_hist_jac.append(diff)

                it += 1
    else:
        print(f'\nThe solution converged after {it} iterations')

        p = pnew
    diff = l2_diff(pnew, p_exact)
    print(f'The l2 difference between the computed solution and '
            f'the exact solution is:\n{diff}')
    ex_norm = np.linalg.norm(p_exact)
    p_norm = np.linalg.norm(pnew)
    norm_diff = ex_norm-p_norm
    print(ex_norm, p_norm, norm_diff)
    percent_accuracy = 100.0 *(1. - (abs(ex_norm-p_norm)/ex_norm))
    print('linalg percentage_accuracy', percent_accuracy)

elif method == 4:   ## Gradient-Descent iteration
    
    tolerance = 1e-3
    max_it = 20000


    it = 0 # iteration counter
    diff = 1.0
    tol_hist_jac = []
    pnew = p0.copy()
    # Initial guess
##    p0 = np.zeros((nx, ny))
    if submethod == 2: 
        # Place holders for the residual r and A(r)
        r = np.zeros((nx, ny))
        Ar = np.zeros((nx, ny))
        p = p0.copy()

        while (diff > tolerance):
            if it > max_it:
                print('\nSolution did not converged within the maximum'
                      ' number of iterations'
                      f'\nLast l2_diff was: {diff:.5e}')
                break
            else:
              if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
            # Residual
            r[1:-1, 1:-1] = -b[1:-1, 1:-1] - A(p)
            # Laplacian of the residual
            Ar[1:-1, 1:-1] = A(r)
            # Magnitude of jump
            alpha = np.sum(r*r) / np.sum(r*Ar)
            # Iterated solution
            pnew = p + alpha*r

            diff = l2_diff(pnew, p)
            tol_hist_jac.append(diff)
            
            # Get ready for next iteration
            it += 1
            np.copyto(p, pnew)
            
    elif submethod == 3: 
        # Place holders for the residual r and A(r)
        r = np.zeros((nx, ny, nz))
        Ar = np.zeros((nx, ny, nz))
        p = p0.copy()
        b = rhs3d_func(X,Y,Z)
        while (diff > tolerance):
            if it > max_it:
                print('\nSolution did not converged within the maximum'
                      ' number of iterations'
                      f'\nLast l2_diff was: {diff:.5e}')
                break
            else:
              if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
            # Residual
            r[1:-1, 1:-1, 1:-1] = -b[1:-1, 1:-1, 1:-1] - A(p)
            # Laplacian of the residual
            Ar[1:-1, 1:-1, 1:-1 ] = A(r)
            # Magnitude of jump
            alpha = np.sum(r*r) / np.sum(r*Ar)
            # Iterated solution
            pnew = p + alpha*r

            diff = l2_diff(pnew, p)
            tol_hist_jac.append(diff)
            
            # Get ready for next iteration
            it += 1
            np.copyto(p, pnew)
    elif submethod == 4: 
        # Place holders for the residual r and A(r)
        r = np.zeros((nx, ny, nz, nt))
        Ar = np.zeros((nx, ny, nz, nt))
        p = p0.copy()
        b = rhs4d_func(X,Y,Z,T)
        while (diff > tolerance):
            if it > max_it:
                print('\nSolution did not converged within the maximum'
                      ' number of iterations'
                      f'\nLast l2_diff was: {diff:.5e}')
                break
            else:
              if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
            # Residual
            r[1:-1, 1:-1, 1:-1, 1:-1 ] = -b[1:-1, 1:-1, 1:-1, 1:-1] - A(p)
            # Laplacian of the residual
            Ar[1:-1, 1:-1, 1:-1, 1:-1] = A(r)
            # Magnitude of jump
            alpha = np.sum(r*r) / np.sum(r*Ar)
            # Iterated solution
            pnew = p + alpha*r

            diff = l2_diff(pnew, p)
            tol_hist_jac.append(diff)
            
            # Get ready for next iteration
            it += 1
            np.copyto(p, pnew)
    else:
        print(f'\nThe solution converged after {it} iterations')
    p = pnew
    diff = l2_diff(pnew, p_exact)
    print(f'The l2 difference between the computed solution and '
            f'the exact solution is:\n{diff}')
    ex_norm = np.linalg.norm(p_exact)
    p_norm = np.linalg.norm(pnew)
    norm_diff = ex_norm-p_norm
    print(ex_norm, p_norm, norm_diff)
    percent_accuracy = 100.0 *(1. - (abs(ex_norm-p_norm)/ex_norm))
    print('linalg percentage_accuracy', percent_accuracy)
    
elif method == 5:   ## Conjugate-Gradient

    tolerance = 1e-10
    max_it = 1000


    it = 0 # iteration counter
    diff = 1.0
    tol_hist_jac = []
    pnew = p0.copy()

    # Initial guess
##    p0 = np.zeros((nx, ny))
    if submethod == 2: 
        # Place holders for the residual r and A(r)
        r = np.zeros((nx, ny))
        Ad = np.zeros((nx, ny))
        p = p0.copy()
##        print ('p', p)
        # Initial residual r0 and initial search direction direction

        r[1:-1, 1:-1] = -b[1:-1, 1:-1] - A(p) 
        d = r.copy()

        while (diff > tolerance):
            if it > max_it:
                print('\nSolution did not converged within the maximum'
                      ' number of iterations'
                      f'\nLast l2_diff was: {diff:.5e}') 
                break
            else:
              if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
                    # Laplacian of the search direction.
            Ad[1:-1, 1:-1] = A(d)
            # Magnitude of jump.
            alpha = np.sum(r*r) / np.sum(d*Ad)
            # Iterated solution
            pnew = p + alpha*d
            # Intermediate computation
            beta_denom = np.sum(r*r)
            # Update the residual.
            r = r - alpha*Ad
            # Compute beta
            beta = np.sum(r*r) / beta_denom
            # Update the search direction.
            d = r + beta*d
            pnew[:, ny-1] = pnew[:, ny-2] + 2.* np.sin(pi*x)*dy
            diff = l2_diff(pnew, p)
            tol_hist_jac.append(diff)
            
            # Get ready for next iteration
            it += 1
            np.copyto(p, pnew)
            
  
    elif submethod == 3: 
        # Place holders for the residual r and A(r)
        r = np.zeros((nx, ny, nz))
        Ad = np.zeros((nx, ny, nz))
        p = p0.copy()
        b = rhs3d_func(X,Y,Z)
        # Initial residual r0 and initial search direction d0
        r[1:-1, 1:-1, 1:-1] = -b[1:-1, 1:-1, 1:-1] - A(p)
        d = r.copy()

        while (diff > tolerance):
            if it > max_it:
                print('\nSolution did not converged within the maximum'
                      ' number of iterations'
                      f'\nLast l2_diff was: {diff:.5e}')
                break
            else:
              if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
            
            # Laplacian of the residual
            Ad[1:-1, 1:-1, 1:-1 ] = A(d)
            # Magnitude of jump.
            alpha = np.sum(r*r) / np.sum(d*Ad)
            # Iterated solution
            pnew = p + alpha*d
            # Intermediate computation
            beta_denom = np.sum(r*r)
            # Update the residual.
            r = r - alpha*Ad
            # Compute beta
            beta = np.sum(r*r) / beta_denom
            # Update the search direction.
            d = r + beta*d

            diff = l2_diff(pnew, p)
            tol_hist_jac.append(diff)
            
            # Get ready for next iteration
            it += 1
            np.copyto(p, pnew)
            
    elif submethod == 4: 
        # Place holders for the residual r and A(r)
        r = np.zeros((nx, ny, nz, nt))
        Ad = np.zeros((nx, ny, nz, nt))
        p = p0.copy()
        b = rhs4d_func(X,Y,Z,T)
        # Initial residual r0 and initial search direction d0
        r[1:-1, 1:-1, 1:-1, 1:-1] = -b[1:-1, 1:-1, 1:-1, 1:-1] - A(p)
        
        d = r.copy()
        while (diff > tolerance):
            if it > max_it:
                print('\nSolution did not converged within the maximum'
                      ' number of iterations'
                      f'\nLast l2_diff was: {diff:.5e}')
                break
            else:
              if (it%100 == 0 ):
                print( 'it, l2norm ', it, diff)
                
            
            # Laplacian of the residual
            Ad[1:-1, 1:-1, 1:-1, 1:-1 ] = A(d)
            # Magnitude of jump.
            alpha = np.sum(r*r) / np.sum(d*Ad)
            # Iterated solution
            pnew = p + alpha*d
            # Intermediate computation
            beta_denom = np.sum(r*r)
            # Update the residual.
            r = r - alpha*Ad
            # Compute beta
            beta = np.sum(r*r) / beta_denom
            # Update the search direction.
            d = r + beta*d
            
            diff = l2_diff(pnew, p)
            tol_hist_jac.append(diff)
                        
            # Get ready for next iteration
            it += 1
            np.copyto(p, pnew)

    else:
        print(f'\nThe solution converged after {it} iterations')
        print('iterations it', it)
        print('\nSolution did  converge with'
                      ' {it} number of iterations '
                      f'\nLast l2_diff was: {diff:.5e}')
    diff = l2_diff(pnew, p_exact)
    print(f'The l2 difference between the computed solution and '
            f'the exact solution is:\n{diff}')
    ex_norm = np.linalg.norm(p_exact)
    p_norm = np.linalg.norm(pnew)
    norm_diff = ex_norm-p_norm
    print(ex_norm, p_norm, norm_diff)
    percent_accuracy = 100.0 *(1. - (abs(ex_norm-p_norm)/ex_norm))
    print('linalg percentage_accuracy', percent_accuracy)
##    p = pnew
    
## End of program logic
t2_stop = process_time()
print("elapsed time in secs: ", t2_stop-t2_start)
## start plotting section
t3_start = process_time()
plotSurface( X, Y, p_exact, "Exact" )
jc1 = 1*int(nx/2)-2
if submethod == 1 or submethod == 2:
##   plotSurface( X, Y, p_exact, "Exact" )
   p_plot = np.zeros((nx,ny))
   p_plot = p
   plotSurface( X, Y, p_plot, "Numerical" )     
elif submethod == 3:
   p_plot = np.zeros((nx,nz))
##   for j in range(ny):
   for k in range( nz ):
      for i in range( nx ):
        p_plot[k][i] = p[i][jc1][k]
   X,Z  = np.meshgrid( x, z )
   plotSurface( X, Z, p_plot, 'Numerical section_plot', zLabel='plot at j=10' )
elif submethod == 4:
   p_plot = np.zeros((nx,nz))
   jc = int(ny/2)-2
   tc = int(nt/2)-2
   for k in range( nz ):
       for i in range( nx ):
              p_plot[i][k] = p[i][k][jc1][tc]
   X,Z  = np.meshgrid( x, z )
   plotSurface( X, Z, p_plot, 'Numerical section_plot', zLabel='plot at j=10' )
    
fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))

if submethod <= 2:
  ax_1.contourf(X, Y, p_plot, 20)
  ax_2.contourf(X, Y, p_exact, 20)
elif submethod > 2:
  ax_1.contourf(X, Z, p_plot, 20)
  ax_2.contourf(X, Z, p_exact, 20)

# plot along the line y=0:
##jc = int(ly/(2*dy))
##ax_3.plot(x, p_exact[:,jc], '*', color='red', markevery=2, label=r'$p_e$')
##ax_3.plot(x, p_plot[:,jc], label=r'$pnew$')
ax_3.plot(y, p_plot[7, :])
ax_3.plot(y, p_exact[7, :], '*', color='black', markevery=2, label='y1=7')
ax_3.plot(y, p_plot[14, :])
ax_3.plot(y, p_exact[14, :], '*', color='red', markevery=2, label='y1=14')
ax_3.plot(y, p_plot[21, :])
ax_3.plot(y, p_exact[21, :], '*', color='green', markevery=2, label='y1=21')
ax_3.plot(y, p_plot[28, :])
ax_3.plot(y, p_exact[28, :], '*', color='blue', markevery=2, label='y1=28')
# add some labels and titles
ax_1.set_xlabel(r'$x$')
ax_1.set_ylabel(r'$y$')
ax_1.set_title('Numereical solution')

ax_2.set_xlabel(r'$x$')
ax_2.set_ylabel(r'$y$')
ax_2.set_title('Analytical solution')

ax_3.set_xlabel(r'$y$')
ax_3.set_ylabel(r'$p$')
ax_3.set_title(r'$p(x,0)$')

ax_3.legend();

plt.show()

t3_stop = process_time()
print("elapsed time in secs: ", t3_stop-t3_start)
## nice reference
##ref https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_02_Conjugate_Gradient.html
