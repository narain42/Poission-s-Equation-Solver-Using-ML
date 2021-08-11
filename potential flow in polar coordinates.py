
import autograd.numpy as np
from autograd import grad, hessian
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

# element-wise gradient is a standard-alone function in v1.2
from autograd import elementwise_grad as egrad
from autograd.misc.flatten import flatten_func, flatten
from scipy.optimize import minimize
import math

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pickle
from time import process_time
## %matplotlib inline

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


def plotHeatmap( Z, title ):
	"""
	Plot a heatmap of a rectangular matrix.
	:param Z: An m-by-n matrix to plot.
	:param title: Figure title.
	:return:
	"""
	fig1 = plt.figure()
	im = plt.imshow( Z, cmap=cm.jet, extent=(1, 60, -np.pi, np.pi), interpolation='bilinear' )
	fig1.colorbar( im )
	plt.title( title )
	plt.xlabel( r"$x_1$" )
	plt.ylabel( r"$x_2$" )
	plt.show()



pi = np.pi


def f(x):

  return 0.

def exact(X,Y):
      sol = (X + (1./X))*(np.cos(Y) )
      
      return sol

def sigmoid(z):
    
 return z/ (1. + np.exp(-z)) 


def sigmoidPrime( z ):
    return egrad(sigmoid)(z)



def sigmoidPrimePrime( z ):
    return egrad(egrad(sigmoid))(z)


netp = 2
def neural_network(x, W):
    W1 = W[0][0]                    
    b1 = W[0][1]
    z1 = np.dot( W1, x ) + b1
    sigma = sigmoid( z1 )
    if (netp ==3):
       W2 = W[1][0]                    
       b2 = W[1][1]
       z2 = np.dot( W2, sigma ) + b2
       sigma = sigmoid( z2 )
    V =  W[netp-1][0]
    V1 = np.dot( V, sigma ) 
    return V1


def dkNnet_dxjk( x, W, j, k ):
##	"""
##	Compute the kth partial derivate of the nnet with respect to the jth input value.
##	:param x: Input vector with two coordinates: (x_1, x_2).
##	:param params: Network parameters (cfr. nnet(.)).
##	:param j: Input coordinate with respect to which we need the partial derivative (0: x_1, 1: x_2).
##	:param k: Partial derivative order (1 or 2 supported).
##	:return: \frac{d^kN}{dx_j^k} evaluated at x = (x_1, x_2).
##	"""
    W1 = W[0][0]                    
    b1 = W[0][1]
    z2  = np.dot( W1, x ) + b1
    sigma = sigmoid( z2 )
    if (netp ==3):
       W2 = W[1][0]                    
       b2 = W[1][1]
       z2  = np.dot( W2, sigma ) + b2
       
    V =  W[netp-1][0]
    if k == 1:
       sigmaPrime = sigmoidPrime( z2 )
    else:
       sigmaPrime = sigmoidPrimePrime( z2 )

    return np.dot( V * ((W1[:,j] ** k)), sigmaPrime )


def error( inputs, W):

      totalError = 0        
      
      for x in inputs:
            
            func = f(x)
            
            bc1 = 0.0
            bc2 = 0.0
            bc3 = neural_network(np.array([1,x[1]]), W)  - 2.*np.cos(x[1])
            bc4 = neural_network(np.array([60.0,x[1]]), W) - (60.0+(1./60.0))*np.cos(x[1])
            bc5 = neural_network(np.array([x[0],-pi]), W) + (x[0]+(1./x[0]))
            bc6 = neural_network(np.array([x[0],pi]), W) + (x[0]+(1./x[0]))
            
            nabla_r = grad(neural_network)(x, W)[0]
            lap1 = hessian(neural_network)(x, W)[0][0]
            lap2 = hessian(neural_network)(x, W)[1][1]
##            lap = dkNnet_dxjk( x, W, 0, 2 ) + (1./x[0]**2)*dkNnet_dxjk( x, W, 1, 2 )+(1./x[0])*dkNnet_dxjk( x, W, 0, 1 )
            lap = (1./x[0]**2)*lap2 + (1./x[0])*nabla_r + lap1
            
            error = lap - func 
                            
            totalError +=  np.mean(error**2  + bc1**2 + bc2**2 + bc3**2 + bc4**2 + bc5**2 + bc6**2 )
            return totalError /float( len( points ) )

if __name__ == '__main__':
    
    t2_start = process_time()
    npr.seed( 11)
    nx = 21
    ny = 21
    pi = np.pi

## very important. set the following parameters
## Restrt True for restart, otherwise false
## Replot True for plotting from pickle dump file
## Bfmth True if BFGS minimization is employed after Adam(default) minimization
## NewCalc True if new or more iterations(Restrt) desired
    
    Restrt = False
    Replot = False
    Bfmeth = True
    Newcalc = True
    
    x1 = np.linspace(1, 60, nx)
    y1 = np.linspace(-pi, pi, ny)
    dy = 2.*pi/ny
    X, Y = np.meshgrid(x1, y1)
    
    
    points = []
    
    for i in range(nx  ):
        for j in range( ny  ):
           points.append( np.array( [x1[i], y1[j]] ) )

##    npr.shuffle(points)

    # Training parameters.
    H = 12                                   # Number of neurons in hidden layer.
    batch_size = 5
    num_epochs = 800
    
    num_batches = int( np.ceil( len( points ) / batch_size ) )
    step_size = 0.005
        
    ## def initParams():
    W =  [(npr.uniform( -1, +1, (H, 2) ), npr.uniform( -1, +1, H )), \
            (npr.uniform( -1, +1, H ),)]

    if (Newcalc):
      if (Restrt):
       ## For a restart, load W
       W = pickle.load(open('W_rest_case3', 'rb'))      

      def batchIndices( i ):
          idx = i % num_batches
          return slice( idx * batch_size, (idx + 1) * batch_size )

    # Define training objective
      def objective( W, i ):
          idx = batchIndices( i )
          return error( points[idx], W )

    # Get gradient of objective using autograd.
      objective_grad = grad( objective )
      train_acc = []
      print( "     Epoch     |    Train accuracy  " )
      
      def printPerf( W, i, _ ):
        if i % num_batches == 0:
            train_acc = error( points, W )
            print( "{:15}|{:20}".format( i // num_batches, train_acc ) )
            
      W = adam( objective_grad, W, step_size=step_size,
               num_iters= num_epochs * num_batches, callback=printPerf)
      printPerf( W, 0, None )
        
      pickle.dump(W, open('W_rest_case3', 'wb'), protocol=4)
    
    if(Bfmeth):      
      flattened_W, unflat_func = flatten(W)
            
      def loss_wrap(flattened_W, points):
          W = unflat_func(flattened_W) # recover
          return error(points, W)

      def error_part(flattened_W):
          return loss_wrap(flattened_W, points)
          
      error_part_grad = grad(error_part)
          
        
      optim_W = minimize(error_part, x0=flattened_W, jac=error_part_grad, method='L-BFGS-B', options={'gtol': 1e-20})

      o = optim_W
      print(o.fun,  o.nfev, '\n', o.message)
      W = unflat_func(o.x)
      pickle.dump(W, open('W_rest_case3', 'wb'), protocol=4)
      
    if(Replot):
      W = pickle.load(open('W_rest_case3', 'rb')) 
    # Plot solutions.
    
    # Make data.
    
    Z_t = np.zeros( (nx,ny ) )
    cp_r = np.zeros( (nx,ny ) )
    surface = np.zeros((nx, ny))
    surface = exact(X,Y)
   
   
    for j, y in enumerate(y1):
      for i, x in enumerate(x1):
      
         Z_t[j][i] = neural_network( [x,y], W )
         v_r = dkNnet_dxjk( [x,y], W, 0, 1 )
         v_t = (1./x)*dkNnet_dxjk( [x,y], W, 1, 1 )
         cp_r[j][i] = 1. - (v_r**2+v_t**2)

            
    ex_norm = np.linalg.norm(surface)
    num_norm = np.linalg.norm(Z_t)
    norm_diff = ex_norm-num_norm
    print(ex_norm, num_norm, norm_diff)
    
    plotSurface(X,Y,surface, 'exact sol')
    plotSurface(X,Y,Z_t, 'Numerical sol')

    fig, ay = plt.subplots(subplot_kw=dict(projection='polar'))
    ay.plot(Y,X)
    ay.set_xticks(np.arange(0,2.*pi,pi/6.0))
    ay.set_ylim(0,60)
    ay.set_yticks(np.arange(1,60,10))
    ay.set_title('Polar Grid Used')
    plt.show()

    fig, az = plt.subplots(subplot_kw=dict(projection='polar'))
    az.contourf(Y, X, Z_t, 40)
    az.set_title('Numerical potential contour')
    plt.show()

    fig, ac = plt.subplots(subplot_kw=dict(projection='polar'))
    ac.contourf(Y, X, surface, 40)
    ac.set_title('exact potential contours')
    plt.show()

    rads = 1.0
    cp_exact = 2.*((rads/X))**2*(np.cos(2.*Y)) - (rads/X)**4
    fig, ab = plt.subplots(subplot_kw=dict(projection='polar'))
    ab.contourf(Y, X, cp_exact, 40)
    ab.set_title('exact Cp contours')
    plt.show()

    
    fig, aa = plt.subplots(subplot_kw=dict(projection='polar'))
    aa.contourf(Y, X, cp_r, 40)
    aa.set_title('Numerical Cp contours')
    plt.show()
        
##    # Plotting the error heatmap.
    plotHeatmap( Z_t-surface, r"Error heatmap for $|\phi_a(\mathbf{x}) - \phi_t(\mathbf{x})|$" )
##
    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))

    ax_1.contourf(X, Y, Z_t, 20)
    
# plot along the line y=0:
    ax_2.contourf(X, Y, surface, 20)

# plot along the line x=0:
    
    ax_3.plot(x1, Z_t[:,5])
    ax_3.plot(x1, surface[:,5], '*', color='black', markevery=2, label='y1=5')
    ax_3.plot(x1, Z_t[:,10])
    ax_3.plot(x1, surface[:,10], '*', color='red', markevery=2, label='y1=10')
    ax_3.plot(x1, Z_t[:,15])
    ax_3.plot(x1, surface[:,15], '*', color='green', markevery=2, label='y1=15')
    ax_3.plot(x1, Z_t[:,19])
    ax_3.plot(x1, surface[:,19], '*', color='blue', markevery=2, label='y1=25')
##
# add some labels and titles
    ax_1.set_xlabel(r'$x$')
    ax_1.set_ylabel(r'$y$')
    ax_1.set_title('Numerical solution')

    ax_2.set_xlabel(r'$x$')
    ax_2.set_ylabel(r'$y$')
    ax_2.set_title('exact solution')

    ax_3.set_xlabel(r'$x$')
    ax_3.set_ylabel(r'$p$')
    ax_3.set_title('')

    ax_3.legend();

    plt.show()
    
    t2_stop = process_time()
    print("elapsed time in secs: ", t2_stop-t2_start)
