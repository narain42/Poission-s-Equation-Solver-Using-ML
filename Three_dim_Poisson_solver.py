import autograd.numpy as np
from autograd import grad, hessian
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

# element-wise gradient is a standard-alone function in v1.2
from autograd import elementwise_grad as egrad
from autograd.misc.flatten import flatten_func, flatten
from scipy.optimize import minimize

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
	im = plt.imshow( Z, cmap=cm.jet, extent=(0, 1, 0, 1), interpolation='bilinear' )
	fig1.colorbar( im )
	plt.title( title )
	plt.xlabel( r"$x_1$" )
	plt.ylabel( r"$x_2$" )
	plt.show()



pi = np.pi


def f(x):
 return -3.*(pi**2)*(np.sin(pi*x[0]))*(np.sin(pi*x[1]))*(np.sin(pi*x[2])) ##case2

def sigmoid(z):
    return z / (1. + np.exp(-z))

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
    return np.dot( V, sigma )

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

##new_zero = np.zeros(21)

hess_net = hessian(neural_network)

def error( inputs, W):

      totalError = 0        
      bc1, bc2, bc3, bc4, bc5, bc6 = 0., 0., 0., 0., 0., 0.
      for x in inputs:
            bc1 =  neural_network( np.array([x[0],x[1],0]), W) - 0.
            bc2 =  neural_network(np.array([x[0],x[1],1]), W)  - 0.
            bc3 = neural_network( np.array([0,x[1],x[2]]), W) - 0.
            bc4 = neural_network( np.array([1.,x[1],x[2]]), W) - 0.
            bc6 = neural_network( np.array([x[0],1,x[2]]), W) - 0.
            bc5 = neural_network( np.array([x[0],0,x[2]]), W) - 0.

##            error = (1.0*(dkNnet_dxjk( x, W, 0, 2 ) + dkNnet_dxjk( x, W, 1, 2 ) \
##                      + dkNnet_dxjk( x, W, 2, 2 )) \
##                       - f(x))
            lap1 =  hessian(neural_network)(x,W)[0][0]
            lap2 =  hessian(neural_network)(x,W)[1][1]
            lap3 =  hessian(neural_network)(x,W)[2][2]
            error = (lap1 + lap2 + lap3 - f(x))
            totalError +=  np.mean(error**2  + (bc1**2 + bc2**2 + bc3**2 + bc4**2+ bc5**2 + bc6**2))
            return totalError/ float( len( points ) )


if __name__ == '__main__':
    from time import process_time
    t2_start = process_time()
    np.random.seed( 11 )
    nx = 30
    ny = 30
    nz = 30
    points = []
   
    x1 = np.linspace(0,1,nx)
    y1 = np.linspace(0,1,ny)
    z1 = np.linspace(0,1,nz)
        
    for i in range(nx  ):
        for j in range( ny  ):
          for k in range( nz  ):
              points.append( np.array( [x1[i], y1[j], z1[k]] ) )

    np.random.shuffle( points )

    # Training parameters.
    H = 12                  # Number of neurons in hidden layer.Seems like optimum value(8)
    batch_size = 5
    num_epochs = 200
    num_batches = int( np.ceil( len( points ) / batch_size ) )
    
    step_size = 0.005

##    def initParams():
    W =  [(np.random.uniform( -1, +1, (H, 3) ), np.random.uniform( -1, +1, H )), \
            (np.random.uniform( -1, +1, H ),)]

## For a restart, load W
##    W = pickle.load(open('W_rest_sin3D', 'rb'))

    def batchIndices( i ):
        idx = i % num_batches
        return slice( idx * batch_size, (idx + 1) * batch_size )

    # Define training objective
    def objective( W, i ):
        idx = batchIndices( i )
        return error( points[idx], W )

    # Get gradient of objective using autograd.
    objective_grad = grad( objective )

    print( "     Epoch     |    Train accuracy  " )
    def printPerf( W, i, _ ):
        if i % num_batches == 0:
            train_acc = error( points, W )
            print( "{:15}|{:20}".format( i // num_batches, train_acc ) )

    
    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    W = adam( objective_grad, W, step_size=step_size,
                             num_iters=num_epochs * num_batches, callback=printPerf )
    printPerf( W, 0, None )
    
    pickle.dump(W, open('W_rest_sin3D', 'wb'), protocol=4)


    def loss_wrap(flattened_W, points):
       W = unflat_func(flattened_W) # recover
       return error(points, W)
 
    grad_wrap = grad(loss_wrap)

    loss_wrap(flattened_W, points)

    def error_part(flattened_W):
      
      return loss_wrap(flattened_W, points)
    
    error_part_grad = grad(error_part)

##%%time
    optim_W = minimize(error_part, x0=flattened_W, jac=error_part_grad, method="BFGS", options={'gtol': 1e-14})

    o = optim_W
    print(o.fun,  o.nfev, '\n', o.message)
    def exact(X,Y,Z):
      sol = (np.sin(pi*X))*(np.sin(pi*Y))*(np.sin(pi*Z))
      return sol

    X, Y , Z = np.meshgrid( x1, y1, z1 )
    
    Z_t = np.zeros(( nx,ny,nz ))
    W = unflat_func(o.x)
    for i in range( nx ):
        for j in range( ny ):
          for k in range( nz ):
            v = np.array( [X[i][j][k], Y[i][j][k], Z[i][j][k]] ) 
            Z_t[i][j][k] = neural_network( v, W )
                        
    surf = np.zeros((nx, ny, nz))
    surf = exact(X,Y, Z)

    print('L2 norm of exact, numerical, and difference')
    ex_norm = np.linalg.norm(surf)
    num_norm = np.linalg.norm(Z_t)
    norm_diff = ex_norm-num_norm
    print(ex_norm, num_norm, norm_diff)
    
    

    Z_plot = np.zeros((nx,ny))
    for i in range( nx ):
       for j in range( ny ):
              Z_plot[i][j] = Z_t[i][j][5]

    X, Y  = np.meshgrid( x1, y1 )
    plotSurface( X, Y, Z_plot, 'section_plot', zLabel='plot at k = 5' )            


    Z_plot = np.zeros((nx,ny))
    for i in range( nx ):
       for j in range( ny ):
          Z_plot[i][j] = Z_t[i][j][10]

    X, Y  = np.meshgrid( x1, y1 )
    plotSurface( X, Y, Z_plot, 'section_plot', zLabel='plot at k=10' )            

    Z_plot = np.zeros((nx,ny))
    for i in range( nx ):
       for j in range( ny ):
              Z_plot[i][j] = Z_t[i][j][20]

    X, Y  = np.meshgrid( x1, y1 )
    plotSurface( X, Y, Z_plot, 'section_plot', zLabel='plot at k=20' )            

    Z_plot = np.zeros((nx,ny))
    for i in range( nx ):
       for j in range( ny ):
              Z_plot[i][j] = Z_t[i][j][29]

    X, Y  = np.meshgrid( x1, y1 )
    plotSurface( X, Y, Z_plot, 'section_plot', zLabel='plot at k=29' )  

    def exact2(X,Y):
      sol = (np.sin(pi*X))*(np.sin(pi*Y))
      return sol
    surf2 = np.zeros((nx,ny))
    X, Y  = np.meshgrid( x1, y1 )
    surf2 = exact2(X,Y)
    plotSurface( X, Y, surf2, 'section_plot', zLabel='2d exact plot' )
 
    Z_plot = np.zeros((nx,nz))
    for i in range( nx ):
       for k in range( nz ):
              Z_plot[i][k] = Z_t[i][15][k]

    X, Z  = np.meshgrid( x1, z1 )
    plotSurface( X, Y, Z_plot, 'section_plot', zLabel='plot at j=15' )  

    def exact2(X, Z):
      sol = (np.sin(pi*X))*(np.sin(pi*Z))
      return sol
    surf2 = np.zeros((ny,nz))
    surf2 = exact2(X,Z)
    plotSurface( X, Z, surf2, 'section_plot', zLabel='2d exact plot' )

    Z_plot = np.zeros((ny,nz))
    for j in range( ny ):
       for k in range( nz ):
              Z_plot[j][k] = Z_t[15][j][k]

    Y, Z  = np.meshgrid( y1, z1 )
    plotSurface( Y, Z, Z_plot, 'section_plot', zLabel='plot at i=15' )

    def exact2(Y,Z):
      sol = (np.sin(pi*Y))*(np.sin(pi*Z))
      return sol
    surf2 = np.zeros((ny,nz))
    surf2 = exact2(Y,Z)
    plotSurface( Y, Z, surf2, 'section_plot', zLabel='2d exact plot' )

    t2_stop = process_time()
    print("elapsed time in secs: ", t2_stop-t2_start)
