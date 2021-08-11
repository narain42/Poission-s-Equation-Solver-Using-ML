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

  return np.exp(-x[3])*(2. + 6.*x[1] + 12.* x[2]**2 + x[0]**2+x[1]**3+x[2]**4)



def sigmoid(z):
    return z / (1. + np.exp(-z))

def sigmoidPrime( z ):
    return egrad(sigmoid)(z)

def sigmoidPrimePrime( z ):
    return egrad(egrad(sigmoid))(z)



def neural_network(x, W):
    W1 = W[0][0]                    
    b1 = W[0][1]
    z1 = np.dot( W1, x ) + b1
    sigma = sigmoid( z1 )
    
    V =  W[1][0]
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
      
    V =  W[1][0]
    if k == 1:
       sigmaPrime = sigmoidPrime( z2 )
    else:
       sigmaPrime = sigmoidPrimePrime( z2 )

    return np.dot( V * ((W1[:,j] ** k)), sigmaPrime )

##new_zero = np.zeros(21)

hess_net = hessian(neural_network)

def error( inputs, W):

      totalError = 0        
      
      for x in inputs:
            bc1, bc2, bc3, bc4, bc5, bc6 = 0., 0., 0., 0., 0., 0.

            bc3 = neural_network( np.array([0,x[1],x[2],x[3]]), W) - np.exp(-x[3])*(x[1]**3+x[2]**4 )
            bc4 = neural_network( np.array([1.,x[1],x[2],x[3]]), W) - np.exp(-x[3])*(x[1]**3+x[2]**4 + 1.)
            bc5 = neural_network( np.array([x[0],0,x[2],x[3]]), W) - np.exp(-x[3])*( x[0]**2+x[2]**4)
            bc6 = neural_network( np.array([x[0],1,x[2],x[3]]), W) - np.exp(-x[3])*(x[0]**2+x[2]**4 + 1.)
            bc7 = neural_network( np.array([x[0],x[1],0,x[3]]), W) - np.exp(-x[3])*( x[0]**2+x[1]**3)
            bc8 = neural_network( np.array([x[0],x[1],1,x[3]]), W) - np.exp(-x[3])*(x[0]**2+x[1]**3 + 1.)
            bc1 =  neural_network( np.array([x[0],x[1],x[2],0]), W) - (x[0]**2+x[1]**3+x[2]**4)
            bc2 =  neural_network(np.array([x[0],x[1],x[2],1]), W)  - np.exp(-1.)*(x[0]**2+x[1]**3+x[2]**4 )

            lap1 =  hessian(neural_network)(x,W)[0][0]
            lap2 =  hessian(neural_network)(x,W)[1][1]
            lap3 =  hessian(neural_network)(x,W)[2][2]
            lap4 =  hessian(neural_network)(x,W)[3][3]
            error = (lap1 + lap2 + lap3+ lap4 - f(x))
            totalError +=  np.mean(error**2  + (bc1**2 + bc2**2 + bc3**2 + bc4**2+ bc5**2 + bc6**2 + bc7**2 + bc8**2))
      return totalError/ float( len( points ) )
  

if __name__ == '__main__':
    from time import process_time
    t2_start = process_time()
    np.random.seed( 11 )
    nx = 11
    ny = 11
    nz = 11
    nt = 11
    points = []
   
    x1 = np.linspace(0,1,nx)
    y1 = np.linspace(0,1,ny)
    z1 = np.linspace(0,1,nz)
    t1 = np.linspace(0,1,nt)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for m in range(nt):  
                    points.append( np.array( [x1[i], y1[j], z1[k], t1[m]] ) )

##   np.random.shuffle( points )

    # Training parameters.
    H = 18                  # Number of neurons in hidden layer.Seems like optimum value(8)
    batch_size = 5
    num_epochs = 200
    num_batches = int( np.ceil( len( points ) / batch_size ) )
    
    step_size = 0.005

##    def initParams():
    W =  [(np.random.uniform( -1, +1, (H, 4) ), np.random.uniform( -1, +1, H )), \
            (np.random.uniform( -1, +1, H ),)]

## For a restart, load W
##    W = pickle.load(open('W_rest_4D', 'rb'))

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


   
    pickle.dump(W, open('W_rest_4D', 'wb'), protocol=4)

    flattened_W, unflat_func = flatten(W)

    def loss_wrap(flattened_W, points):
       W = unflat_func(flattened_W) # recover
       return error(points, W)
 
    grad_wrap = grad(loss_wrap)

    loss_wrap(flattened_W, points)

    def error_part(flattened_W):
      '''make partial function. 
      Use this simple method because
      autograd or scipy does not like functool.partial'''
    # warning: global points is used
      return loss_wrap(flattened_W, points)
    

    error_part_grad = grad(error_part)

##    %%time
    optim_W = minimize(error_part, x0=flattened_W, jac=error_part_grad, method="BFGS", options={'gtol': 1e-14})

    o = optim_W
    print(o.fun,  o.nfev, '\n', o.message)

    X, Y , Z, T = np.meshgrid( x1, y1, z1, t1 )
    
    Z_t = np.zeros(( nx,ny,nz,nt ))
    W = unflat_func(o.x)
    for i in range( nx ):
        for j in range( ny ):
          for k in range( nz ):
            for m in range( nt ):
               v = np.array( [ X[i][j][k][m], Y[i][j][k][m], Z[i][j][k][m], T[i][j][k][m] ] )              # Input values.
            
               Z_t[i][j][k][m] = neural_network( v, W )

                        
    def exact(X,Y,Z, T):
      sol = np.exp(-T)*(X**2 + Y**3 + Z**4)  ## case1
      return sol
    surf = np.zeros((nx, ny, nz, nt))
    surf = exact(X,Y, Z, T)


##    print('exact sol')
##    print(surf)
##    print('numerical sol')
##    print(Z_t)

    print('L2 norm of exact, numerical, and difference')
    ex_norm = np.linalg.norm(surf)
    num_norm = np.linalg.norm(Z_t)
    norm_diff = ex_norm-num_norm
    print(ex_norm, num_norm, norm_diff)
    
 
    Z_plot = np.zeros((nx,ny))
    for i in range( nx ):
       for j in range( ny ):
         Z_plot[i][i] = Z_t[i][j][5][5]
    X, Y  = np.meshgrid( x1, y1 )
    plotSurface( X, Y, Z_plot, 'section_plot', zLabel='plot at k = 5, t=5' )            


    def exact2(X,Y):
      sol = np.exp(-.5)*(X**2 + Y**3 + .5**4)
      return sol
    surf2 = np.zeros((nx,ny))
    X, Y  = np.meshgrid( x1, y1 )
    surf2 = exact2(X,Y)
    plotSurface( X, Y, surf2, 'section_plot', zLabel='2d exact plot' )

    Z_plot = np.zeros((nx,nz))
    for i in range( nx ):
       for k in range( nz ):
              Z_plot[i][k] = Z_t[i][5][k][5]

    X, Z  = np.meshgrid( x1, z1 )
    plotSurface( X, Z, Z_plot, 'section_plot', zLabel='plot at j=5, t=5' )  

    def exact2(X, Z):
      sol = np.exp(-.5)*(X**2 + .5**3 + Z**4)
      return sol
    surf2 = np.zeros((ny,nz))
    surf2 = exact2(X,Z)
    plotSurface( X, Z, surf2, 'section_plot', zLabel='2d exact plot' )

    Z_plot = np.zeros((ny,nz))
    for j in range( ny ):
       for k in range( nz ):
         Z_plot[j][k] = Z_t[5][j][k][5]

    Y, Z  = np.meshgrid( y1, z1 )
    plotSurface( Y, Z, Z_plot, 'section_plot', zLabel='plot at i=5, t=5' )

    def exact2(Y,Z):
      sol =  np.exp(-.5)*(.5**2 + Y**3 + Z**4)
      return sol
    surf2 = np.zeros((ny,nz))
    surf2 = exact2(Y,Z)
    plotSurface( Y, Z, surf2, 'section_plot', zLabel='2d exact plot' )

    Z_plot = np.zeros((nx,nt))
    for i in range( nx ):
       for m in range( nt ):
         Z_plot[i][m] = Z_t[i][5][5][m]

    X, T  = np.meshgrid( x1, t1 )
    plotSurface( X, T, Z_plot, 'section_plot', zLabel='plot at j=5, k=5' )

    def exact2(X,T):
      sol =  np.exp(-T)*(X**2 + .5**3 + .5**4)
      return sol
    surf2 = np.zeros((nx,nt))
    surf2 = exact2(X, T)
    plotSurface( X, T, surf2, 'section_plot', zLabel='2d exact plot' )



t2_stop = process_time()
print("elapsed time in secs: ", t2_stop-t2_start)
