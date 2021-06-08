import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from numpy import unravel_index
import pickle
from time import process_time
## %matplotlib inline

t2_start = process_time()
nx = 7
ny = 7

dx = 1. / nx
dy = 1. / ny
lval = 0.
hval = 1.0

x1_space = np.linspace(0, 1, nx)

y1_space = np.linspace(0.4, 1, ny)

x2_space = np.linspace(0, 1.4, nx)

y2_space = np.linspace(0.0, 0.4, ny)

y_space = np.hstack((y1_space, y2_space))
x_space = np.hstack((x1_space, x2_space))

nx = 14
ny = 14

X, Y = np.meshgrid(x_space, y_space)

pi = np.pi

def analytic_solution(X,Y):
##    sol = ( Y **2) * np.sin(np.pi*X)
    sol = np.exp(-X)*(X+Y**3)
    return sol

surface = np.zeros((nx, ny))
surface = analytic_solution(X,Y)


def plot2d(surf, X, Y, msg):       
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, surf, rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    legend = ax.legend(loc='best', shadow=True, fontsize='x-small', title=msg)
    legend.get_frame().set_facecolor('white')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    plt.show()
    
plot2d(surface, X, Y, "Exact Sol")


def f(x):

  return np.exp(-x[0])*(x[0] - 2. + x[1]**3 + 6.*x[1])


def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoidPrime( z ):
	"""
	Derivative of sigmoid function.
	:param z: Input vector of values.
	:return: Element-wise sigmoid'(z).
	"""
	return sigmoid( z ) * ( 1.0 - sigmoid( z ) )


def sigmoidPrimePrime( z ):
	"""
	Second derivative of sigmoid function.
	:param z: Input vector of values.
	:return: Element-wise sigmoid''(z).
	"""
	return sigmoid( z ) * ( 1.0 - sigmoid( z ) ) * ( 1.0 - 2.0 * sigmoid( z ) )


def neural_network(x, W):
    Wt = W[0][0]                    
    b = W[0][1]
    V = W[1][0]                 
    z = np.dot( Wt, x ) + b
    sigma = sigmoid( z )                
    return np.dot( V, sigma )


def dkNnet_dxjk( x, W, j, k ):
	"""
	Compute the kth partial derivate of the nnet with respect to the jth input value.
	:param x: Input vector with two coordinates: (x_1, x_2).
	:param params: Network parameters (cfr. nnet(.)).
	:param j: Input coordinate with respect to which we need the partial derivative (0: x_1, 1: x_2).
	:param k: Partial derivative order (1 or 2 supported).
	:return: \frac{d^kN}{dx_j^k} evaluated at x = (x_1, x_2).
	"""
	Wt = W[0][0]
	b = W[0][1]
	V = W[1][0]
	z = np.dot(Wt, x) + b
	if k == 1:
		sigmaPrime = sigmoidPrime( z )
	else:
		sigmaPrime = sigmoidPrimePrime( z )
	return np.dot( V * (Wt[:,j] ** k), sigmaPrime )


def loss_function(W, step):
    loss_sum = 0.
    bc2, bc5, bc4, bc6 = 0., 0., 0., 0.    
    for yi in y_space:
        for xi in x_space:
                        
            input_point = np.array([xi,yi])
                        
            func = f(input_point)
            
            bc1 =  neural_network( np.array([xi,0]), W) - np.exp(-xi)*xi
            if (xi >= 0.0 and xi <= 1.):
              bc2 =  neural_network(np.array([xi,1]), W)  - np.exp(-xi)*(xi+1.) ##derichlet bc
            bc3 = neural_network( np.array([0,yi]), W) - yi**3
            if (yi >= 0.4 and yi <= 1.):
              bc4 = neural_network( np.array([1.,yi ]), W) - (1.+yi**3)*np.exp(-1.)
            if (xi >= 1. and xi <= 1.4):  
              bc5 =  neural_network(np.array([xi,0.4]), W)  - np.exp(-xi)*(xi+0.4**3) ##derichlet bc
            if (yi >= 0.0 and yi <= 0.4):
              bc6 = neural_network( np.array([1.4,yi ]), W) - (1.4+yi**3)*np.exp(-1.4)
            error = ((dkNnet_dxjk( input_point, W, 0, 2 ) + dkNnet_dxjk( input_point, W, 1, 2 ) ) \
                       - func)
            err_sqr = np.mean(error**2 + (bc1**2 + bc2**2 + bc3**2 + bc4**2 + bc5**2 + bc6**2 ))
            loss_sum += err_sqr
        
    return loss_sum


np.random.seed( 11 )
H = 7
W = [(np.random.uniform( -1, +1, (H,2) ), np.random.uniform( -1, +1, H )), \
     (np.random.uniform( -1, +1, H ),)]


## For a restart, load W
##W = pickle.load(open('W_rest_cont_L2_shaped', 'rb'))


surface2 = np.zeros((nx, ny))

for i, x in enumerate(x_space):
  for j, y in enumerate(y_space):
     net_outt = neural_network([x, y], W)
     surface2[j][i] = net_outt


plot2d(surface2, X, Y, "Initial Cond")


def callback(W, step, g):
    if step % 25 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                     loss_function(W, step)))


for k in range(5):
    W =  adam(grad(loss_function), W, step_size = 0.05, 
                num_iters=201, callback=callback)

pickle.dump(W, open('W_rest_cont_L2_shaped', 'wb'), protocol=4)

x_space = np.sort(x_space, axis =0)
y_space = np.sort(y_space, axis =0)
X, Y = np.meshgrid(x_space, y_space)

surface = np.zeros((nx, ny))
surface = analytic_solution(X,Y)

plot2d(surface2, X, Y, "Exact Sol")
     
for i, x in enumerate(x_space):
  for j, y in enumerate(y_space):
     net_outt = neural_network( [x, y], W)
     surface2[j][i] = net_outt

plot2d(surface2, X, Y, "Numerical Sol")


def plotHeatmap( Z, title ):
	"""
	Plot a heatmap of a rectangular matrix.
	:param Z: An m-by-n matrix to plot.
	:param title: Figure title.
	:return:
	"""
	fig1 = plt.figure()
	im = plt.imshow( Z, cmap=cm.jet, extent=(0, 1.4, 0, 1), interpolation='bilinear' )
	fig1.colorbar( im )
	plt.title( title )
	plt.xlabel( r"$x_1$" )
	plt.ylabel( r"$x_2$" )
	plt.show()


print( surface[2] )
print( surface2[2] )

fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))

# For more info on contour plots
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html
#
ax_2.contourf(X, Y, surface2, 20)
ax_1.contourf(X, Y, surface, 20)

# plot along the line y=0:
jc = 1
ax_3.plot(x_space, surface[:,jc], '*', color='red', markevery=2, label=r'$p_e$')
ax_3.plot(x_space, surface2[:,jc], label=r'$pnew$')

# add some labels and titles
ax_1.set_xlabel(r'$x$')
ax_1.set_ylabel(r'$y$')
ax_1.set_title('Exact solution')

ax_2.set_xlabel(r'$x$')
ax_2.set_ylabel(r'$y$')
ax_2.set_title('Numerical solution')

ax_3.set_xlabel(r'$x$')
ax_3.set_ylabel(r'$p$')
ax_3.set_title(r'$p(x,0)$')

ax_3.legend();

plt.show()


# Plotting the error heatmap.
plotHeatmap( surface-surface2, r"Error heatmap for predicted - analytical solution" )



t2_stop = process_time()
print("elapsed time in secs: ", t2_stop-t2_start)
