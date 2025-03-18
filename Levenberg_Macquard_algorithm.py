""" 
Project work for Data Analysis - Levenberg Macquardt Method

This program performs the Levenberg-Macquardt for the model function, which is the Gaussian function in this case. 
This iterative algorithm finds a minimiser of a given non linear least-squares problem. 
Here we introduced Gaussian noise into the initial Gaussian model function, and try to refit the function back using the LM algorithm.
The accuracy of the algorithm was compared to an in-built library which is curve_fit from scipy.optimize. 

"""


#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Function definition for the model Gaussian
def model_fn(x, p):
    """
    Parameters
    ----------
    x : x values of data
    p : parameter array

    Returns
    -------
    The gaussian fn with the parameters

    """
    return p[0]*np.exp(-(x-p[1])*(x-p[1])/2/p[2]/p[2]) + p[3]




#Function definition for the model Gaussian for curve_fit
def model_fn_for_curvefit(x, p0, p1, p2, p3):
    """
    Parameters
    ----------
    x : x values of data
    p0 : Amplitude
    p1 : Mean
    p2 : Std deviation
    p3 : Baseline shift

    Returns
    -------
    Gaussian fn for curvefit analysis

    """
    return p0*np.exp(-(x-p1)*(x-p1)/2/p2/p2) + p3





#Function to compute the Jacobian matrix with pchange as pertubation 
def jacobian(model_fn, x, p, nrow, ncol, pchange = 1e-5):
    """
    Parameters
    ----------
    model_fn : Gaussian fn
    x : x values of data
    p : parameter array
    nrow : length of x values array
    ncol : length of parameters array
    pchange : small pertubation to parameter. The default is 1e-5.

    Returns
    -------
    J : Calculates the Jacobian matrix J, which contains the derivatives of the model with respect to each parameter

    """
    J = np.zeros((nrow, ncol))
    for i in range(ncol):
        dp = np.zeros((ncol))
        dp[i] = pchange #small pertubation to the parameter
        J[:, i] = (model_fn(x, p + dp) - model_fn(x, p)) / dp[i]
    return J




#Implements the Levenberg-Marquardt algorithm for non-linear least squares fitting
def levmacq(x,y, model_fn, lenx, lenp, W, p0, lamda, err, rho0, max_step):
    """
    

    Parameters
    ----------
    x : x values of data
    y : y values of data
    model_fn : Gaussian function
    lenx : Length of x values of data
    lenp : Length of parameters array 
    W : Weight matrix, we took it as identity matrix
    p0 : Initial guess of the parameters
    lamda :quality for the choice of parameter, to decide good direction for the fit
    err : tolerance
    rho0 : inital guess for figure of merit
    max_step : maximum number of steps

    Returns
    -------
    p : optimum parameters for the final fit
    cov : covariance matrix,with the diagonal elements as the error correction
     

    """
    #To overlay the plots as they fit at different iterations
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], color='red', label='Fitting')
    ax.scatter(x, y,color='green', label='Data')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_title('Measurement Data and fitted function')
    plt.show(block=False)
    
    
    p = p0
    h = np.array([1000]) #initializes the update vector h for parameter changes
    chi2_prev = np.dot((y - model_fn(x,p)).T,np.dot(W, y - model_fn(x,p)))#squared error between the data and model with current parameters
    step = 0
    while np.sqrt(sum(h*h))>err:#iterates until the magnitude of h falls below the specified error tolerance
        J = jacobian(model_fn, x, p, lenx, lenp, pchange = 1e-5)
        L= np.dot(np.dot(J.T,W),J) + lamda*np.identity(lenp) 
        h = 2*np.dot(np.linalg.inv(L),np.dot((y - model_fn(x,p)),np.dot(W, J)))
        chi2= np.dot((y - model_fn(x,p+h)).T,np.dot(W, y - model_fn(x,p+h)))#chi-square value with the proposed parameter update 
        #loop to check convergence criteria and go to good direction in steps
        #If the new chi-square is better, lamda is adjusted based on rho
        #rho is large, the step is accepted, and lamda can be increased for faster convergence
        #otherwise, lamda is decreased to ensure a stable update
        while True:
            if chi2 > chi2_prev:
                lamda = min(lamda*10,1e7)
            else:
                rho = (chi2_prev- chi2) / np.dot(h.T, np.dot(L,h))
             
                if rho > rho0:
                    lamda = min(lamda*10,1e7)
                else : 
                    lamda= max(lamda/10,1e-7)
                    break;
            
            L= np.dot(np.dot(J.T,W),J) + lamda*np.identity(lenp)
            h = np.dot(np.linalg.inv(L),np.dot((y - model_fn(x,p)),np.dot(W, J)))
            chi2= np.dot((y - model_fn(x,p+h)).T,np.dot(W, y - model_fn(x,p+h)))
       
        p=p+h 
        #update the plot as function fits
        x_model = np.linspace(min(x), max(x), 1000)
        y_model = model_fn(x_model, p)
        line.set_data(x_model, y_model)
        
        text = "Optimized parameters:\n"
        for k in range(len(p)):
            text += f"p[{k+1}] = {p[k]:.3f}\n"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',bbox=dict(facecolor='lightgray',alpha=1.0))
        
        plt.pause(1)       
        chi2_prev = chi2
        step +=1
        if step > max_step:
            break; 
        print(f'step = {step}, h = {h}')
    
    L= np.dot(np.dot(J.T,W),J) + lamda*np.identity(lenp)
    L = np.linalg.inv(L) # covariance matrix is defined as inverse of L
    cov = np.zeros((lenp,lenp))
    for i in range(lenp): cov[i][i] = np.sqrt(L[i][i]) # diagonal elements has the errors of the fit
    for i in range(lenp):  
        for j in range(i+1,lenp):
            cov[i][j] = L[i][j]/(cov[i][i]*cov[j][j])
            cov[j][i] = cov[i][j] 
    return p,cov



n=50 #number of data points
x = np.linspace(-10,10,n)
y_h = model_fn(x, [3,1.2,0.5,1.5]) #the parameter array for the actual y data.
noise = np.random.normal(1, 0.1, len(y_h)) # adding gaussian noise (mu,sigma,size)
y = y_h + noise
data = np.column_stack((x, y_h, y))
np.savetxt('Measurement_data.txt', data, header='x y_h y', comments='') 
data = np.loadtxt('Measurement_data.txt', skiprows=1) 
x = data[:, 0]
y = data[:, 2]
p = np.array([5,5,5,5]) #initial guess for parameters, the guess can be changed for each parameter of the function, but not too much, please :,)
W = np.identity(n) 
lenp = len(p)
lenx = len(x)

#setting parameters for LM algorithm
err = 10**-3
max_step = 500
lamda= 1
rho0= 2
p,cov = levmacq(x,y, model_fn, lenx, lenp, W, p, lamda, err, rho0, max_step)
print(f"optimised parameters :=  {p},\n The covariance matrix is = \n{cov}")
e=np.diag(cov) #extracting diagonal errors from covariance matrix
for k in range(len(e)):
     print (f' Parameter {k+1} has error correction of {e[k]}')


#checking with inbuilt curve_fit from scipy package and writing results to file
p_t,cov_t = curve_fit(model_fn_for_curvefit, x,y, p,method = 'lm')
print(f"p_t = {p_t},\ncov_t = \n{cov_t}")
e_t=np.diag(cov_t) #extracting diagonal errors from covariance matrix
for k in range(len(e_t)):
     print (f' Parameter {k+1} has error correction from curve fit as{e_t[k]}')
data = np.column_stack((p,e,p_t,e_t)) # writing results to file
np.savetxt('Result.txt', data, header='Model_optimum_parameters Error_correction Optimum_params_curvefit Curvefit_errors', comments='')
