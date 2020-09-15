'''
This module contains various 1D numerical integration methods for a function f(x) between a and b.
'''

# ______________________________________________________________________________________

def integrate(f, a, b, n, method='trapz'):
    '''
    This function integrates y=f(x) with the midpoint, trapezoidal, Simpson's 1/3 and 3/8 rules.
    
    Inputs:
    f: function to be integrated.
    a: lower limit of integration
    b: upper limit of integration
    n: number of sub-intervals
    method: 'midpoint', 'trapz', 'simp_1/3', 'simp_3/8'
    
    Returns:
    I: the computed value of the integral.
    '''
    
    import numpy as np
    
    if n < 1:
        raise ValueError('You must have at least one interval.')
    
    # 1d array of x positions: n sub-intervals => n+1 points
    x = np.linspace(a, b, n+1)
    
    # Sub-interval length
    delta_x = (b - a) / n
    
    # Values of f at points x
    y = f(x)
    
    if method == 'midpoint':    # _______________________
        I = 0
        for i in range(n):
            mid = (x[i+1] + x[i]) / 2
            I += f(mid) * delta_x
    
    elif method == 'trapz':    # _______________________
        I = 0
        for i in range(n):
            I += ((y[i] + y[i+1]) / 2) * delta_x
    
    elif method == 'simp_1/3':    # _______________________
        if n % 2 != 0:
            raise ValueError('n must be an even number for Simpson\'s 1/3 rule.')
            
        I = 0
        for i in range(0, n, 2):
            I += ((y[i] + 4 * y[i+1] + y[i+2]) / 3) * delta_x
            
    elif method == 'simp_3/8':    # _______________________
        if n % 3 != 0:
            raise ValueError('n must be a multiple of 3 for Simpson\'s 3/8 rule.')
            
        I = 0
        for i in range(0, n-1, 3):
            I += (y[i] + 3 * y[i+1] + 3 * y[i+2] + y[i+3]) * (3 / 8) * delta_x
    else:    # _______________________
        raise Exception('Available integration methods are midpoint, trapz, simp_1/3 and sim_3/8.')
    
    return I

# ______________________________________________________________________________________

def romberg(f, a, b, tol=1e-10, maxit=20):
    '''
    This function integrates y=f(x) with the Romberg method.
    
    Inputs:
    f: function to be integrated.
    a: lower limit of integration
    b: upper limit of integration
    tol: required tolerance
    maxit: maximum number of iterations
    
    Returns:
    I: the computed value of the integral.
    '''
    
    import numpy as np
    
    # The Romberg matrix
    I = np.zeros((maxit,maxit))
    
    # Initialization
    n = 1
    I[0,0] = integrate(f, a, b, n, method='trapz')
    iter = 0
    
    # Loop until convergence or maxit is attained
    while ( iter < maxit ):
        iter += 1
   
        # Divide h by 2 at each iteration
        n = 2**iter
        
        # Compute an approximation with trapezoidal rule with h = ( b - a ) / 2^iter
        I[iter,0] = integrate(f, a, b, n, method='trapz')
   
        # Compute the improved estimates at each level
        for k in range(1,iter+1):
            j = iter - k;
            I[j,k] = ( 4**k * I[j+1,k-1] - I[j,k-1] ) / ( 4**k - 1 )
   
        # Compute relative error and check convergence
        ea = np.abs( ( I[0,iter] - I[1,iter-1] ) / I[0,iter] )
        if ea < tol: break
    
    integral = I[0,iter]
    niter = iter
    
    return( integral, niter, ea )

# ______________________________________________________________________________________