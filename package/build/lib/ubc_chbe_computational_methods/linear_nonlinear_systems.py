'''
This module contains various linear system solution methods.
'''

# _____________________________________________________________________________________

def GaussElimination( A, b ):

    import numpy as np
    
    # Make a copy so as not to modify global variables
    # Multiply by 1.0 so as to convert everything to float (numpy related)
    A = A.copy() * 1.0
    b = b.copy() * 1.0
    
    # Check the matrix and rhs sizes
    nrow = A.shape[0]
    ncol = A.shape[1]
    if nrow != ncol:
        raise ValueError('Matrix A is not a square matrix')
    bsize = b.size
    if nrow != bsize:
        raise ValueError('Number of rows of matrix A and rhs b do not match')
    
    # Gauss elimination
    n = ncol
    for k in range(0,n-1):
        for i in range(k+1,n):
            # Compute factor
            factor = A[i,k] / A[k,k]
            
            # Update matrix coefficients
            for j in range(k,n):
                A[i,j] = A[i,j] - factor * A[k,j]
        
            # Update rhs vector
            b[i] = b[i] - factor * b[k]
        
    # Test if solution is unique
    if A[n-1,n-1] == 0:
        raise ValueError('No unique solution')
    
    # Backward substitution
    x = np.transpose([np.zeros(ncol)])
    x[n-1] = b[n-1] / A[n-1,n-1]
    for i in range(n-2,-1,-1):
        sumx = 0
        for j in range(i+1,n):
            sumx = sumx + A[i,j] * x[j]
        x[i] = ( b[i] - sumx ) / A[i,i]
    
    return x
    
# _____________________________________________________________________________________

def GaussPivoting( A, b ):

    import numpy as np
    
    # Make a copy so as not to modify global variables
    # Multiply by 1.0 so as to convert everything to float (numpy related)
    A = A.copy() * 1.0
    b = b.copy() * 1.0
    
    # Check the matrix and rhs sizes
    nrow = A.shape[0]
    ncol = A.shape[1]
    if nrow != ncol:
        raise ValueError('Matrix A is not a square matrix')
    bsize = b.size
    if nrow != bsize:
        raise ValueError('Number of rows of matrix A and rhs b do not match')

    # Gauss elimination
    n = ncol
    for k in range(0,n-1):
        # Partial pivoting
        # Look for largest pivot A[k,j] for j=k to n-1
        max = np.abs( A[k,k] )
        imax = k
        for j in range(k+1,n):
            if np.abs( A[j,k] ) > max:
                max = np.abs( A[j,k] )
                imax = j
                
        # If imax != k, swap k and imax
        if imax != k:
            print(f'Swap {k} and {imax}')
            
            # Swap matrix coefficients
            for j in range(k,n):
                tmp = A[k,j].copy()
                A[k,j] = A[imax,j].copy()
                A[imax,j] = tmp
       
            # Swap right hand side vector
            tmp = b[k].copy()
            b[k] = b[imax].copy()
            b[imax] = tmp
        
        # Actual elimination
        for i in range(k+1,n):
            # Compute factor
            factor = A[i,k] / A[k,k]
            
            # Update matrix coefficients
            for j in range(k,n):
                A[i,j] = A[i,j] - factor * A[k,j]
        
            # Update rhs vector
            b[i] = b[i] - factor * b[k]
                   
    # Test if solution is unique
    if A[n-1,n-1] == 0:
        raise ValueError('No unique solution')
    
    # Backward substitution
    x = np.transpose([np.zeros(ncol)])
    x[n-1] = b[n-1] / A[n-1,n-1]
    for i in range(n-2,-1,-1):
        sumx = 0
        for j in range(i+1,n):
            sumx = sumx + A[i,j] * x[j]
        x[i] = ( b[i] - sumx ) / A[i,i]
  
    return x
    
# _____________________________________________________________________________________

def Tridiag(e, f, g, b):

    import numpy as np
    
    n = f.size
    
    # Make a copy so as not to modify global variables
    # Multiply by 1.0 so as to convert everything to float (numpy related)
    f = f.copy() * 1.0
    b = b.copy() * 1.0
    
    # Gauss elimination
    for j in range(1,n):
        factor = e[j] / f[j-1] * 1.0 # Just to make it float
        f[j] = f[j] - factor * g[j-1]
        b[j] = b[j] - factor * b[j-1]
    
    # Backward substitution
    x = np.transpose([np.zeros(n)])
    x[n-1] = b[n-1] / f[n-1]
    for i in range(n-2,-1,-1):
        x[i] = ( b[i] - g[i] * x[i+1] ) / f[i]
    
    return x
    
# _____________________________________________________________________________________

def LUfactorization( A ):

    import numpy as np
    
    # Check the matrix and rhs sizes
    nrow = A.shape[0]
    ncol = A.shape[1]
    if nrow != ncol:
        raise ValueError('Matrix A is not a square matrix')
    n = ncol
    
    # Create the matrices L and U
    L = np.matrix(np.zeros((n,n))) * 1.0 # Convert to float in case it's not
    U = np.copy(A) * 1.0 # Convert to float in case it's not
    
    # Set 1s on the diagonal of L
    for i in range(0,n):
        L[i,i] = 1
    
    # Gauss elimination   
    for k in range(0,n-1):
        for i in range(k+1,n):
            # Compute factor
            factor = U[i,k] / U[k,k]
            
            # Update matrix coefficients
            for j in range(k,n):
                U[i,j] = U[i,j] - factor * U[k,j]
        
            # Update rhs vector
            L[i,k] = factor
    
    return L, U
    
# _____________________________________________________________________________________

def LUsolve( L, U, b ):

    import numpy as np
    
    # Size of the system
    n = b.size
    
    # Create the two column vectors
    x = np.transpose([np.zeros(n)])
    y = np.transpose([np.zeros(n)])
    
    # Forward substitution Ly=b
    y[0] = b[0] / L[0,0]
    for i in range(1,n):
        sumy = 0
        for j in range(0,i):
            sumy = sumy + L[i,j] * y[j]
        y[i] = ( b[i] - sumy ) / L[i,i]
        
    # Backward substitution Ux=y
    x[n-1] = y[n-1] / U[n-1,n-1]
    for i in range(n-2,-1,-1):
        sumx = 0
        for j in range(i+1,n):
            sumx = sumx + U[i,j] * x[j]
        x[i] = ( y[i] - sumx ) / U[i,i]        
    
    return x
    
# _____________________________________________________________________________________

def ComputeInvMat( A ):
    
    import numpy as np
    
    # Check the matrix and rhs sizes
    nrow = A.shape[0]
    ncol = A.shape[1]
    if nrow != ncol:
        raise ValueError('Matrix A is not a square matrix')
    n = ncol   
    
    # Compute the LU factorization of A
    (L,U) = LUfactorization( A )
    
    # Create the Am1 matrix
    Am1 = np.matrix(np.zeros((n,n)))
    
    # Create the rhs vector b
    b = np.transpose([np.zeros(n)])
    
    # Compute Am1
    for i in range(0,n):
        # Set the ith position of b to 1
        b[i] = 1.
        
        # Solve Av=LUv=b
        v = LUsolve( L, U, b )
        
        # Copy v into the ith column of Am1
        for j in range(0,n):
            Am1[j,i] = v[j]
        
        # Reset ith position of b to 0
        b[i] = 0.
    
    return Am1
    
# _____________________________________________________________________________________

def MultiNewton( function, Jacobian, x0, h=0., tol=1e-5, max_iter=100, conv_hist=True):

    import numpy as np
    
    # Initialization
    iteration = 0
    x = x0.reshape(-1, 1)
    n = x.size   
    eps_r = 100 # initial value for eps_r, can be any value bigger than ~1
    
    history = []
    
    while (eps_r >= tol) and (iteration < max_iter):
        
        # Compute the function value and the Jabobian matrix at x
        f = function( x )
        J = Jacobian( x, h )
        
        # Solve the linear system J*dx = f
        dx = GaussElimination( J, f )
        
        # Update x as x(iter+1) = x(iter) - dx
        x = x - dx
        
        # Compute the maximum relative error
        eps_r = 0.
        for i in range(0, n):
            ea = abs(dx[i])
            if abs(x[i]) > 1e-16:
                ea = 100. * ea / abs(x[i])
            if ea > eps_r:
                eps_r = ea.item()
        
        # Compute the norm of f1
        f = function( x )
        normf = sum(f*f)**0.5
        
        # convergence history____________________
        history.append([iteration, normf.item(), eps_r])

        iteration += 1
        if iteration == max_iter: print('Max iteration reached!')
    
    # show convergence history____________________
    if conv_hist:
        import pandas as pd
        
        df = pd.DataFrame(history, columns=['Iteration', 'Norm($ f(x_r) $)', '$ \epsilon_r $'])
        pd.options.display.float_format = '{:.10g}'.format
        display(df)
    
    return ( x, normf, eps_r, iteration )
