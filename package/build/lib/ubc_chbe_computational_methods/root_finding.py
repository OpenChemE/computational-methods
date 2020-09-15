'''
This module contains various root-finding methods.
'''

# ______________________________________________________________________________________

def interval_finder(function, a, b, delta_x=0.01, with_plot=True):
    '''
    This function implements the incremental search method and
    finds the intervals in [a, b] where the input function changes sign.
    The ouput of this function is a list that contains pairs of numbers.
    Each pair represents the lower bound and the upper bound of the
    corresponding intervals containing the roots.
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    # initializing____________________
    intervals = []
    
    x_l, x_u = a, a + delta_x
    f = function
	
    # First check {a, b}
    if f(a) == 0:
        print(f'x = {a:g} is a root of the function!\n')
    elif f(b) == 0:
        print(f'x = {b:g} is a root of the function!\n')
    
    # main root-searching loop____________________
    while x_u <= b:

        if np.sign(f(x_l) * f(x_u)) == -1:
            intervals.append([x_l, x_u])

        x_l += delta_x
        x_u += delta_x

    if intervals == []:
        print('No intervals found.')
        return None
    else:
        # displaying intervals____________________
        print('The intervals where the function has at least a root:\n')
        for i in intervals:
            print(f'{i[0]:.10g}\t\t{i[1]:.10g}')
        print(' ')
		
        # plotting intervals____________________
        if with_plot:
            
            x = np.linspace(a, b, 100)
            y = f(x)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

            ax.plot(x, y, color='red') # plot the function

            for i in intervals:
                ax.plot(i, [0, 0], '-o', color='blue')

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            
            ax.set_xlim(a, b)
            
        return intervals

# ______________________________________________________________________________________

def bisection(function, a, b, tol=1e-5, max_iter=100, conv_hist=True, with_plot=False):
    '''
    This function implements the bisecbtion method for finding the root of
    the input function in the interval [a, b] for which f(a).f(b) < 0.
    
    Returns:
    (root, f(root))
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()

    
    f = function
    iteration = 0
    history = []
    
    a_plot = a
    b_plot = b
    
    x_r_old = f(a) # initial guess for the root, can be any other value.
    eps_r = 100 # initial value for eps_r, can be any value bigger than ~1
    
    # initial check for sign change:____________________
    if np.sign(f(a) * f(b)) == 1:
        raise ValueError(f'Function does not change sign in [{a}, {b}]')
    elif np.sign(f(a) * f(b)) == 0:
        raise Exception(f'Either {a} or {b} is a root of the function!')
    
    while (eps_r >= tol) and (iteration < max_iter):
        
        x_r = (a + b) / 2.
        
        if abs(x_r_old) > 1.e-12:
            eps_r = abs(x_r - x_r_old) / abs(x_r_old)
        else:
            eps_r = abs(x_r - x_r_old)
        
        # convergence history____________________
        history.append([iteration, a, b, x_r, f(x_r), eps_r])
            
        # main root check block
        iteration += 1
        if iteration == max_iter: print('Max iteration reached!')
            
        if np.sign(f(a) * f(x_r)) == -1:
            b = x_r
        elif f(x_r) == 0:
            return x_r, f(x_r)
        else:
            a = x_r
            
        x_r_old = x_r
    
    # show convergence history____________________
    if conv_hist:
        import pandas as pd
        df = pd.DataFrame(history, columns=['Iteration', '$ a $', '$ b $', '$ x_r $', '$ f(x_r) $', '$ \epsilon_r $'])
        pd.options.display.float_format = '{:.10g}'.format
        display(df)
        
    # plotting the root____________________    
    if with_plot:
        
        x = np.linspace(a_plot, b_plot, 100)
        y = f(x)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

        # plot the function
        f_curve = ax.plot(x, y, color='red', label='Function')
        # plot the root
        ax.plot(x_r, f(x_r), 'o', color='blue', markersize=5, label='Root')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.set_xlim(a_plot, b_plot)
        
        ax.legend(loc='best')
        fig.tight_layout()
        
    return x_r, f(x_r)

# ______________________________________________________________________________________

def false_position(function, a, b, tol=1e-5, max_iter=100, conv_hist=True, with_plot=False):
    '''
    This function implements the false position method for finding the root of
    the input function in the interval [a, b] for which f(a).f(b) < 0.
    
    Returns:
    (root, f(root))
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    f = function
    iteration = 0
    
    a_plot = a
    b_plot = b
    
    x_r_old = f(a) # initial guess for the root, can be any other value.
    eps_r = 100 # initial value for eps_r, can be any value bigger than ~1
    
    # initial check for sign change:____________________
    if np.sign(f(a) * f(b)) == 1:
        raise ValueError(f'Function does not change sign in [{a}, {b}]')
    elif np.sign(f(a) * f(b)) == 0:
        raise Exception(f'Either {a} or {b} is a root of the function!')
    
    history = []
    
    while (eps_r >= tol) and (iteration < max_iter):
                
        x_r = a - ((b - a) * f(a)) / (f(b) - f(a))

        if abs(x_r_old) > 1.e-12:
            eps_r = abs(x_r - x_r_old) / abs(x_r_old)
        else:
            eps_r = abs(x_r - x_r_old)
        
        # convergence history____________________
        history.append([iteration, a, b, x_r, f(x_r), eps_r])
        
        # main root check block
        iteration += 1
        if iteration == max_iter: print('Max iteration reached!')
            
        if f(a) * f(x_r) < 0:
            b = x_r
        elif f(x_r) == 0:
            return x_r, f(x_r)
        else:
            a = x_r
    
        x_r_old = x_r
        
    # show convergence history____________________
    if conv_hist:
        import pandas as pd
        df = pd.DataFrame(history, columns=['Iteration', '$ a $', '$ b $', '$ x_r $', '$ f(x_r) $', '$ \epsilon_r $'])
        pd.options.display.float_format = '{:.10g}'.format
        display(df)
        
    # plotting the root____________________
    if with_plot:
        x = np.linspace(a_plot, b_plot, 100)
        y = f(x)

        fig, ax = plt.subplots(figsize=(4, 4))
        
        # x,y=0 axis
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

        # plot the function
        f_curve = ax.plot(x, y, color='red', label='Function')
        # plot the root
        ax.plot(x_r, f(x_r), 'o', color='blue', markersize=5, label='Root')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.set_xlim(a_plot, b_plot)
        
        ax.legend(loc='best')
        
    return x_r, f(x_r)

# ______________________________________________________________________________________

def Newton_Raphson(function, derivative, a, tol=1e-5, max_iter=100, conv_hist=True):
    '''
    This function implements the Newton-Raphson method. You need to supply
    the derivative of the function in addition to the function itself, but
    only one initial guess is required.
    
    Returns:
    (root, f(root))
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    f = function
    df = derivative
    
    iteration = 0
    
    x_r_old = f(a) # initial guess for the root, can be any other value.
    eps_r = 100 # initial value for eps_r, can be any value bigger than ~1
    
    history = []
    
    while (eps_r >= tol) and (iteration < max_iter):
        
        x_r = a - (f(a) / df(a))
        
        if abs(x_r_old) > 1.e-12:
            eps_r = abs(x_r - x_r_old) / abs(x_r_old)
        else:
            eps_r = abs(x_r - x_r_old)
        
        # convergence history____________________
        history.append([iteration, a, x_r, f(x_r), eps_r])

        iteration += 1
        if iteration == max_iter: print('Max iteration reached!')
        
        a = x_r
        x_r_old = x_r
    
    # show convergence history____________________
    if conv_hist:
        import pandas as pd
        df = pd.DataFrame(history, columns=['Iteration', '$ a $', '$ x_r $', '$ f(x_r) $', '$ \epsilon_r $'])
        pd.options.display.float_format = '{:.10g}'.format
        display(df)
        
    return x_r, f(x_r)

# ______________________________________________________________________________________

def secant(function, a, delta_x=1e-10, tol=1e-5, max_iter=100, conv_hist=True):
    '''
    This function implements the secant method. Contrary to the Newton-Raphson
    method, there is no need to supply the function derivate. The derivative 
    is estimated by a finite-difference approximation with the step size of 
    delta_x.
    
    Returns:
    (root, f(root))
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    f = function
    df = lambda a: (f(a + delta_x) - f(a)) / delta_x
    
    iteration = 0
    
    x_r_old = f(a) # initial guess for the root, can be any other value.
    eps_r = 100 # initial value for eps_r, can be any value bigger than ~1
    
    history = []
    
    while (eps_r >= tol) and (iteration < max_iter):
        
        x_r = a - (f(a) / df(a))
        
        if abs(x_r_old) > 1.e-12:
            eps_r = abs(x_r - x_r_old) / abs(x_r_old)
        else:
            eps_r = abs(x_r - x_r_old)
        
        # convergence history____________________
        history.append([iteration, a, x_r, f(x_r), eps_r])

        iteration += 1
        if iteration == max_iter: print('Max iteration reached!')
        
        a = x_r
        x_r_old = x_r
    
    # show convergence history____________________
    if conv_hist:
        import pandas as pd
        df = pd.DataFrame(history, columns=['Iteration', '$ a $', '$ x_r $', '$ f(x_r) $', '$ \epsilon_r $'])
        pd.options.display.float_format = '{:.10g}'.format
        display(df)
        
    return x_r, f(x_r)

# ______________________________________________________________________________________

def golden_minimum(function, a, b, tol=1e-5, max_iter=100, conv_hist=True, with_plot=False):
    '''
    This function implements the golden number minimum search for finding the minimum of
    the input function in the interval [a, b].
    
    Returns:
    (xmin, f(xmin))
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    f = function
    iteration = 0
    history = []
    
    phi = ( 1. + 5.**0.5 ) /  2.;
    
    a_plot = a
    b_plot = b
    
    xmin = a # initial guess for the min, can be any other value.
    eps_r = 1 # initial value for eps_r, can be any value bigger than ~1
    
    while (eps_r >= tol) and (iteration < max_iter):
        
        d = ( phi - 1. ) * ( b - a )
        x1 = a + d
        x2 = b - d
        
        # Compute f at x1 and x2
        fx1 = f(x1)
        fx2 = f(x2)
        
        # Update a, b and xmin
        if fx1 < fx2:
            xmin = x1
            a = x2
            fxmin = fx1
        else:
            xmin = x2
            b = x1
            fxmin = fx2
            
        # Compute relative error eps_r
        eps_r = ( 2. - phi ) * np.abs( b - a )
        if abs(xmin) > 1.e-12:
            eps_r = eps_r / np.abs( xmin )
        
        # convergence history____________________
        history.append([iteration, a, b, xmin, fxmin, eps_r])
            
        # main minimum check block
        iteration += 1
        if iteration == max_iter: print('Max iteration reached!')
            
    # show convergence history____________________
    if conv_hist:
        import pandas as pd
        df = pd.DataFrame(history, columns=['Iteration', '$ a $', '$ b $', '$ x_{min} $', '$ f(x_{min}) $', '$ \epsilon_r $'])
        pd.options.display.float_format = '{:.10g}'.format
        display(df)
        
    # plotting the minimum____________________    
    if with_plot:
        
        x = np.linspace(a_plot, b_plot, 100)
        y = f(x)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

        # plot the function
        f_curve = ax.plot(x, y, color='red', label='Function')
        # plot the minimum
        ax.plot(xmin, fxmin, 'o', color='blue', markersize=5, label='Minimum')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.set_xlim(a_plot, b_plot)
        
        ax.legend(loc='best')

    return xmin, fxmin

# ______________________________________________________________________________________