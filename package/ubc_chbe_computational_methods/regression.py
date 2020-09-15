'''
This module contains various regression methods.
'''

# ______________________________________________________________________________________

def linear_regression(x, y, with_plot=True, with_output=True):
    '''
    This function finds the coefficients a = [a_1, a_0] 
    associated with the best linear fit to the data.
    
    Returns:
    (a, r_squared)
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    # preliminary check____________________
    if len(x) != len(y):
        raise ValueError('Size of vectors x and y should be equal.')
        
    n = len(x)
    s_x = sum(x)
    s_y = sum(y)
    s_xy = sum(x * y)
    s_x2 = sum(x**2)
    
    a = np.zeros(2)
    a[1] = ((n * s_xy) - (s_x * s_y)) / ((n * s_x2) - (s_x)**2)
    a[0] = (s_y - a[1] * s_x) / n
    
    # compute R^2
    s_r = sum((y - a[0] - a[1] * x)**2)
    s_t = sum((y - np.mean(y))**2)
    r_squared = 1 - (s_r / s_t)
    
    # output____________________
    if with_output:
        print(f'''\ra_0 = {a[0]:.6g}
                  \ra_1 = {a[1]:.6g}\n
                  \rR^2 = {r_squared:.4f}\n''')

    # plotting____________________    
    if with_plot:
        
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = a[0] + a[1] * x_fit

        fig, ax = plt.subplots()
        
        # make plot area larger to see things more clearly
        factor = 0.1
        dx = max(np.hstack((x_fit, x))) - min(np.hstack((x_fit, x)))
        x_l = min(np.hstack((x_fit, x))) - factor * dx
        x_u = max(np.hstack((x_fit, x))) + factor * dx

        dy = max(np.hstack((y_fit, y))) - min(np.hstack((y_fit, y)))
        y_l = min(np.hstack((y_fit, y))) - factor * dy
        y_u = max(np.hstack((y_fit, y))) + factor * dy

        ax.set_xlim(x_l, x_u)
        ax.set_ylim(y_l, y_u)
        
        # plot the data
        ax.scatter(x, y, s=30, color='red', label='Data', alpha=0.6)
        # plot the linear fit
        ax.plot(x_fit, y_fit, '-', color='black', linewidth=2, label='Linear fit')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.legend(loc='best')
        ax.text(0.95, 0.05, f'$R^2 = {r_squared:.2f}$',
                transform=ax.transAxes,
                fontsize=16,
                horizontalalignment='right')
        
    return a, r_squared

# ______________________________________________________________________________________

def quadratic_regression(x, y, with_plot=True, with_output=True):
    '''
    This function finds the coefficients a = [a_0, a_1, a_2] associated
    with the best quadratic fit to the data of the form y = a_0 + a_1*x + a_2*x^2.
    
    Returns:
    (a, r_squared)
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    # preliminary check____________________
    if len(x) != len(y):
        raise ValueError('Size of vectors x and y should be equal.')
        
    n = len(x)
    s_x = sum(x)
    s_y = sum(y)
    s_xy = sum(x * y)
    s_x2y = sum(x**2 * y)
    s_x2 = sum(x**2)
    s_x3 = sum(x**3)
    s_x4 = sum(x**4)
    
    # fitting____________________
    # construct the matrix of coefficients [A]:
    A = np.array([
        [n, s_x, s_x2],
        [s_x, s_x2, s_x3],
        [s_x2, s_x3, s_x4]
    ])
    # construct the vector of contants [b]:
    b = np.array([
        [s_y],
        [s_xy],
        [s_x2y]
    ])
    
    # the unknowns a_0, a_1 and a_2 are solved for:
    import linear_nonlinear_systems as lnls
    
    a = lnls.GaussElimination(A, b)
    a = a.flatten() # this is done to get a vector instead of a matrix
    
    # compute R^2____________________
    s_r = sum((y - a[0] - a[1] * x - a[2] * x**2)**2)
    s_t = sum((y - np.mean(y))**2)
    r_squared = 1 - (s_r / s_t)
    
    # output____________________
    if with_output:
        print(f'''\ra_0 = {a[0]:.6g}
                  \ra_1 = {a[1]:.6g}
                  \ra_2 = {a[2]:.6g}\n
                  \rR^2 = {r_squared:.4f}\n''')
    
    # plotting____________________
    if with_plot:
        
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = a[0] + a[1] * x_fit + a[2] * x_fit**2

        fig, ax = plt.subplots()
        
        # make plot area larger to see things more clearly
        factor = 0.1
        dx = max(np.hstack((x_fit, x))) - min(np.hstack((x_fit, x)))
        x_l = min(np.hstack((x_fit, x))) - factor * dx
        x_u = max(np.hstack((x_fit, x))) + factor * dx

        dy = max(np.hstack((y_fit, y))) - min(np.hstack((y_fit, y)))
        y_l = min(np.hstack((y_fit, y))) - factor * dy
        y_u = max(np.hstack((y_fit, y))) + factor * dy

        ax.set_xlim(x_l, x_u)
        ax.set_ylim(y_l, y_u)
        
        # plot the data
        ax.scatter(x, y, s=30, color='red', label='Data', alpha=0.6)
        # plot the linear fit
        ax.plot(x_fit, y_fit, '-', color='black', linewidth=2, label='Quadratic fit')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.legend()
        ax.text(0.95, 0.05, f'$R^2 = {r_squared:.2f}$',
                transform=ax.transAxes,
                fontsize=16,
                horizontalalignment='right')
        
    return a, r_squared

# ______________________________________________________________________________________

def polynomial_regression(x, y, order=1, with_plot=True, with_output=True):
    '''
    This function finds the coefficients of the best polynomial fit
    of the form y = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + ... + a_n*x^n
    stored in the vector a.
    
    Returns:
    (a, r_squared)
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    load_matplotlib_settings()
    
    # preliminary check____________________
    if len(x) != len(y):
        raise ValueError('Size of vectors x and y should be equal.')

    if len(x) == 1:
        raise ValueError('Provide at least 2 data points.')

    n = len(x)

    if order > n - 1:
        raise ValueError('Order of polynomial too high for this number of data points.')

    sum_xp = np.zeros(2 * order + 1)
    for p in range(2 * order + 1):
        sum_xp[p] = sum(x**p)

    sum_yxp = np.zeros(order + 1)
    for p in range(order + 1):
        sum_yxp[p] = sum(y * x**p)

    # fitting____________________
    # construct the matrix of coefficients [A]:
    A = np.zeros((order + 1, order + 1))

    # construct the vector of contants [b]:
    b = np.zeros(order + 1)

    for k in range(order + 1):
        b[k] = sum_yxp[k]
        for l in range(order + 1):
            A[k, l] = sum_xp[k + l]

    # the unknowns coefficients are solved for:
    import linear_nonlinear_systems as lnls
    
    a = lnls.GaussElimination(A, b)
    a = a.flatten() # this is done to get a vector instead of a matrix

    # compute R^2____________________
    y_fit = np.zeros(len(x))
    for i in range(n):
            y_fit[i] = a[0]
            for j in range(1, order + 1):
                y_fit[i] = y_fit[i] + a[j] * x[i]**j
                
    s_r = sum((y - y_fit)**2)
    s_t = sum((y - np.mean(y))**2)
    r_squared = 1 - (s_r / s_t)

    # output____________________
    if with_output:
        for i in range(len(a)):
            print(f'a_{i} = {a[i]:.6g}')
            
        print(f'\nR^2 = {r_squared:.4f}\n')

    # plotting____________________
    if with_plot:
        
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = np.zeros(1000)

        for i in range(len(x_fit)):
            y_fit[i] = a[0]
            for j in range(1, order + 1):
                y_fit[i] = y_fit[i] + a[j] * x_fit[i]**j

        fig, ax = plt.subplots()

        # make plot area larger to see things more clearly
        factor = 0.1
        dx = max(np.hstack((x_fit, x))) - min(np.hstack((x_fit, x)))
        x_l = min(np.hstack((x_fit, x))) - factor * dx
        x_u = max(np.hstack((x_fit, x))) + factor * dx

        dy = max(np.hstack((y_fit, y))) - min(np.hstack((y_fit, y)))
        y_l = min(np.hstack((y_fit, y))) - factor * dy
        y_u = max(np.hstack((y_fit, y))) + factor * dy

        ax.set_xlim(x_l, x_u)
        ax.set_ylim(y_l, y_u)

        # plot the data
        ax.scatter(x, y, s=30, color='red', label='Data', alpha=0.6)
        # plot the fit
        ax.plot(x_fit, y_fit, '-', color='black', linewidth=2, label='Polynomial fit')
        
        ax.set_title(f'Polynomial order = {order}',
                     fontname='Times New Roman',
                     fontsize=16)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.legend()
        ax.text(0.95, 0.05, f'$R^2 = {r_squared:.2f}$',
                transform=ax.transAxes,
                fontsize=16,
            horizontalalignment='right')
        
    return a, r_squared

# ______________________________________________________________________________________

def Generalized_Least_Squares(x, y, functions, with_plot=True, with_output=True):
    '''
    This function computes the coefficients of the basis functions in
    the Generalized Least-Squares (GLS) regression method of the form
    y = a_0*z_0(x) + a_1*z_1(x)  + a_2*z_2(x) + a_3*z_3(x) + ... + a_n*z_n(x)
    stored in the vector a.
    
    Returns:
    (a, r_squared)
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # preliminary check____________________
    if len(x) != len(y):
        raise ValueError('Size of vectors x and y should be equal.')

    if len(x) == 1:
        raise ValueError('Provide at least 2 data points.')
    
    n = len(x)
    
    Z = np.zeros((n, 0))
    
    for i in range(len(functions)):
        col = functions[i](x).reshape(-1, 1)
        Z = np.hstack((Z, col))
    
    # the unknowns coefficients are solved for:
    import linear_nonlinear_systems as lnls
    A = Z.T @ Z
    b = Z.T @ y.T
    
    a = lnls.GaussElimination(A, b)
    a = a.flatten() # this is done to get a vector instead of a matrix

    # compute R^2____________________
    y_fit = np.zeros(len(x))
    for i in range(len(x)):
            for j in range(len(functions)):
                y_fit[i] = y_fit[i] + a[j] * functions[j](x[i])
                
    s_r = sum((y - y_fit)**2)
    s_t = sum((y - np.mean(y))**2)
    r_squared = 1 - (s_r / s_t)

    # output____________________
    if with_output:
        for i in range(len(a)):
            print(f'a_{i} = {a[i]:.6g}')
            
        print(f'\nR^2 = {r_squared:.4f}\n')

    # plotting____________________
    if with_plot:
        
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = np.zeros(1000)

        for i in range(len(x_fit)):
            for j in range(len(functions)):
                y_fit[i] = y_fit[i] + a[j] * functions[j](x_fit[i])

        fig, ax = plt.subplots()

        # make plot area larger to see things more clearly
        factor = 0.1
        dx = max(np.hstack((x_fit, x))) - min(np.hstack((x_fit, x)))
        x_l = min(np.hstack((x_fit, x))) - factor * dx
        x_u = max(np.hstack((x_fit, x))) + factor * dx

        dy = max(np.hstack((y_fit, y))) - min(np.hstack((y_fit, y)))
        y_l = min(np.hstack((y_fit, y))) - factor * dy
        y_u = max(np.hstack((y_fit, y))) + factor * dy

        ax.set_xlim(x_l, x_u)
        ax.set_ylim(y_l, y_u)

        # plot the data
        ax.scatter(x, y, s=30, color='red', label='Data', alpha=0.6)
        # plot the fit
        ax.plot(x_fit, y_fit, '-', color='black', linewidth=2, label='GLS fit')
        
        ax.set_title(f'Generalized Least Squares (GLS) regression',
                     fontname='Times New Roman',
                     fontsize=14)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.legend()
        ax.text(0.95, 0.05, f'$R^2 = {r_squared:.2f}$',
                transform=ax.transAxes,
                fontsize=16,
            horizontalalignment='right')
        
    return a, r_squared

# ______________________________________________________________________________________