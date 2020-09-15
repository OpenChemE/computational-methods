'''
This module contains various methods for solving ordinary differential equations.
'''

# ______________________________________________________________________________________

def euler_ode(dy_dt, t, y0, with_plot=False):
    '''
    This function solves a first-order ODE of the type dy/dy = f(t, y) over a specified time-span.
    
    Returns:
    y = Numerical solution vector of the ODE
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # step size
    h = t[1] - t[0]
    
    y = np.zeros(t.size)

    # setting initial condition
    y[0] = y0

    # solving in time _______________________________
    for i in range(0, t.size - 1):
        y[i+1] = y[i] + dy_dt(t[i], y[i]) * h
    
    # plotting _______________________________
    if with_plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, y, linewidth=3, marker='o', markerfacecolor='none', color='black')
        
        ax.set_title('Euler\'s Method')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.grid(linewidth=0.5, alpha=0.25)
        
    return y

# ______________________________________________________________________________________

def midpoint_ode(dy_dt, t, y0, with_plot=False):
    '''
    This function solves a first-order ODE of the type dy/dy = f(t, y) over a specified time-span.
    
    Returns:
    y = Numerical solution vector of the ODE
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # step size
    h = t[1] - t[0]
    
    y = np.zeros(t.size)

    # setting initial condition
    y[0] = y0

    # solving in time _______________________________
    for i in range(0, t.size - 1):
        t_mid = (t[i] + t[i+1]) / 2
        y_mid = y[i] + dy_dt(t[i], y[i]) * (h / 2)
        
        y[i+1] = y[i] + dy_dt(t_mid, y_mid) * h
    
    # plotting _______________________________
    if with_plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, y, linewidth=3, marker='o', markerfacecolor='none', color='blue')
        
        ax.set_title('Midpoint Method')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.grid(linewidth=0.5, alpha=0.25)
        
    return y

# ______________________________________________________________________________________

def RK4_ode(dy_dt, t, y0, with_plot=False):
    '''
    This function solves a first-order ODE of the type dy/dy = f(t, y) over a specified time-span
    using the classical fourht-order Runge-Kutta method.
    
    Returns:
    y = Numerical solution vector of the ODE
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # step size
    h = t[1] - t[0]
    
    y = np.zeros(t.size)

    # setting initial condition
    y[0] = y0

    # solving in time _______________________________
    for i in range(0, t.size - 1):
        
        if np.isnan(y[i]):
            y[i] = 0
            break
            
        k1 = dy_dt(t[i], y[i])
        k2 = dy_dt(t[i] + 0.5 * h, y[i] + 0.5 * k1 * h)
        k3 = dy_dt(t[i] + 0.5 * h, y[i] + 0.5 * k2 * h)
        k4 = dy_dt(t[i] + h, y[i] + k3 * h)

        y[i+1] = y[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h
            
    # plotting _______________________________
    if with_plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, y, linewidth=3, color='olive')
        
        ax.set_title('4th-Order Runge-Kutta Method')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.grid(linewidth=0.5, alpha=0.25)
        
    return y

# ______________________________________________________________________________________

def euler_ode_sys(dy_dt, t, y0):
    '''
    This function solves a system of first-order ODEs of the type
    [dy1/dt, dy2/dt, ..., ] = [f1(t, y1, y2, ...), f2(t, y1, y2, ...), ...]
    over a specified time-span using Euler's method.
    
    Input:
    dy_dt: list of RHS functions
    y0: list of initial conditions
    
    Returns:
    sol = Numerical solution vector of the ODE with each column corresponding
    to each unknown function y1, y2, ..., yn.
    '''
    
    import numpy as np
    
    # step size
    h = t[1] - t[0]
    
    # number of equations _______________________________
    n = len(y0)
    
    sol = np.zeros((t.size, n))

    # setting initial condition _______________________________
    for j in range(n):
        sol[0, j] = y0[j]

    # solving in time _______________________________
    for i in range(0, t.size - 1):
        
        # loop over all equations
        for j in range(n):
            
            RHS = dy_dt(t[i], sol[i, :])
            sol[i+1, j] = sol[i, j] + RHS[j] * h
        
    return sol

# ______________________________________________________________________________________

def RK4_ode_sys(dy_dt, t, y0):
    '''
    This function solves a system of first-order ODEs of the type
    [dy1/dt, dy2/dt, ..., ] = [f1(t, y1, y2, ...), f2(t, y1, y2, ...), ...]
    over a specified time-span using the classical fourth-order Runge-Kutta method.
    
    Input:
    dy_dt: list of RHS functions
    y0: list of initial conditions
    
    Returns:
    sol = Numerical solution vector of the ODE with each column corresponding
    to each unknown function y1, y2, ..., yn.
    '''
    
    import numpy as np
    
    # step size
    h = t[1] - t[0]
    
    # number of equations _______________________________
    n = len(y0)
    
    sol = np.zeros((t.size, n))

    # setting initial condition _______________________________
    for j in range(n):
        sol[0, j] = y0[j]
    
    # solving in time _______________________________
    for i in range(0, t.size - 1):
        
        # loop over all equations
        for j in range(n):
            
            k1 = dy_dt(t[i], sol[i, :])
            
            k2 = dy_dt(t[i] + 0.5 * h, sol[i, :] + 0.5 * k1 * h)
            k3 = dy_dt(t[i] + 0.5 * h, sol[i, :] + 0.5 * k2 * h)
            k4 = dy_dt(t[i] + h, sol[i, :] + k3 * h)
            
            phi = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            sol[i+1, j] = sol[i, j] + phi[j] * h
        
    return sol

# ______________________________________________________________________________________