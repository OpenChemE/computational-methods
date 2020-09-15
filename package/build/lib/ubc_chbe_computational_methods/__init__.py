import ubc_chbe_computational_methods.root_finding
import ubc_chbe_computational_methods.root_finding
import ubc_chbe_computational_methods.linear_nonlinear_systems
import ubc_chbe_computational_methods.regression
import ubc_chbe_computational_methods.numerical_integration
import ubc_chbe_computational_methods.ode_solve

def load_matplotlib_settings():
    '''
    This part is for matplotlib configuration.
    '''
    
    import matplotlib as mpl

    mpl.rcParams.update({'mathtext.fontset': 'cm'})
    mpl.rcParams.update({'axes.labelsize': 20})
    mpl.rcParams.update({'axes.titlesize': 16})
    mpl.rcParams.update({'axes.linewidth': 0.5})
    mpl.rcParams.update({'axes.labelpad': 12})
    mpl.rcParams.update({'xtick.labelsize': 10})
    mpl.rcParams.update({'ytick.labelsize': 10})