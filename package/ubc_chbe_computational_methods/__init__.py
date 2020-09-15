import root_finding
import linear_nonlinear_systems
import regression
import numericalintegration
import ode_solve

def load_matplotlib_settings():
    '''
    This part is for matplotlib configuration
    '''
    
    import matplotlib as mpl

    mpl.rcParams.update({'mathtext.fontset': 'cm'})
    mpl.rcParams.update({'axes.labelsize': 22})
    mpl.rcParams.update({'axes.titlesize': 16})
    mpl.rcParams.update({'axes.linewidth': 0.5})
    mpl.rcParams.update({'axes.labelpad': 12})
    mpl.rcParams.update({'xtick.labelsize': 10})
    mpl.rcParams.update({'ytick.labelsize': 10})