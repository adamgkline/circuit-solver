import torch as tc
# ReLU = tc.nn.ReLU()

class ResistiveElement(object):
    """
    Class for nonlinear resistive elements that admit a cocontent. Allows for arbitrary number of parameters.

    Make sure that cocontent function rho is defined with torch functions.

    """
    def __init__(self, rho, N_params, name, type=None, param_ranges=None, init_mode='geometric_mean', init_params=None):
        self.rho = rho
        self.N_params = N_params
        self.name = name
        self.type = type
        # param_ranges: list of (min, max) tuples, one for each parameter
        # None values mean no constraint for that bound
        self.param_ranges = param_ranges or [(None, None)] * N_params
        # init_mode: initialization mode ('constant', 'uniform', 'normal', 'geometric_mean')
        self.init_mode = init_mode
        # init_params: parameters for initialization (depends on init_mode)
        # For 'constant': single value or list of values (one per parameter)
        # For 'uniform': not needed (uses param_ranges)
        # For 'normal': (mean, std) tuple or list of tuples
        # For 'geometric_mean': not needed (uses param_ranges)
        self.init_params = init_params
    
    def __str__(self):
        return self.name

    def __repr__(self):
        # doing this because there should never be more than one instance of a resistive element at a time.
        return self.name

    def get_shape(self):
        # return (self.N_params,)
        return self.N_params

    def rho(self, x, theta):
        return self.rho(x, theta)
    
    def gamma(self, x, theta):
        func = lambda x : self.rho(x, theta)
        return tc.func.grad(func)(x)

    def d_gamma_d_x(self, x, theta):
        func = lambda x : self.gamma(x, theta)
        return tc.func.grad(func)(x)

    def d_gamma_d_theta(self, x, theta):
        # func = lambda x : self.d_rho_d_theta(x, theta)
        # return tc.func.grad(func)(x)
        func = lambda theta : self.gamma(x, theta)
        return tc.func.grad(func)(theta)

    def d_rho_d_theta(self, x, theta):
        func = lambda theta : self.rho(x, theta)
        return tc.func.grad(func)(theta)


# Cocontent functions
def rho_diode(x, theta):
    return theta[0] / theta[1] * (tc.exp(theta[1] * x) - theta[1] * x - 1)

def rho_adj_diode(x, theta):
    return rho_diode(x - theta[2], theta[:2])

def rho_adj_diode_II(x, theta):
    return theta[0] / theta[1] * (tc.exp(theta[1] * x - theta[2]) - theta[1] * x + theta[2] - 1)

def rho_ideal_diode(x, theta):
    # return theta[0] / 2 * ReLU(x) * x
    return theta[0] / 2 * tc.relu(x) * x

def rho_linear(x, theta):
    return 1/2 * theta[0] * x**2

def rho_diode_sym(x, theta):
    return rho_diode(x, theta) + rho_diode(-x, theta)

def rho_ideal_diode_sym(x, theta):
    return rho_ideal_diode(x, theta) + rho_ideal_diode(-x, theta)

def rho_adj_ideal_diode(x, theta):
    return rho_ideal_diode(x - theta[1], theta[0][None,...])






# Reverse-direction components
# Use a callable class instead of lambda so it can be pickled
class ReversedFunction:
    """Wrapper class that reverses a function's first argument. Picklable."""
    def __init__(self, func):
        self.func = func

    def __call__(self, x, theta):
        return self.func(-x, theta)

def reverse(func):
    """Create a reversed version of a cocontent function that can be pickled"""
    return ReversedFunction(func)

def reverse_element(resistive_element):
    name = resistive_element.name
    rho = reverse(resistive_element.rho)
    return ResistiveElement(rho,
        N_params=resistive_element.N_params,
        name=name,
        param_ranges=resistive_element.param_ranges,
        init_mode=resistive_element.init_mode,
        init_params=resistive_element.init_params)

def element_generator(*args, **kwargs):
    """
    Factory function that returns a callable generator for creating ResistiveElement instances.

    Returns a callable that creates ResistiveElement instances, allowing override of any parameter.
    """
    def generator(**override_kwargs):
        # Merge default kwargs with overrides (overrides take precedence)
        final_kwargs = {**kwargs, **override_kwargs}
        return ResistiveElement(*args, **final_kwargs)
    return generator


# Asymmetric resistive elements
Diode = element_generator(rho_diode, N_params=2, name='Diode', type='Diode', param_ranges=[(1E-1, 1E1), (1E-1, 1E1)])  # [conductance, steepness]

AdjDiode = element_generator(rho_adj_diode, N_params=3, name='AdjDiode', type='AdjDiode', param_ranges=[(1E-1, 1E1), (1E-1, 1E1), (-1,1)]) # [conductance, steepness, offset]

AdjDiodeII = element_generator(rho_adj_diode_II, N_params=3, name='AdjDiode', type='AdjDiode', param_ranges=[(1E-1, 1E1), (1E-1, 1E1), (-1,1)])

IdealDiode = element_generator(rho_ideal_diode, N_params=1, name='IdealDiode',type='IdealDiode', param_ranges=[(1E-1, 1E1)])  # [conductance]

AdjIdealDiode = element_generator(rho_adj_ideal_diode, N_params=2, name='AdjIdealDiode', type='AdjIdealDiode', param_ranges=[(1E-1,1E1,-1,1)]) # [conductance, offset]



# Symmetric resistive elements  
DiodeSym = element_generator(rho_diode_sym, N_params=2, name='DiodeSym', type='DiodeSym', param_ranges=[(1E-2, 1E2), (1E-1, 1E1)])
IdealDiodeSym = element_generator(rho_ideal_diode_sym, N_params=2, name='IdealDiodeSym', type='IdealDiodeSym', param_ranges=[(1E-2, 1E2), (1E-2, 1E2)])
Resistor = element_generator(rho_linear, N_params=1, name='Resistor', type='Resistor', param_ranges=[(1E-2, 1E2)])  # [conductance]



