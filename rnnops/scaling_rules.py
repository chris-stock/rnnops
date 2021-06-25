# rules for producing local balancing dynamics

class LocalScalingRule(object):
    """
    Base class for any rule producing somatic state variables consistent with
    the general diagonal-Lax dynamics
    """

    def state(self):
        # returns a vector of states to be used in the matrix update step
        pass

    def weight_updates(self):
        # returns updates to the reccurrent weight matrix
        # TODO: implement Lax dynamical system in terms of state
        pass

    def __init__(self):
        pass


class CostFunctionRule(LocalScalingRule):
    """
    Base class for any rule based on a 
    """

    def __init__(self):
        pass

    def gradient(self):
        # returns the vector dC/dh
        pass

    def state(self):
        return -self.gradient()


class RobustnessCostFunction(CostFunctionRule):

    def __init__(self):
        pass

    def gradient(self):
        #TODO: implement me!
        pass


class ScaledPowerLawCostFunction(CostFunctionRule):
    # implements the so-called Scaled Power Law cost function:
    # c_ij = \alpha_ij | J_ij |^p

    # alpha is the NxN matrix of elementwise scales
    # p is the exponent (any real number allowed) 

    def __init__(self,
        alpha,
        p
        ):

        self.alpha = alpha
        self.p = p

        pass

    def gradient(self):
        #TODO: implement me!
        pass