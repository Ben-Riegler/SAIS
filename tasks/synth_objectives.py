
import torch

from botorch.utils.transforms import unnormalize

from botorch.test_functions.synthetic import (Ackley, 
                                              EggHolder, 
                                              Cosine8, 
                                              Hartmann, 
                                              Rosenbrock, 
                                              Griewank,
                                              Michalewicz,
                                              Branin,
                                              Powell
                                              )

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ackley_builder(d):
    """
    Build normalized Ackley with unit square input
    """
    x_bounds = torch.tensor([[-32.768 for _ in range(d)], 
                      [32.768 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )
    
    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Ackley(dim = d)(x).unsqueeze(-1)
        return 1 - y/(20+torch.e)
    
    return f

def egg2_builder(d = 2):
    """
    Build normalized Egg holder with unit square input
    must be evaluted in 2D
    """
    if d != 2:
        raise ValueError("Egg holder only defined for 2D")
    
    x_bounds = torch.tensor([[-512 for _ in range(d)], 
                      [512 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = EggHolder(negate=True)(x).unsqueeze(-1)
        return (y + 1049.1316) / (959.6407 + 1049.1316)
    
    return f

def cos8_builder(d=8):
    """
    Build normalized Cos8 holder with unit square input
    must be evaluted in 8D
    """
    if d != 8:
        raise ValueError("Cos8 only defined for 8D")
    
    x_bounds = torch.tensor([[-1 for _ in range(d)], 
                      [1 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Cosine8()(x).unsqueeze(-1)
        return (y + 8.8) / (0.8 + 8.8)
    
    return f

def hart6_builder(d=6):
    """
    Build normalized Hartmann-6 with unit square input
    """
    
    x_bounds = torch.tensor([[0 for _ in range(d)], 
                      [1 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Hartmann(dim = d, negate=True)(x).unsqueeze(-1)
        return (y - 0.) / (3.32237 - 0.)
    
    return f

def ros_builder(d):
    """
    Build normalized Rosenbrock with unit square input
    """

    if d < 2:
        raise ValueError("Rosenbrock only defined for at least 2D")
    
    x_bounds = torch.tensor([[-5 for _ in range(d)], 
                      [10 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Rosenbrock(dim = d, negate=True)(x).unsqueeze(-1)
        return (y + (d-1)*(100**3+81)) / (0. + (d-1)*(100**3+81))
    
    return f

def griewank_builder(d):
    """
    Build normalized Griewank with unit square input
    """

    
    x_bounds = torch.tensor([[-10 for _ in range(d)], 
                      [10 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Griewank(dim =d, negate=True)(x).unsqueeze(-1)
        return (y + 2) / (0. + 2)
    
    return f


def mic_builder(d):
    """
    Build normalized Michalewicz with unit square input
    """

    x_bounds = torch.tensor([[0 for _ in range(d)], 
                             [torch.pi for _ in range(d)]],
                             dtype=dtype,
                             device=device,
                             )
    if d == 2:
        y_max = 1.8013
    elif d == 5:
        y_max = 4.687658
    elif d == 10:
        y_max = 9.66015

    else:
        raise ValueError("d needs to be 2, 5 or 10")
    
    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Michalewicz(dim = d, negate=True)(x).unsqueeze(-1)

        return y / y_max
    
    return f


def bran_builder(d=2):
    """
    Build normalized Branin with unit square input
    """

    x_bounds = torch.tensor([[-5, 0], [10, 15]],
                             dtype=dtype,
                             device=device,
                             )
    
    if d != 2:

        raise ValueError("input needs to be 2D")
    
    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = Branin(negate=True)(x).unsqueeze(-1)

        return (y + 308.1291) / (-0.397887 + 308.1291)
    
    return f

def powell_builder(d=4):
    """
    Build Powell with unit square input
    """

    x_bounds = torch.tensor([[-4 for _ in range(d)], 
                             [5 for _ in range(d)]],
                             dtype=dtype,
                             device=device,
                             )
    
    def f(x):
        
        # unnormalize
        x = unnormalize(x, x_bounds) 
        y = 1 - Powell(dim = d)(x).unsqueeze(-1)

        return y
    
    return f



synOBJECTIVES = {"ackley": ackley_builder,
              "egg2": egg2_builder,
              "cos8": cos8_builder,
              "hart6": hart6_builder,
              "ros": ros_builder,
              "grie": griewank_builder,
              "mic": mic_builder,
              "bran": bran_builder,
              "pow": powell_builder,
              }
