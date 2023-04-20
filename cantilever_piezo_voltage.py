# This code calculated the output voltage of a piezoelectric cantilever
# The model is cantilever with tip mass and a piezoelectric layer
import numpy as np
from scipy.optimize import root


# 1 # cantilever film dimensions
M1 = 11 * 1e-6                  ## tip mass in kg (referring to raindrop)
J1 = M1 * ( 5e-3 )**2           # tip mass moment of inertia
Ys = 0.5 * 1e9                  # substrate young's modulus
rho_s = 1390                    # substarte density
Yp = 2.45 * 1e9                 # piezo young's moudulus
rho_p = 1780                    # piezo density
h1s = 0.5 * 1e-3                ## substrate thickness(m)
h1p = 0.028 * 1e-3              # piezo thickness(m)
L1 = 20 * 1e-3                  ## substrate length(m)
Lp1 = 0 * L1                    # 0 means the piezo cover from the fixed end
Lp2 = 1 * L1                    # piezo length
b1s = 13 * 1e-3                 ## substrate width
b1p = b1s                       # piezo width
b1pp = b1p                      # electrode width

# 2 # piezoelectric parameters
d31 = -190 * 1e-12              # piezoelectric constant(m/V)
vtheta = 1e-7                   ## piezoelectric coupling coefficient, vtheta = Yp*d31*b1pp*h1pc
epsilon = 15.93 * 1e-9          # absolute permittivity 3500*8.854*1e-12 # F/m
Cp = 0.5 * 1e-9                 ## Capaticance of the piezo layer

# 3 # cantilever mechanical parameters
m1 = rho_s*b1s*h1s + rho_p*b1p*h1p                             # unit mass of the cantilever
n = Ys * b1s / ( Yp * b1p )                                         # stiffness ratio of substrate to piezo modulus
h1pa = ( h1p**2 + 2*n*h1p*h1s + n * h1s**2 ) / 2 / ( h1p + n*h1s )  # piezo layer surface distance to neutral axis
h1sa = ( h1p**2 + 2*h1p*h1s + n * h1s**2 ) / 2 / ( h1p + n*h1s )    # substrate layer surface distance to neutral axis
h1pc = n * h1s * ( h1p+h1s ) / 2 / ( h1p + n*h1s )                  # 
h1a = -h1sa                                                         # distance from neutral axis to bottom of substrate
h1b = h1pa - h1p                                                    # distance from neutral axis to top of substrate
h1c = h1pa                                                          # distance from neutral axis to top of piezo layer
EI1 = b1s/3*Ys*(h1b**3-h1a**3)+b1p/3*Yp*(h1c**3-h1b**3)                 # bending stiffness of the cantilever

# 4 # find natural frequency by det(A) = 0
    # where A x C = 0. C is the vector of the 4 unknowns (C1, C2, C3, C4)
    # C is coefficient of eigenfunction ( W(x) )of deflection function ( w(x, t) )
    # A is the matrix of the coefficients of the unknowns
    # Refer to: Analysis and Verification of V-shaped Piezoelectric Energy Harvesters with Angle and Tip Mass

#
def det_A_v1(alpha: np.float64) -> float:
    # natural frequency omega_i = alpha_i * sqrt( EI1 / m1 )
    # Here we only consider the first natural frequencies
    # formula (3.1.47)
    cos = np.cos(alpha*L1)
    sin = np.sin(alpha*L1)
    cosh = np.cosh(alpha*L1)
    sinh = np.sinh(alpha*L1)
    det_value = 1 + cos*cosh + \
                (M1*alpha/m1)*(cos*sinh-sin*cosh) - \
                (J1*alpha**3/m1)*(cosh*sin+sinh*cos) + \
                (M1*J1*alpha**4/m1**2)*(1-cos*cosh)
            
    return det_value

def det_A_v2(alpha: np.float64) -> float:
    # natural frequency omega_i = alpha_i * sqrt( EI1 / m1 )
    # Here we only consider the first natural frequencies
    # formula (3.1.47)
    cos = np.cos(alpha*L1)
    sin = np.sin(alpha*L1)
    cosh = np.cosh(alpha*L1)
    sinh = np.sinh(alpha*L1)
    det_value = 1 + cos*cosh + \
                (M1*alpha/m1)*(cos*sinh-sin*cosh) + \
                (J1*alpha**3/m1)*(cosh*sin+sinh*cos) + \
                (M1*J1*alpha**4/m1**2)*(cos*cosh-1)
            
    return det_value

print(det_A_v1(50), det_A_v2(50))