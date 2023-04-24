# This code calculated the output voltage of a piezoelectric cantilever
# The model is cantilever with tip mass and a piezoelectric layer
import numpy as np
from scipy.optimize import root
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import timeit

class piezo_film():
    def __init__(self):
        # 1 # cantilever film dimensions
        self.M1 = 11 * 1e-6                 ## tip mass in kg (referring to raindrop)
        self.J1 = self.M1 * ( 5e-3 )**2     # tip mass moment of inertia
        self.Ys = 0.5 * 1e9                 # substrate young's modulus
        self.rho_s = 1390                   # substarte density
        self.Yp = 2.45 * 1e9                # piezo young's moudulus
        self.rho_p = 1780                   # piezo density
        self.h1s = 0.5 * 1e-3               ## substrate thickness(m)
        self.h1p = 0.028 * 1e-3             # piezo thickness(m)
        self.L1 = 20 * 1e-3                 ## substrate length(m)
        self.Lp1 = 0 * self.L1              # 0 means the piezo cover from the fixed end
        self.Lp2 = 1 * self.L1              # piezo length
        self.b1s = 13 * 1e-3           ## substrate width
        self.b1p = self.b1s       # piezo width
        self.b1pp = self.b1p           # electrode width

        # 2 # piezoelectric parameters
        self.d31 = -190 * 1e-12             # piezoelectric constant(m/V)
        self.vtheta = 0.85*1e-7             ## piezoelectric coupling coefficient, vtheta = Yp*d31*b1pp*h1pc
        self.epsilon = 15.93 * 1e-9         # absolute permittivity 3500*8.854*1e-12 # F/m
        self.Cp = 0.5 * 1e-9                ## Capaticance of the piezo layer

        # 4 # external circuit parameters
        self.R = 1 * 1e6                    ## external circuit load resistance
        self.zeta = 0.043                   ## damping ratio (Including internal damping and air damping)
        
        self.force = lambda t: 0.08*24
        self.time_end = 0.2
        self.time_step = 1000
        
        self.cal_properties()


    def cal_properties(self):
        # 3 # cantilever mechanical parameters
        self.m1 = self.rho_s*self.b1s*self.h1s + self.rho_p*self.b1p*self.h1p                               # unit mass of the cantilever
        n = self.Ys * self.b1s / ( self.Yp * self.b1p )                                                     # stiffness ratio of substrate to piezo modulus
        h1pa = ( self.h1p**2 + 2*n*self.h1p*self.h1s + n * self.h1s**2 ) / 2 / ( self.h1p + n*self.h1s )    # piezo layer surface distance to neutral axis
        h1sa = ( self.h1p**2 + 2*self.h1p*self.h1s + n * self.h1s**2 ) / 2 / ( self.h1p + n*self.h1s )      # substrate layer surface distance to neutral axis
        h1pc = n * self.h1s * ( self.h1p+self.h1s ) / 2 / ( self.h1p + n*self.h1s )                  # 
        h1a = -h1sa                                                                                         # distance from neutral axis to bottom of substrate
        h1b = h1pa - self.h1p                                                                               # distance from neutral axis to top of substrate
        h1c = h1pa                                                                                          # distance from neutral axis to top of piezo layer
        self.EI1 = self.b1s/3*self.Ys*(h1b**3-h1a**3)+self.b1p/3*self.Yp*(h1c**3-h1b**3)                         # bending stiffness of the cantilever
        self.get_A1()
        
    # 5 # find natural frequency by det(A) = 0
        # where A x C = 0. C is the vector of the 4 unknowns (C1, C2, C3, C4)
        # C is coefficient of eigenfunction ( W(x) )of deflection function ( w(x, t) )
        # A is the matrix of the coefficients of the unknowns
        # Refer to: Analysis and Verification of V-shaped Piezoelectric Energy Harvesters with Angle and Tip Mass
    def det_A(self, alpha: np.float64) -> float:
        # natural frequency omega_i = alpha_i * sqrt( EI1 / m1 )
        # Here we only consider the first natural frequencies
        # Refer to: Analysis and Verification of V-shaped Piezoelectric Energy Harvesters with Angle and Tip Mass
        # formula (3.1.47) or 
        # Piezoelectric Energy Harvesting by Alper Erturk Appendix C formula (C.18)
        cos = np.cos(alpha*self.L1)
        sin = np.sin(alpha*self.L1)
        cosh = np.cosh(alpha*self.L1)
        sinh = np.sinh(alpha*self.L1)
        det_value = (1 + cos*cosh
                    +(self.M1*alpha/self.m1)*(cos*sinh-sin*cosh)
                    -(self.J1*alpha**3/self.m1)*(cosh*sin+sinh*cos)
                    -(self.M1*self.J1*alpha**4/self.m1**2)*(cos*cosh-1))
        return det_value

    # 6 # find the mode shape coefficient A1
    def get_A1(self):
        self.alpha = root(self.det_A, 0).x[0]      # Natural frequency of the cantilever omega = alpha^2*sqrt(EI1/m1)
        cos_L = np.cos(self.alpha*self.L1)
        sin_L = np.sin(self.alpha*self.L1)
        cosh_L = np.cosh(self.alpha*self.L1)
        sinh_L = np.sinh(self.alpha*self.L1)
        cos_2L = np.cos(2*self.alpha*self.L1) 
        sin_2L = np.sin(2*self.alpha*self.L1)
        cosh_2L = np.cosh(2*self.alpha*self.L1)
        sinh_2L = np.sinh(2*self.alpha*self.L1)

        self.sigma = (sin_L - sinh_L + (self.alpha*self.M1/self.m1)*(cos_L - cosh_L))/\
                (cos_L + cosh_L - (self.alpha*self.M1/self.m1)*(sin_L - sinh_L))       # Piezoelectric Energy Harvesting by Alper Erturk Appendix C formula (C.20)
        self.A1 = math.sqrt((4*self.alpha)/((-self.sigma**2*sin_2L
                            +self.sigma**2*sinh_2L
                            -4*self.sigma**2*cos_L*sinh_L
                            +4*(self.sigma**2+1)*sin_L*cosh_L
                            -2*self.sigma*cos_2L
                            +2*self.sigma*cosh_2L
                            +8*self.sigma*sin_L*sinh_L
                            +4*self.alpha*self.L1
                            +sin_2L
                            +sinh_2L
                            +4*cos_L*sinh_L
                            )*self.m1))
        return self.A1

    # 7 # find the output voltage by solving PDE

    def phi(self, x):
        return self.A1*(np.cos(self.alpha*x)
                   -np.cosh(self.alpha*x)
                   +self.sigma*np.sin(self.alpha*x)
                   -self.sigma*np.sinh(self.alpha*x))

    def d_phi(self, x):
        return self.A1*self.alpha*(-np.sin(self.alpha*x)
                                   -np.sinh(self.alpha*x)
                                   +self.sigma*np.cos(self.alpha*x) 
                                   -self.sigma*np.cosh(self.alpha*x))

    def cantilever_actuator_eq_solver(self, t, in_, para ):
        [ Cp, R, vphi, zeta, omega ] = para
        eta1 = in_[ 0 ]
        eta1dot = in_[ 1 ]
        v2 = in_[ 2 ]
        eta1ddot = -2 * zeta * omega * eta1dot - omega**2 * eta1 - vphi*v2 - self.force(t)
        v2dot = -1 / ( Cp*R ) * v2 + vphi/Cp*eta1dot

        return [ eta1dot, eta1ddot, v2dot ]

    def voltage(self):
        self.get_A1()
        vphi = self.vtheta*(self.d_phi(self.Lp2) 
                            - self.d_phi(self.Lp1))
        para = [ self.Cp, self.R, vphi, self.zeta, self.alpha**2*math.sqrt(self.EI1/self.m1) ]
        return solve_ivp( self.cantilever_actuator_eq_solver, 
                         [ 0, self.time_end ], 
                         [ 0, 0, 0 ], 
                         args=[ para ], 
                         t_eval=np.linspace( 0, self.time_end, self.time_step ) )
    
    def time_span(self, time_span: float, step: float=1000):
        self.time_end = time_span
        self.time_step = step
    
    def tip_mass(self, mass: float):
        self.M1 = mass
        
    def set_force(self, force_func):
        self.force = force_func
        
    def total_damping(self, damping: float):
        self.zeta = damping
        
    def load_resistance(self, resistance: float):
        self.R = resistance
    
    def piezo_coupling(self, coupling: float):
        self.vtheta = coupling
        
model = piezo_film()
model.tip_mass(11 * 1e-6)
#model.set_force(lambda t: 0.08*24)     # Modify the force function to have different voltage output
#model.time_span(0.2, 10000)

r = model.voltage()
print(timeit.timeit( model.voltage, number=1000 ))
# time the voltage function


#print( r.y[ 0 ] )
#print( r.y[ 2 ] )
figure, ax = plt.subplots( 2, 1 )
ax[ 0 ].plot( r.t, r.y[ 0 ] * model.phi(model.L1) * 1e3 )
ax[ 0 ].set_ylabel( 'Displacement' )
ax[ 0 ].set_xlabel( 'time' )
ax[ 1 ].plot( r.t, r.y[ 2 ] )
ax[ 1 ].set_ylabel( 'Voltage' )
ax[ 1 ].set_xlabel( 'time' )
plt.savefig( 'Cantilever_mass_fcn_python.png' )