# This code calculated the output voltage of a piezoelectric cantilever
# The model is cantilever with tip mass and a piezoelectric layer
import numpy as np
from scipy.optimize import root
import math
from scipy.integrate import solve_ivp
import timeit
from typing import Callable, Optional

class piezo_film():
    def __init__(self):
        self.debug=False
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
        self.b1s = 13 * 1e-3                ## substrate width
        self.b1p = self.b1s                 # piezo width
        self.b1pp = self.b1p                # electrode width

        # 2 # piezoelectric parameters
        self.d31 = -190 * 1e-12             # piezoelectric constant(m/V)
        self.vtheta = 0.85*1e-7             ## piezoelectric coupling coefficient, vtheta = Yp*d31*b1pp*h1pc
        self.epsilon = 15.93 * 1e-9         # absolute permittivity 3500*8.854*1e-12 # F/m
        self.Cp = 0.5 * 1e-9                ## Capaticance of the piezo layer

        # 4 # external circuit parameters
        self.R = 1 * 1e6                    ## external circuit load resistance
        self.zeta = 0.043                   ## damping ratio (Including internal damping and air damping)
        
        self.force: Callable[[float], float] = lambda t: 1
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
        '''
        ### This function solve the (C.33) in Appendix C of Piezoelectric Energy Harvesting by Alper Erturk.

        We first find the natural frequency by solving det(A) = 0 from (C.17)
        Then we got the phi(x) in (C.19) with only Ar left as unknown.
        But we know that (C.33). S solving this equation with only Ar left as unknown. We can get the Ar.

        *Note: Ar is A1 in the code.*

        ### return: 
            A1: int, the mode shape coefficient
        '''
        self.alpha = root(self.det_A, 0).x[0]       # Natural frequency of the cantilever omega = alpha^2*sqrt(EI1/m1)
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
        if self.debug == True:
            print(f"Natural frequency omega: {self.alpha**2*math.sqrt(self.EI1/self.m1)}")
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
        eta1ddot = -2 * zeta * omega * eta1dot - omega**2 * eta1 - vphi*v2 + self.force(t)
        v2dot = -1 / ( Cp*R ) * v2 - vphi/Cp*eta1dot

        return [ eta1dot, eta1ddot, v2dot ]
    
    
    def voltage(self, method: str='RK45', force: Callable[[float], float] = None, max_step: Optional[float] = 0.001):
        if force is not None:
            self.force = force

        vphi = self.vtheta*(self.d_phi(self.Lp2) 
                            - self.d_phi(self.Lp1))
        para = [ self.Cp, self.R, vphi, self.zeta, self.alpha**2*math.sqrt(self.EI1/self.m1) ]
        return solve_ivp( self.cantilever_actuator_eq_solver,
                         [ 0, self.time_end ], 
                         [ 0, 0, 0 ], 
                         args=[ para ], 
                         t_eval=np.linspace( 0, self.time_end, self.time_step ),
                         max_step=max_step,
                         method=method ) # type: ignore
    
    def voltage_to_force_A(self, voltages: np.ndarray, dt: float):
        '''
        Assume w(x, t)=ΣΦ(x)η(t)
        This method is based on the relationship between the v(t) and η(t)
        
        return:
            force: estimated force
            A item: 2nd derivative of the voltage times a constant
            B item: 1st derivative of the voltage times a constant
            C item: voltage times a constant
            D item: 1st integral of the voltage times a constant
        '''

        # Differentiation of the voltage
        d_voltages = np.gradient(voltages, dt)
        dd_voltages = np.gradient(d_voltages, dt)
        # Integration of the voltage
        i_voltages = np.cumsum(voltages)*dt

        omega = self.alpha**2*math.sqrt(self.EI1/self.m1)
        vphi = self.vtheta*(self.d_phi(self.Lp2) - self.d_phi(self.Lp1))

        A = (self.Cp/(-vphi))

        B = (1/(-vphi*self.R)
             + (2*self.zeta*omega*self.Cp)/(-vphi))
        
        C = ((2*self.zeta*omega)/(-vphi*self.R)
             + (self.Cp*omega**2)/(-vphi)
             + vphi)

        D = (omega**2)/(-vphi*self.R)

        return (A*dd_voltages + B*d_voltages + C*voltages + D*i_voltages), A*dd_voltages, B*d_voltages, C*voltages, D*i_voltages

    def voltage_to_eta(self, voltages: np.ndarray, dt: float):
    
        # Differentiation of the voltage
        d_voltages = np.gradient(voltages, dt)
        # Integration of the voltage
        i_voltages = np.cumsum(voltages)*dt
        
        eta = (self.Cp*voltages + i_voltages/self.R)/(-vphi)
        d_eta = (self.Cp*d_voltages + voltages/self.R)/(-vphi)
        
        # Get reference dd_eta from actual function
        vphi = self.vtheta*(self.d_phi(self.Lp2) 
                            - self.d_phi(self.Lp1))
        para = [ self.Cp, self.R, vphi, self.zeta, self.alpha**2*math.sqrt(self.EI1/self.m1) ]
        dd_eta_ref = np.array([self.cantilever_actuator_eq_solver(dt*i, [eta[i], d_eta[i], voltages[i]], para)[1] for i in range(len(eta))])

        # Calculate dd_eta from voltage

        return eta, d_eta, dd_eta_ref


    def time_span(self, time_span: float, step: float=1000):
        self.time_end = time_span
        self.time_step = step
        self.dt = time_span/step
    
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
        
    def set_thickness(self, substrate: float = None, piezofilm: float = None):
        '''
        parameters:
            substrate: thickness of the substrate (Unit: m)
            piezofilm: thickness of the piezofilm (Unit: m)
        '''
        if substrate is not None:
            self.h1s = substrate
        if piezofilm is not None:
            self.h1p = piezofilm
        self.cal_properties()
    
    def set_youngs(self, substrate: float = None, piezofilm: float = None):
        '''
        parameters:
            substrate: Young's modulus of the substrate (Unit: Pa)
            piezofilm: Young's modulus of the piezofilm (Unit: Pa)
        '''
        if substrate is not None:
            self.Ys = substrate
        if piezofilm is not None:
            self.Yp = piezofilm
        self.cal_properties()