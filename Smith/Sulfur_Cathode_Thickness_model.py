"""
    This file runs and executes a youyr model, calculating the cell/device/system properties of interest as a function of time. 

    The code structure at this level is meant to be very simple, and delegates most functions to lower-level modules, which can be updated with new capabilties, over time.  The code:

        1 - Reads inputs and initializes the model
        
        2 - Calls the residual function and integrates over the user-defined    
            time span.
        3 - The simulation then returns the solution vector at the end of the 
            integration, which can be processed as needed to plot or analyze any quantities of interest.
"""

# Import necessary modules:
from scipy.integrate import solve_ivp #integration function for ODE system.

# Either directly in this file, or in a separate file that you import, define: 
#   - A residual function called 'residual'
#   - An array 'time_span' which has [0, t_final] where t_final is the total 
#       length of time you want the model to simulate.
#   - An intial solution vector SV_0
solution = solve_ivp(residual, time_span, SV_0,rtol=1e-4, atol=1e-6)
