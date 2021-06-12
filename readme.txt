This is an experimental repo for implementing Fokker-Planck Solvers.

The main learning procedure is based on Taylor series approximation in time variable. Complexity of 
time derivatives grow too fast since they are computed through the differential operator appearing 
in the PDE p_t = Lp. But nonetheless for low dimensions it seems to be capable of learning qualitatively 
right dynamics for first few hundred time steps assuming time steps are < 0.005.