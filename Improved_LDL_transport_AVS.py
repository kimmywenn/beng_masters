__author__ = "Mostafa Mahmoudi <mm4238@nau.edu>"
__date__ = "29-06-2022"
__copyright__ = "Copyright (C) 2022 " + __author__
__license__ = "Free - Refer to README.MD to learn how to cite this package."


from dolfin import *

import numpy as np

import scipy

from ufl import operators as ufl_op

import time


# TO RUN: mpirun -np #of_procs python filename.py

set_log_level(30)


relative_tolerance = 1e-14 #1e-6 This was original value

stabilized = True #False

aneurysm_bc_type = 'neumann'

parameters['form_compiler']['representation'] = 'uflacs'

parameters['form_compiler']['optimize'] = True

parameters['form_compiler']['cpp_optimize'] = True

root_dir = '/scratch/mm4238/transport/3A_2016_LAD/'

if MPI.rank(mpi_comm_world()) == 0:
	print 'Root_dir:', root_dir

results_dir = root_dir + 'LDL_transport/Results/'

mesh_filename = root_dir + 'mesh_transport/3A_2016_LAD.xml'

bc_file = root_dir + 'mesh_transport/BCnodeFacets.xml' #ORder: inlet, wall, outlet1

if MPI.rank(mpi_comm_world()) == 0:
	print 'Loading mesh:', mesh_filename

# mesh = Mesh()
# hdf = HDF5File(mpi_comm_world(), root_dir + "h5_vel_wss/" + "1A_2012_LAD.h5", "r")
# hdf.read(mesh, "/mesh", False)
# facet_domains = FacetFunction("size_t", mesh)
# hdf.read(facet_domains, "/boundaries")
# hdf.close()


mesh = Mesh(mesh_filename)
facet_domains = MeshFunction('size_t', mesh, bc_file)

# Define a MeshFunction over 3 subdomains
subdomains = MeshFunction('size_t', mesh, 3)


V0 = FunctionSpace(mesh, 'DG', 0)
D_multiple = Function(V0) #2 different Diffusion coefs

# Defining the planes to create the region of interest

P1 = [15.19896, 23.9704, -4.91926]
n1 = [0.885152, -0.4096, 0.220757]

P2 = [13.62255, 24.20334, -5.312]
n2 = [-0.8656, 0.275995, -0.41785]

# my_eps = 0.01
class Omega_inside(SubDomain): #Region of interest
	def inside(self, x, on_boundary):
		return False  if (  n1[0] * (x[0] - P1[0] ) + n1[1] * ( x[1] - P1[1] ) + n1[2] * ( x[2] - P1[2] ) < 0  )  or (  n2[0] * (x[0] - P2[0] ) + n2[1] * ( x[1] - P2[1] ) + n2[2] * ( x[2] - P2[2] ) < 0 )  else True

class Omega_out1(SubDomain): #Inlet Region
	def inside(self, x, on_boundary):  #With eqn of a plane
		return True if (  n1[0] * (x[0] - P1[0] ) + n1[1] * ( x[1] - P1[1] ) + n1[2] * ( x[2] - P1[2] ) > 0 )    else False

class Omega_out2(SubDomain):  #Outlet Region
	def inside(self, x, on_boundary):
		return True if (  n2[0] * (x[0] - P2[0] ) + n2[1] * ( x[1] - P2[1] ) + n2[2] * ( x[2] - P2[2] ) > 0  ) else False


n_normal = FacetNormal(mesh)


# input_vel_name = root_dir + 'h5_vel_wss/velocity_fineMesh.h5'
# hdf_velocity = HDF5File(mesh.mpi_comm(), input_vel_name, "r")

# input_wss_name = root_dir + 'h5_vel_wss/wss_fineMesh.h5'
# hdf_wss = HDF5File(mesh.mpi_comm(), input_wss_name, "r")

velocity_filename = root_dir + 'vel_xml/velocity_fineMesh_'

wss_filename = root_dir + 'wss_xml/wss_fineMesh_'

velocity_start =  191 #2020
velocity_stop = 290   #3000
velocity_interval = 1 #20

wss_start =  191 #2020
wss_stop = 290   #3000
wss_interval = 1 #20

#velocity_in = HDF5File(mpi_comm_world(), velocity_filename, 'r')
#velocity_prefix = '/velocity/vector_'

#Initial Condition from file
IC_flag = False #if True uses an I.C file
IC_file = root_dir + 'results/concentration2nd_D5_16000.h5'
IC_start_index = 16000

if MPI.rank(mpi_comm_world()) == 0:

	print 'Num global vertices', mesh.num_vertices()




max_save_interval = 1 *1  #For writing max calues

solution_save_interval = 500 *1  #writing solution

solution_save_last= 45000 *1 #write more files for last cycles
solution_save_last_freq = 100 *1



h =  CellDiameter(mesh)

basis_order = 1



# Parameters

D = 2.8e-5 # [mm2/s]
D_high = 1e-4

t_start = 0.0

t_stop = 10.0 #10cycles (T=0.88)

n_tsteps = 5000 * 10  #240000

tsteps_per_file = 50 * 1  #number of time steps btwn files (delta_t_file = 0.0176). = delta_t_file/(dt=T/1000)

dt = (t_stop - t_start) / n_tsteps #dt = 0.00088


# Mark subdomains with numbers 0 and 1
subdomains.set_all(2) #initialize the whole domain
subdomain0 = Omega_out1()
subdomain0.mark(subdomains, 0)
subdomain1 = Omega_out2()
subdomain1.mark(subdomains, 1)
subdomain2 = Omega_inside()
subdomain2.mark(subdomains, 2)

# Loop over all cell numbers, find corresponding
# subdomain number and fill cell value in k
sss = len(subdomains.array())
D_values = [D_high, D_high, D ]  # values of k in the two subdomains
D_numpy = np.zeros((sss))

for cell_no in xrange(sss):
	subdomain_no = subdomains.array()[cell_no]
	D_numpy[cell_no] = D_values[subdomain_no]


#D_multiple = Function(V0) #Defined earlier
D_multiple.vector().set_local(D_numpy)

out_D_multiple = File(results_dir + 'Domain.pvd') 
out_D_multiple << D_multiple


#out_D = HDF5File(mpi_comm_world(),results_dir + 'D_coef.h5','w')
#out_bcs << facet_domains
#out_D.write(D_multiple,'D')
#out_D.close()

# Create FunctionSpaces

Q = FunctionSpace(mesh, 'CG', basis_order)

V = VectorFunctionSpace(mesh, 'CG', 1)

# Create mappings

v2d_V = dof_to_vertex_map(V)




#********************************************************
#*************** Shear Stress Index (SSI) ***************
#********************************************************

wss_initial = Function(V)
wss_final = Function(V)
TAWSS = Function(V)
wss_t = Function(V)
SSI = Function(Q)
OSI = Function(Q)
#************************ TAWSS *************************

# T_integral = 1 #[s]
# TAWSS = Constant(0.0)
# mag_wss_t = Constant(0.0)
# wss_t.assign( Constant((0.0, 0.0, 0.0)) )

# for wss_file_counter in xrange(wss_start,200):

# 	if MPI.rank(mpi_comm_world()) == 0:
# 		print 'processing..' + 'wss_' +str(wss_file_counter)
# 	#wss_initial = Function(VectorFunctionSpace(mesh, 'CG', 1),wss_filename +str(wss_file_counter)+'.xml')
# 	#wss_final = Function(VectorFunctionSpace(mesh, 'CG', 1),wss_filename +str(wss_file_counter+1)+'.xml')
# 	hdf_wss.read(wss_initial, '/wss_' + str(wss_file_counter) + '/vector_0')
# 	hdf_wss.read(wss_final, '/wss_' + str(wss_file_counter+1) + '/vector_0')


# 	mag_wss_initial = sqrt(inner(wss_initial , wss_initial))/1000
# 	mag_wss_final = sqrt(inner(wss_final , wss_final))/1000

# 	TAWSS += 0.5*(mag_wss_final + mag_wss_initial)*0.01

# 	#wss_t += 0.5*(wss_final.vector().get_local() + wss_initial.vector().get_local())*0.01/1000

# 	wss_t.vector().set_local(  wss_t.vector().get_local()
# 						  + 0.5*(wss_final.vector().get_local() + wss_initial.vector().get_local())*0.01/1000)
# 	wss_t.vector().apply('')


#************************* OSI **************************

# mag_wss_t = sqrt(inner(wss_t , wss_t))

# OSI = 0.5*(1 - mag_wss_t/TAWSS)

# SSI = project(OSI, Q)

# out_OSI = File(results_dir + 'OSI.pvd') 
# out_OSI << SSI


#********************************************************
#*************** LDL Transport Parameters ***************
#********************************************************

c_0 = 3.12e-6 #g/cm3 : Uniform Constant LDL Concentration at Inlet
c_0_space = interpolate(Constant(c_0), Q)

rhs_space = Function(Q)
rhs_space = interpolate(Constant(0.0), Q)

#********* Endothelial Cell Shape Index (ECSI) **********
a1 = 0.380
a2 = -0.79
a3 = 0.225
a4 = -0.043
#*********** Number of Mitotic Celss (#MC) **************
b1 = 0.003739 # [cells]
b2 = 14.75
#************* Number of Leaky Cells (#LC) **************
c1 = 0.307 # [cells]
c2 = 0.805

#************ OLD: Vesicular Permeability (P-v) **************
# P_v_space = Function(Q)
# P_v = 1.92e-9 # [cm/s]
# P_v_space = interpolate(Constant(P_v), Q)

#************ NEW: Vesicular Permeability (P-v) **************
# Base permeability [cm/s]
P_v = 1.92e-9

# Geometry of semi circular valve leaflet
t_leaflet = 0.0694		# Adjust based on thickness of leaflet (cm) (in this case, stenosed leaflet)
v_radius = 1.0			# measured from lab samples (cm)

# Tunable parameters 		
r_min = 0.0                  
r_max = v_radius                     # Adjust based on your leaflet  base-edge length (cm) (also equal to radius of human aorta)
k = 1e-4  		     	# sensitivity parameter to stiffness, tuned to fit scale of P_v

# Parametrizing the fractional layer thicknesses
f_fib = 0.44  # Fraction of fibrosa layer
f_vent = 0.25  # Fraction of ventricularis layer
f_spong = 1 - f_fib - f_vent  # Fraction of spongiosa layer

# Layer depth based on fractions (z-value when layer ends)
z_ventricularis = f_vent * t_leaflet
z_spongiosa = f_spong * t_leaflet + z_ventricularis

# Stiffness values (Young's modulus) in kPa
E_base   = 150    # annular (base) region
E_free   = 42.6    # free edge
E_vent   = 26.9    # ventricularis layer
E_spong  = 15.4   # spongiosa layer
E_fib    = 37.1    # fibrosa layer

# Apply exponential decay formula
P_base  = P_v * np.exp(-k * E_base)
P_free  = P_v * np.exp(-k * E_free)
P_vent  = P_v * np.exp(-k * E_vent)
P_spong = P_v * np.exp(-k * E_spong)
P_fibrosa   = P_v * np.exp(-k * E_fib)

class P_v_3D(UserExpression):
    def eval(self, value, x):
    # Radial distance from center of semicircular leaflet
    r = np.sqrt(x[0]**2 + x[1]**2)
    r_clamped = np.clip(r, r_min, r_max)

    # Radial gradient from base to free edge
    p_r = P_free + (P_base - P_free) * (r_clamped - r_min) / (r_max - r_min)

    # Z-direction by layer
    if x[2] < z_ventricularis:
        p_z = P_vent
    elif x[2] < z_spongiosa:
        p_z = P_spong
    else:
        p_z = P_fib

    # Combined permeability
    value[0] = p_r * p_z

# Compute spatially varying P_v
P_v_expr = P_v_3D(degree=1)
P_v_space = interpolate(P_v_expr, Q)

#**************** Biological Properties *****************
R_cell = 15e-4 # [cm]: Radius of endothelial cell
w = 14e-7 # [cm]: Half width of leaky junction
a_LDL = 11e-7 # [cm]: Radius of LDL molecules
unit_area = 0.64e-2 # [cm2]
mu_p = 1e-2 # [g/cm.s]: Plasma viscosity
l_lj = 2e-4 # [cm]: length of leaky junction
delta_p_end = 2333.142e-2 # [g/cm2]: Endothelial pressure difference

#************ Three-Pore Model Parameters ***************
alpha_lj = a_LDL/w
Phi_lj = 1 - alpha_lj
F_lj = 1.0 - 1.004*alpha_lj + 0.418*pow(alpha_lj, 3) - 0.169*pow(alpha_lj, 5) # Hinderance Factor for Diffusion in a Pore

P_app_space = Function(Q, name="Permeability")


#********************************************************
#****************** Required Functions ******************
#********************************************************

v = TestFunction(Q)
c_trial = TrialFunction(Q)
c = Function(Q, name="Concentration")

c_prev = Function(Q)

#********************* Velocities *********************

velocity = Function(V)

velocity_initial = Function(V)

velocity_final = Function(V)

#*********************** WSS ***********************

wss = Function(V)

wss_initial = Function(V)

wss_final = Function(V)

#****************** Concentration Functions ******************

cdot = Function(Q)
cdot_prev = Function(Q)
dc = Function(Q)

flux = TrialFunction(Q)

phi = TestFunction(Q)

flux_sol = Function(Q)

#****************** Three-Pore Functions ******************

V_inf = Function(Q)
K_space = Function(Q)
wss_space = Function(Q)

#********************************************************
#****************** Boundary Conditions *****************
#********************************************************

#**************** BCs for OATPTransport *****************

bc1 = DirichletBC(Q, c_0, facet_domains, 1) # Inlet: 0 drichlet

bc2 = DirichletBC(Q, 0.0, facet_domains, 2) # Wall: flux

bc3 = DirichletBC(Q, 0.0, facet_domains, 3) # Outlet1: 0 dirichlet

bc4 = DirichletBC(Q, 0.0, facet_domains, 4) # Outlet2: 0 dirichlet

bc5 = DirichletBC(Q, 0.0, facet_domains, 5) # Outlet2: 0 dirichlet

#****************** BCs for dc Equation *****************
# For non-zero Dirichlet BC on transport equation, the BC for dc equation should be ZERO
# For Neumann BC on transport equation, the BC for dc equation should NOT be specified

bc_dc1 = DirichletBC(Q, 0.0, facet_domains, 1) # Inlet: 0 drichlet

bc_dc2 = DirichletBC(Q, 0.0, facet_domains, 2) # Wall: flux

bc_dc3 = DirichletBC(Q, 0.0, facet_domains, 3) # Outlet1: 0 dirichlet

bc_dc4 = DirichletBC(Q, 0.0, facet_domains, 4) # Outlet2: 0 dirichlet

bc_dc5 = DirichletBC(Q, 0.0, facet_domains, 5) # Outlet2: 0 dirichlet


if aneurysm_bc_type == 'neumann':
	bcs = [bc1] 
	bcs_dc = [bc_dc1] 

elif aneurysm_bc_type == 'dirichlet':
	bcs = [bc1, bc2]
	bcs_dc = [bc_dc1, bc_dc2]

else:
	print 'Need to pick Neumann or Dirichlet BC for aneurysm'



if aneurysm_bc_type == 'neumann':
	ds = Measure("ds")(subdomain_data=facet_domains)


#********************************************************
#******************* Generalized alpha ******************
#********************************************************

m = v * cdot * dx

M = assemble(v * c_trial * dx)
rho_inf = 0.0 #.2
alpha_m = .5 * (3. - rho_inf) / (1. + rho_inf)
alpha_f = 1. / (1. + rho_inf)
gamma = .5 + alpha_m - alpha_f

tstep = 0

#******************* Initial Condition ******************
if (IC_flag == True):
 IC_in =  HDF5File(mpi_comm_world(), IC_file, 'r')
 IC_in.read(c_prev, '/concentration/vector_0')
 IC_in.close()
 tstep = IC_start_index
 t_start = IC_start_index * dt
 if MPI.rank(mpi_comm_world()) == 0:
	print 'Using I.C.'
else:
 c_prev.assign(interpolate(Constant(0.0), Q))

cdot_prev.assign(interpolate(Constant(0.0), Q))

t = t_start

file_initial = velocity_start
file_final = file_initial + velocity_interval

wss_file_initial = wss_start
wss_file_final = wss_file_initial + wss_interval


if file_final > velocity_stop:
	file_final = velocity_start


#out = HDF5File(mpi_comm_world(),results_dir + 'concentration.h5','w')
#out_velocity = HDF5File(mpi_comm_world(),results_dir + 'velocity.h5','w')

#out_max_vals = open(results_dir + 'max_vals.dat', 'w')

#prm = parameters["krylov_solver"]
#prm["nonzero_initial_guess"] = True

#R_b_space = interpolate(Constant(R_b), Q)

#******************* Time Stepping ******************

start_time = time.clock()
#problem = LinearVariationalProblem(a, L, c_sol, bcs)

#We set solver inside the for loop here
#solver = LinearVariationalSolver(problem)
#solver.parameters["linear_solver"] ="bicgstab"
#solver.parameters["preconditioner"] ="bjacobi"
#solver.parameters['krylov_solver']['nonzero_initial_guess'] = True


prm = parameters['krylov_solver'] # short form
prm['absolute_tolerance'] = 1e-17
prm['relative_tolerance'] = 1e-16
prm['monitor_convergence'] = True
#Uncomment below of ill-conditioning issues arise
#prm['absolute_tolerance'] = 1e-17
#prm['relative_tolerance'] = 1e-10
file_conc_output = File(results_dir + 'LDL_transport.pvd')
file_P_app_output = File(results_dir + 'P_app.pvd')
#file_K_output = File(results_dir + 'K_wall.pvd')
#file_wss_output = File(results_dir + 'mag_wss.pvd')
while tstep < n_tsteps:

	velocity_initial = Function(VectorFunctionSpace(mesh, 'CG', 1),velocity_filename +str(file_initial)+'.xml')
	velocity_final = Function(VectorFunctionSpace(mesh, 'CG', 1),velocity_filename +str(file_final)+'.xml')

	# hdf_velocity.read(velocity_initial, '/Velocity_' + str(file_initial) + '/vector_0')
	# hdf_velocity.read(velocity_final, '/Velocity_' + str(file_final) + '/vector_0')


	wss_initial = Function(VectorFunctionSpace(mesh, 'CG', 1),wss_filename +str(wss_file_initial)+'.xml')
	wss_final = Function(VectorFunctionSpace(mesh, 'CG', 1),wss_filename +str(wss_file_final)+'.xml')

	# hdf_wss.read(wss_initial, '/wss_' + str(wss_file_initial) + '/vector_0')
	# hdf_wss.read(wss_final, '/wss_' + str(wss_file_final) + '/vector_0')


	for tstep_inter in xrange(tsteps_per_file):
		tstep += 1
		if tstep > n_tsteps:
			break
		t += dt



		alpha = (tstep_inter * 1.0) / (tsteps_per_file * 1.0)

		velocity.vector().set_local(  (1. - alpha) * velocity_initial.vector().get_local()
							  + alpha * velocity_final.vector().get_local())
		velocity.vector().apply('')

		wss.vector().set_local(  (1. - alpha) * wss_initial.vector().get_local()
							  + alpha * wss_final.vector().get_local())
		wss.vector().apply('')

	
		mag_wss = sqrt(inner(wss , wss))

		# Initialize c and cdot
		c.vector().set_local((1. - alpha_f) * c_prev.vector().get_local()+ alpha_f * c_prev.vector().get_local())
		c.vector().apply('')
		cdot.vector().set_local((1. - alpha_m) * cdot_prev.vector().get_local() + alpha_m * (gamma - 1.) / gamma * cdot_prev.vector().get_local())
		cdot.vector().apply('')
		
		#******************* Three-Pore Model ******************

		ECSI = a1*ufl_op.exp(a2*mag_wss) + a3*ufl_op.exp(a4*mag_wss) # Endothelial Cells Shape Index

		MC = b1*ufl_op.exp(b2*ECSI) # Number of mitotic cells
		LC = c1 + c2*MC # Number of leaky cells

		phi = LC*3.1415*pow(R_cell, 2)/unit_area # Fraction of leaky junctions
		Ap_S_ratio = 4.0*w*phi/R_cell # Fraction of the surface area occupied by the leaky junctions
		D_lj = D*F_lj # Diffusion Coefficient in Leaky Junctions
		P_slj = D_lj/l_lj # Permeability of a single leaky junction
		P_lj = Ap_S_ratio*Phi_lj*P_slj # Permeability of leaky junctions

		L_pslj = pow(w, 2)/(3*mu_p*l_lj) # Hydraulic conductivity of a single leaky junction
		L_plj = Ap_S_ratio*L_pslj # Hydraulic conductivity of leaky junctions
		J_vlj = L_plj*delta_p_end # Volume flux through leaky junctions

		sigma_flj = 1.0 - (2./3.)*pow(alpha_lj, 2.)*(1. - alpha_lj)*F_lj - (1. - alpha_lj)*(2./3. + (2./3.)*alpha_lj - (7./12.)*pow(alpha_lj, 2.)) # Solvent drag coefficient
		
		
		Pe_lj = J_vlj*(1 - sigma_flj)/P_lj # Modified Peclet number for leaky junctions


		Z_lj = Pe_lj/(exp(Pe_lj) - 1.) # Fractional reduction factor in solute concentration gradient at the pore entrance

		P_applj = P_lj*Z_lj + J_vlj*(1. - sigma_flj) # Apparent permeability of leaky junctions

		P_app = P_v_space + P_applj # Apparent permeability of endothelial cell layer: leaky junctions, vesicular junctions, normal junctions = 0.0


		n =  -v * dot(velocity , grad(c_trial) ) * dx \
			- D_multiple * dot(grad(v), grad(c_trial)) * dx \
			+ P_app * c_trial * v * ds(2)

		if aneurysm_bc_type == 'neumann':
			rhs = rhs_space


		if stabilized: #!! c_prev in res_n should go to rhs below
			res_n = dot(velocity , grad(c_trial) )  - div(D_multiple * grad(c_trial))  #Maybe has to be advective form like the weak form 
			tau_m = (4. / dt**2 \
				 + dot(velocity, velocity) / h**2 \
				 + 9. * basis_order**4 * D_multiple**2 / h**4)**(-.5)
			n -= tau_m * res_n * dot(grad(v),velocity) * dx 

			m1 =  v * c_trial * dx  +  tau_m * c_trial* dot(grad(v), velocity) * dx # the cdot term in stablization should go here 
			M = assemble(m1)

		N = assemble(n)

		iteration = 0
		err = 1. + relative_tolerance
		while err > relative_tolerance:
			iteration += 1

			K = alpha_m / gamma / dt / alpha_f * M - N

			G = -1. * M * cdot.vector() + N * c.vector() #+ N_rhs    #assemble(N) # Actually -G from Jansen
			
			for bc in bcs_dc:
				bc.apply(K, G)
			solve(K, dc.vector(), G, 'gmres', 'default') #prec:'bjacobi'
			c.vector().set_local(c.vector().get_local() + dc.vector().get_local())
			c.vector().apply('')
			cdot.vector().set_local(
				(1. - alpha_m / gamma) * cdot_prev.vector().get_local()
				+ alpha_m / gamma / dt / alpha_f
				  * (c.vector().get_local() - c_prev.vector().get_local()))
			cdot.vector().apply('')
			err = norm(dc.vector(), 'linf') / max(norm(c.vector(), 'linf'), DOLFIN_EPS)

			if MPI.rank(mpi_comm_world()) == 0:
				print 'Iteration:', iteration, 'Error:', err
			# Enforce Dirichlet BCs
			for bc in bcs:
				bc.apply(c.vector())



		c.vector().set_local(
				(1. - 1. / alpha_f) * c_prev.vector().get_local()
				+ 1. / alpha_f * c.vector().get_local())
		c.vector().apply('')

		cdot.vector().set_local(
				(1. - 1. / alpha_m) * cdot_prev.vector().get_local()
				+ 1. / alpha_m * cdot.vector().get_local())
		cdot.vector().apply('')
		c_prev.vector().set_local(c.vector().get_local())
		c_prev.vector().apply('')
		cdot_prev.vector().set_local(cdot.vector().get_local())
		cdot_prev.vector().apply('')


		# Plot and/or save solution

		if tstep % max_save_interval == 0:
			if aneurysm_bc_type == 'dirichlet':

				flux_form = -D * inner(grad(c), n)

				LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)

				RHS_flux = assemble(flux_form * phi * ds)

				LHS_flux.ident_zeros()

				solve(LHS_flux, flux_sol.vector(), RHS_flux)

				max_val = MPI.max(mpi_comm_world(),
								  np.amax(flux_sol.vector().get_local()))

				min_val = MPI.min(mpi_comm_world(),
								  np.amin(flux_sol.vector().get_local()))

				if MPI.rank(mpi_comm_world()) == 0:

					print 'time =', t, 'of', t_stop, '...', \
						  'Max=', max_val, \
						  'Min=', min_val, \
						  'Time elapsed:', (time.clock() - start_time) / 3600., \
						  'Time remaining:', 1. / 3600. * (n_tsteps-tstep) \
										 * (time.clock()-start_time) / tstep


			else:

				max_val =   MPI.max(mpi_comm_world(),np.amax(c.vector().get_local()))
				min_val = MPI.min(mpi_comm_world(), np.amin(c.vector().get_local()))


				if MPI.rank(mpi_comm_world()) == 0:
					print 'time =', t, 'of', t_stop, '...', \
						  'Max=', max_val, \
						  'Min=', min_val, \
						  'Time elapsed:', (time.clock() - start_time) / 3600., \
						  'Time remaining:', 1. / 3600. * (n_tsteps-tstep) \
										 * (time.clock()-start_time) / tstep


		if ( tstep % solution_save_interval == 0  ) :
			file_conc_output << c 
			P_app_space = project(P_app, Q, solver_type='cg', preconditioner_type='hypre_amg')
			#K_space = project(K_wall, Q, solver_type='cg', preconditioner_type='hypre_amg')
			#wss_space = project(mag_wss, Q, solver_type='cg', preconditioner_type='hypre_amg')
			file_P_app_output << P_app_space
			#file_K_output << K_space
			#file_wss_output << wss_space

			#out = HDF5File(mpi_comm_world(),results_dir + 'concentration2nd_D5_' + str(tstep) + '.h5' ,'w')
			#out.write(c,'concentration',tstep)
			#out.close()

			if aneurysm_bc_type == 'dirichlet':
				flux_form = -D * inner(grad(c), n_normal )
				LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)
				RHS_flux = assemble(flux_form * phi * ds)
				LHS_flux.ident_zeros()
				solve(LHS_flux, flux_sol.vector(), RHS_flux)
				out_flux.write(flux_sol,'flux_sol', tstep)

		if tstep % solution_save_last_freq  == 0 and \
				(tstep >= solution_save_last) :
			file_conc_output << c
			P_app_space = project(P_app, Q, solver_type='cg', preconditioner_type='hypre_amg')
			file_P_app_output << P_app_space

			#out_velocity = HDF5File(mpi_comm_world(),results_dir + 'velocity_'+ str(tstep) + '.h5','w')
			#out_velocity.write(velocity,'velocity',tstep)
			#out_velocity.close()
			#out = HDF5File(mpi_comm_world(),results_dir + 'concentration2nd_D5_' + str(tstep) + '.h5' ,'w')
			#out.write(c,'concentration',tstep)
			#out.close()

			if aneurysm_bc_type == 'dirichlet':
				flux_form = -D * inner(grad(c), n_normal)
				LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)
				RHS_flux = assemble(flux_form * phi * ds)
				LHS_flux.ident_zeros()
				solve(LHS_flux, flux_sol.vector(), RHS_flux)
				out_flux.write(flux_sol,'flux_sol', tstep)


	file_initial = file_final
	file_final = file_initial + velocity_interval

	wss_file_initial = wss_file_final
	wss_file_final = wss_file_initial + wss_interval

	if file_final > velocity_stop:
		file_final = velocity_start

	if wss_file_final > wss_stop:
		wss_file_final = wss_start

