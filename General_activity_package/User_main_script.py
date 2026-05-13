import MDAnalysis as mda
from Gij_pairs_at_TL import run_parallel,subvolumes_and_components,SBA,expand_symmetric_upper_triangle#,extract_independent_densities
from Activity import water_activity
import numpy as np
from Activity import Latex_Analytical_expressions_for_potential_derivatives as Latex_expressions
#########################  USER INPUTS  ########################################

# 1) Here we put our topology and trajectory files.

topol="Test_NaCl.gro" #Put your topology
traj="Test_NaCl.xtc" #Put your trajectory


u=mda.Universe(topol,traj)

# 2) Here we define the fraction of the total volume in a linspace. These values are the default.
Largest_subvolume = 0.4     #In this case that would be 40% of the total volume.
Total_number_of_subvolumes = 50     #This is the amount of subvolumes that we choose from 0.1% of the total volume (default) until our choice for the largest subvolume.

# 3) Here we put the residue name of your solvent. This makes sure that the ordering of residues is correct, since we need to have the solvent first.
solvent=['SOL']

# 4) For each subvolume the total number of particles for each constituent is counted. By "particle" it is understood that the center of mass (COM) for each molecule is computed and that is counted as a particle.
# If the box is really big and the number of solvent (water) is too large, that might heavily affect to the performance in both, memory and time-wise. 
# In these cases we can choose to turn the computation of the center of mass for the solvent off. In this case, the heaviest atom of the solvent will be counted as the closest to the COM and used when counting the total number of particles for a given subvolume.
# This approach is especially favoured towards water, since the Oxygen is a good approximation as its COM. 
# If we want to compute the center of mass of the solvent for each solvent molecule and for each snapshot, Center_of_mass_of_solvent = 'on'. Otherwise, Center_of_mass_of_solvent = 'off'.
Center_of_mass_of_solvent = 'off'

# 5) We need to provide the experimental value of the bulk density of the solvent and its molar mass in order to obtain its activity:
Experimental_bulk_density = 997.05e3 #g/m³, this is the experimental water bulk density at 25⁰C and 1 atm.
Molar_mass = 18.015 #g/mol, for the case of water.

# 6) For a given subvolume of certain size and a given time frame, NSVs is the Number of SubVolumes that are sampled to increase the statistical accuracy.
NSVs=100 #In this work NSVs=100, but it does not need to be so large for bigger systems, especially if the total amount of snapshots is high.

# 7) With this parameter we choose up to which value of lambda we want to make the initial fit.
threshold_to_fit=0.3

# 8) When we compute the water activity we need to integrate Equation 12 of the main manuscript (integral shown in Eq. 13). The lower limit of integration is the number density of water for the current system. The upper
# integration limit is the experimental number density of water in the bulk. The transition between these two limits was chosen to be lineal, and the number of points in between is defined here
# as Number_of_integration_points. 
Number_of_integration_points=100

######################### Until here are the inputs that the user can modify. After that the program starts. ##############################

# 9) Here we get a list with the subvolumes, the total volume, the components and the number densities in N/Å³
subvolumes,V_total,components,Number_densities,name_of_residues=subvolumes_and_components(u,solvent,Largest_subvolume,Total_number_of_subvolumes,Center_of_mass_of_solvent)

# 10) The elements of the list All_Gijs contains dictionaries of the Gij pairs for each subvolume. Hence len(All_Gijs)=Total_number_of_subvolumes.
All_Gijs = run_parallel(subvolumes,u,V_total,components,NSVs)

# 11) Gij_TL is a list containing the Kirkwood-Buff integrals at the thermodynamic limit (TL), obtained by fitting the SBA curves for small lambdas.
#For example, for 3 components Gij_TL will contains G11,G12,G13,G22,G23,G33 in this order. In other words, the upper triangle values of a symmetric matrix.
#By default the SBA curves and fits are plotted and saved with plot="yes". We can disable that by writing any other entry, like plot="no".
Gij_TL=SBA(All_Gijs,subvolumes,V_total,threshold_to_fit,plot="yes")

# 12) Here we build a list with all the Gij components as it would be in the matrix. For example, for n=2, Flattened_matrix_of_Gijs=[G11,G12,G12,G22], since G12=G21. This list is our input to compute the activities.
Flattened_matrix_of_Gijs = expand_symmetric_upper_triangle(Gij_TL)

# 13) From here we obtain the (water) activity coefficient given an input matrix with all Gijs (Flattened_matrix_of_Gijs) and the experimental bulk number density (final density or upper limit of integration).
#We need to integrate eq. 12 from the manuscript given solvent starting and final number densities. The final number density is computed here:
Avogadro_number = 6.02e23 #1/mol
Volume_conversion_from_cubicmeters_to_cubicangstrom = 1e30 #The final number density is given in 1/Å³.
#With these variables we compute first the total number density as 1/m³
Total_number_solvent_molecules_per_cubic_meter = (Experimental_bulk_density*Avogadro_number)/Molar_mass
#The final number density is in 1/Å³
Final_density=Total_number_solvent_molecules_per_cubic_meter/Volume_conversion_from_cubicmeters_to_cubicangstrom

#Then, the water activity for a n=len(components) number of components is:
activity_coefficient=water_activity(len(components),Number_densities,Flattened_matrix_of_Gijs,Final_density,Number_of_integration_points)


print(f'water activity coefficient= {activity_coefficient}')

Water_molar_fraction = Number_densities[0]/(Number_densities[0]+np.sum(Number_densities[1:]))

activity = activity_coefficient*Water_molar_fraction

print(f'water activity= {activity}')

# 14) Optional! By uncommenting the two following lines we can print in latex format the expressions for the chemical potential derivatives.
# n=3 #This is the number of components
# Latex_expressions(n) #This function prints the expressions for the chemical potential derivatives given a number of components (n).


