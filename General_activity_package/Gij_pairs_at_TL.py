import itertools
import MDAnalysis as mda
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.stats import linregress
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from collections import Counter
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

def COMs_for_residues_and_all_frames(u,resnames):
    coms_dict = {resname: [] for resname in resnames}
    for ts in tqdm(u.trajectory):
        for resname in resnames:
            selection = u.select_atoms(f'resname {resname}')
            types_of_residues = selection.residues
            if len(types_of_residues[0].atoms)==1: #If we have monoatomic components (like Na⁺), then the center of mass will be its direct coordinates!
                coms_dict[resname].append(np.array(types_of_residues.atoms.positions))
            else: #If we have more than 1 atom in our molecule, the COM is computed.
                coms_this_frame = [res.atoms.center_of_mass() for res in selection.residues]
                coms_dict[resname].append(np.array(coms_this_frame))
    for resname in resnames:
        coms_dict[resname] = np.array(coms_dict[resname])
    return coms_dict

def Get_heaviest_atom_from_solvent(u, solvent_resname="SOL"):
    solvent_residues = u.select_atoms(f"resname {solvent_resname[0]}").residues # We get the solvent residues
    first_solvent = solvent_residues[0] # All the residues of the same type have the same name, so let's select the first one only.
    Heaviest_atom = max(first_solvent.atoms, key=lambda a: a.mass) # Here we obtain the heaviest atom from that solvent residue. This is used as a center of mass of the residue.
    Heaviest_atom_name = Heaviest_atom.name # We want to know what is the name of the heaviest atom
    coms_dict = {resname: [] for resname in solvent_resname}
    heaviest_atoms = u.select_atoms(f"resname {solvent_resname[0]} and name {Heaviest_atom_name}") # We select all the heaviest atoms from the solvent residues. 
    for ts in tqdm(u.trajectory):
        Positions_of_the_heaviest_atoms = [pos for pos in heaviest_atoms.positions]
        coms_dict[solvent_resname[0]].append(np.array(Positions_of_the_heaviest_atoms))
    coms_dict[solvent_resname[0]]=np.array(coms_dict[solvent_resname[0]])
    return coms_dict # We use these atoms as a COM for the solvent. For example, for the case of water will be the oxygen.

def subvolumes_and_components(u,solvent,largest_subV,TotNumb_subvolumes,solvent_COMs,smallest_subV=0.001):
    residue_counts = Counter(res.resname for res in u.residues) #Here we gather the residues and how many of those.
    other_resnames_than_the_solvent = [resname for resname, _ in residue_counts.most_common() if resname not in solvent] #List with the rest of residues ordered from more to less amount.
    ordered_resnames = solvent + other_resnames_than_the_solvent #We ensure that the solvent is first
    if solvent_COMs=='on': #Only if we want to compute the center of mass of every molecule. The components are the COM coordinates, one per molecule.
        components = COMs_for_residues_and_all_frames(u,ordered_resnames)
    else: #In this case for the solvent (water) we get the oxygen as the "COM", since it's cheaper than computing the COMs for each water molecule for a really large system. For the rest of the molecules the COM is computed.
        Heaviest_atoms_from_solvent = Get_heaviest_atom_from_solvent(u,solvent) #Coordinates of the heaviest atom, oxygen for water.
        COMs_from_the_rest = COMs_for_residues_and_all_frames(u,other_resnames_than_the_solvent) #COM for the rest of the molecules.
        components = Heaviest_atoms_from_solvent|COMs_from_the_rest #Merging both dictionaries.
    List_with_total_residues=[] #This is in order to compute the number densities in N/Å³
    List_names_of_residues = []
    for indx, name_of_residue in enumerate(components): #Here we inform what are the names of our residues and how many atoms each contains.
        Total_num_molecules=len(components[name_of_residue][0]) #Check just for the first frame, the total number of residues do not change over time.
        List_with_total_residues.append(Total_num_molecules)
        List_names_of_residues.append(name_of_residue)
        print(f"{name_of_residue}: {Total_num_molecules} residues")
    V_total = np.mean([np.prod(u.dimensions[:3]) for ts in u.trajectory])  # An average for the total volume of our box (for NPT, in case of NVT is the exact one)
    Number_densities = List_with_total_residues/V_total #in N/Å³ (also average number densities, since the averaged volume is used)
    # Defining the sizes of subvolumes
    start_log=np.log10(smallest_subV)
    stop_log=np.log10(largest_subV)
    subv_coeffs=np.logspace(start_log,stop_log,TotNumb_subvolumes)
    subvolumes = list([subv_coeffs[i]*V_total for i in range(len(subv_coeffs))]) #The subvolumes are the fraction we give times the total volume.
    return subvolumes,V_total,components,Number_densities,List_names_of_residues

def computing_G_lamb(V,u, V_total, components,NSVs):
    R_max = ((3 * V) / (4 * np.pi))**(1/3) #Radius of the subvolume
    N_accumulated = {name: [] for name in components} #for each component we create an empty list in a dictionary.
    for ts in u.trajectory: #loop over all the frames in the trajectory.
            box = list(u.dimensions[:3])+[90.0, 90.0, 90.0]
            for _ in range(NSVs): #Loop over the amount of subvolumes of a given size that we want to sample for each snapshot.
                subvol_center = np.random.uniform(0, u.dimensions[:3]) # Each subvolume is centered randomly within the box
                for indx, name_of_residue in enumerate(components):
                    coordinates = components[name_of_residue][ts.frame] #These are the coordinates of a given residue at a given frame.
                    dists = distance_array(coordinates, subvol_center[None,:], box=box).flatten() #Distance from the coordinates from the COM of the residue to the center of the subvolume. PBC are used, indicated in box=box.
                    selected = coordinates[dists <= R_max] # Here we select those residues within the subvolume with radius R_max.
                    N_accumulated[name_of_residue].append(len(selected)) #N_accumulated gives us a dictionary, where the keys are the name of the residues and the values are a list of how many residues per subvolume over all frames. For example, for only 2 subvolumes and 3 frames we would have a list with 5 components.
    N_arrays = {name: np.array(counts) for name, counts in N_accumulated.items()} #Converting the list of residue count to numpy arrays Nij
    N_means = {name: np.mean(vals) for name, vals in N_arrays.items()} #computing the mean/ensemble average <Nij>
    G_results = {} #Dictionary where we will put all the Gij for all the components
    lmbda = (V / V_total)**(1/3) #Defining lambda as a function of the subvolumes.
    for i, j in itertools.combinations_with_replacement(components.keys(), 2): #Here we compute the Gij based on the SBA theory.
            Ni, Nj = N_arrays[i], N_arrays[j]
            mean_i, mean_j = N_means[i], N_means[j]
            cov_ij = np.mean(Ni * Nj) - mean_i * mean_j
            denom = mean_i * mean_j
            G_ij = V_total * lmbda**3 * (cov_ij / denom)
            if i == j:
                G_ij -= V_total * lmbda**3 / mean_i
            G_results[(i, j)] = G_ij
    return G_results

def SBA(output,subvolumes,V_total,threshold_to_fit=0.3,plot="yes"):
    slope_all=[] #All the slopes will be saved here
    intercept_all=[] #Similarly for the intersections. 
    #Here we compute all the lambda*Gij and put it into a dictionary:
    lambda_G = defaultdict(list)
    for i, G_dict in enumerate(output):
        lamb = (subvolumes[i] / V_total)**(1/3)
        for pair, Gij in G_dict.items():
            lambda_G[pair].append(lamb * Gij)
    for pair in lambda_G:
        lambda_G[pair] = np.array(lambda_G[pair])
    lambdas = (subvolumes / V_total)**(1/3) #A list with all lambdas as a function of subvolumes.
    mask = lambdas < threshold_to_fit #In this mask we filter the lambdas below a threshold for fitting.
    fit_results = {} #In this dictionary we want to save slopes, interceptions and such.
    for pair, values in lambda_G.items():
        slope, intercept, r_value, p_value, stderr = linregress(lambdas[mask], values[mask])
        fit_results[pair] = {
            'slope': slope,
            'intercept': intercept,
            'r': r_value,
            'p': p_value,
            'stderr': stderr
        }
    for pairs, the_rest in fit_results.items(): #Collecting all the slopes and intercepts for each pair in lists.
        slopes=the_rest['slope']
        intercepts=the_rest['intercept']
        slope_all.append(slopes)
        intercept_all.append(intercepts)
    if plot=="yes": #Alternative option to see the plot of the λ G(λ) vs λ curves and the initial fit for small λ.
        colors = cm.rainbow(np.linspace(0, 1, len(output[0])))  # Choose a colormap
        plt.figure()
        for elements,pairs in enumerate(lambda_G):
            pairs_str = f"{pairs[0]},{pairs[1]}" #This line is to enable the subindices in the legend.
            plt.scatter((subvolumes / V_total)**(1/3), lambda_G[pairs], color=colors[elements], label=f'Computed $λ G_{{{pairs_str}}}(λ)$')
            plt.plot((subvolumes / V_total)**(1/3), slope_all[elements] * (subvolumes / V_total)**(1/3) + intercept_all[elements], color=colors[elements],linestyle='--', label=f'Linear Fit: $G^\infty = {slope_all[elements]:.4f}$')
        plt.axvline(x=threshold_to_fit,color='black',linestyle='--')
        plt.xlabel(r'$\lambda = \left(V/V_0\right)^{1/3}$', fontsize=12)
        plt.ylabel(r'$\lambda\,G(\lambda)$', fontsize=12)
        plt.title("SBA Method: Extrapolating to Thermodynamic Limit", fontsize=14)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.savefig("SBA_and_fit.pdf",bbox_inches='tight',format = 'pdf')
        plt.show()
    return slope_all

def run_parallel(subvolumes,u,V_total,components,NSVs): #This function enables to run each subvolume to a different CPU in parallel.
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPUs for parallel execution...")
    function = partial(computing_G_lamb,u=u,V_total=V_total,components=components,NSVs=NSVs)
    with Pool(num_cpus) as pool:
        try:
            output = pool.map(function, subvolumes)
        except ValueError as e:
            print("Error in subprocess:", e)
            pool.terminate()
            return None
    return output

def expand_symmetric_upper_triangle(upper_triangle_values):
    """
    Given a list of values from the upper triangle (including diagonal)
    of a symmetric matrix, expand it to the full symmetric matrix and
    return a flattened list row-by-row.
    """
    # Determine the size n of the matrix
    n = int((-1 + (1 + 8 * len(upper_triangle_values))**0.5) / 2)
    assert n * (n + 1) // 2 == len(upper_triangle_values), "List length does not match any symmetric matrix upper triangle size"
    # Initialize empty matrix
    matrix = np.zeros((n, n))
    idx = 0
    # Fill upper triangle
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = upper_triangle_values[idx]
            matrix[j, i] = upper_triangle_values[idx]  # Mirror to lower triangle
            idx += 1
    # Flatten row by row
    return matrix.flatten().tolist()