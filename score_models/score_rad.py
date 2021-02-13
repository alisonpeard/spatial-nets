import os
# import pickle
import argparse
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from spatial_nets.locations import *
from spatial_nets import utils


parser = argparse.ArgumentParser(description='Parse input files')

# Positional arguments
parser.add_argument('flow_file', help='The name of the file containing flows')
parser.add_argument('dmat_file', help='The file constaining the distance matrix')

# Optional arguments
parser.add_argument('-c', '--usecoords', action='store_true',
                    help='use coords instead of distance matrix')
parser.add_argument('-e', '--exact', action='store_true',
                    help='Use exact calculation of p-values')
parser.add_argument('-s', '--significance', type=float, default=0.01,
                    help='Significance threshold')
parser.add_argument('-o', '--output', default=None,
                    help='Name of output file for incdices')

# Parse arguments
args = parser.parse_args()
flow_file = args.flow_file
dmat_file = args.dmat_file
sig = args.significance
output_file = args.output

if output_file:
    assert (ext := os.path.splitext(output_file)[1]) == '.npz', \
            f"invalid file extension {ext}; use '.npz'"

assert sig > 0, 'significance should be positive'

#flow_file = os.path.join('data', 'UK_commute2011.npz')
# We read the flows to parse the filename
print(f"\nReading flows from '{os.path.basename(flow_file)}'")
T_data = utils.load_flows(flow_file, zero_diag=True)

nb_locations, nb_pairwise_flows = T_data.shape[0], T_data.nnz
print(f'Nb of locations loaded: {nb_locations}')
print(f'Effective density of flow data: {nb_pairwise_flows / nb_locations**2:.3f}')

#dmat_file = os.path.join('data', 'UK_geodesic_dmat.mat')
# Read the dmat file or the coordinates
if args.usecoords:
    assert (ext := os.path.splitext(dmat_file)[1]) == '.npy', \
            f"unsupported extension {ext}; use '.npy'"
    print(f"\nReading coordinates from '{os.path.basename(dmat_file)}'")
    dmat = np.load(dmat_file)
else:
    print(f"\nReading distance matrix from '{os.path.basename(dmat_file)}'")
    dmat = utils.load_dmat(dmat_file)

locs = DataLocations(dmat, T_data)


# Start the calculations proper
fmat = locs.radiation_matrix(finite_correction=False)  # normalised later

# Radiation: unconstrained
ct = 'unconstrained'
T_unc = locs.constrained_model(fmat, ct)

# Radiation: production
ct = 'production'
T_prod = locs.constrained_model(fmat, ct)
pmat_prod = locs.probability_matrix(T_prod, ct)

# Radiation: attraction
ct = 'attraction'
fmat = locs.radiation_matrix(finite_correction=False)
T_attrac = locs.constrained_model(fmat, ct)
pmat_attrac = locs.probability_matrix(fmat, ct)

# Radiation: doubly
ct = 'doubly'
T_doubly = locs.constrained_model(fmat, ct)
pmat_doubly_prod = locs.probability_matrix(T_doubly, 'production')
pmat_doubly_attrac = locs.probability_matrix(T_doubly, 'attraction')

models = [T_unc, T_prod, T_attrac, T_doubly]

cpc = [CPC(T_data, model) for model in models]
cpl = [CPL(T_data, model) for model in models]
rms = [RMSE(T_data, model) for model in models]
results = np.array([cpc, cpl, rms])

rows = ['CPC', 'CPL', 'RMSE']
cols = ['', 'Rad. Unc.', 'Rad. Prod.', 'Rad. Attr.', 'Rad. Doubly']
tab = [[rows[i]] + results[i].tolist() for i in range(len(rows))]

print()
print(tabulate(tab, headers=cols, floatfmt='.3f'))
print()

# Calculate the number of significant edges
models = [pmat_prod, pmat_attrac, pmat_doubly_prod, pmat_doubly_attrac]
names = ['radiation', 'radiation', 'rad-doubly-p', 'rad-doubly-a']
types = ['production', 'attraction', 'production', 'attraction']

exact_mat = np.zeros((4, len(models)))
approx_mat = np.zeros((4, len(models)))

save_dict = dict()

for k, (pmat, constr) in enumerate(tqdm(zip(models, types), total=len(names))):
    if args.exact:
        pvals = locs.pvalues_exact(pmat, constraint_type=constr)
        significant = pvals < sig
        exact_plus, exact_minus = significant[:, 0], significant[:, 1]

        n = len(exact_plus)
        nplus, nminus = np.sum(exact_plus), np.sum(exact_minus)
        exact_mat[:, k] = [nplus, nminus, n - nplus - nminus, n]

        name = f'{names[k]}_{constr}_exact'
        save_dict[name] = pvals.copy()  # copy maybe not needed

    pvals = locs.pvalues_approx(pmat, constraint_type=constr)
    significant = pvals < sig
    approx_plus, approx_minus = significant[:, 0], significant[:, 1]

    n = len(approx_plus)
    nplus, nminus = np.sum(approx_plus), np.sum(approx_minus)
    approx_mat[:, k] = [nplus, nminus, n - nplus - nminus, n]

    # Approx p-vals aren't saved
    # name = f'{names[k]}_{constr}_approx'
    # save_dict[name] = pvals.copy()  # copy maybe not needed

rows = ['Positive (observed larger)', 'Negative (model larger)',
        'Not-significant', 'Total']
cols = ['', 'Rad. Prod.', 'Rad. Attr.', 'Rad. Dp', 'Rad. Da']

if args.exact:
    print('\nExact p-value')
    tab = [[rows[k]] + exact_mat[k].tolist() for k in range(len(rows))]
    print(tabulate(tab, headers=cols))

print('\nNormal approximation')
tab = [[rows[k]] + approx_mat[k].tolist() for k in range(len(rows))]
print(tabulate(tab, headers=cols))

# Save to file
if output_file:
    print(f'Saving file: {output_file}')
    np.savez(output_file, **save_dict)
    # with open(output_file, 'wb') as f:
    #     f.write(pickle.dumps(save_dict))

print('\nDone!\n')
