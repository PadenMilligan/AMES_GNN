
from typing import Dict, List

from mendeleev import element
import numpy as np
from scipy.constants import physical_constants
import torch
from torch_geometric.data import Data
import math

from atomic_structure_graphs import AtomicStructureGraphs
from device import device
from features import Features
from QM9_utils import read_QM9_structure

class QM9CovalentMolecularGraphs(AtomicStructureGraphs):

    """

    A class to read molecule information from the database file
    and convert it to torch_geometric Data graph. In this class, graphs
    are constructed in a chemically intuitive way: a node (atom) has edges
    only to other nodes that are at a distance that is up to alpha times the
    sum of their respective covalent radii away, where alpha is
    a factor >= 1 (default 1.1). In this mode edges will correspond
    to chemical bonds. Covalent radii are extracted from Mendeleev for
    each listed species.

    """

    def __init__(
        self,
        species_list: List[str],
        edge_features: Features,
        bond_angle_features: Features,
        dihedral_features: Features = None,
        node_feature_list: List[str] = ['atomic_number'],
        n_total_node_features: int = 10,
        n_max_neighbours: int = 6,
        pooling: str = "add",
        alpha: float = 1.1
    ) -> None:

        # initialise the base class

        super().__init__(
           species_list, 
           edge_features,
           bond_angle_features,
           dihedral_features,
           node_feature_list,
           n_total_node_features, 
           pooling
        )

        self.alpha = alpha  # alpha is the scaling factor for bond (edge)
        # critera, i.e. two atoms are bonded if their
        # separation is r <= alpha*(rc1 + rc2), where
        # rci are the respective covalent radii

        self.covalent_radii = self.get_covalent_radii()

        # define a conversion factor from Hartrees to eV
        self.Hartree2eV = physical_constants["Hartree energy in eV"][0]

        """
        The following dictionary defines the atomic ref energies, needed
        to calculate atomisation energies. These values are in Hartree
        """


    def get_covalent_radii(self) -> Dict[str, float]:

        """

        Sets up and returns a dictionary of covalent radii (in Ang)
        for the list of species in its argument

        :return: covalent_radii: dict of covalent radius for eash species (in Angstrom)
        :rtype: dict

        """

        covalent_radii = {}
        
        for label in self.species:

            spec = element(label)

            covalent_radii[label] = spec.covalent_radius / 100.0
            # mendeleev stores radii in pm, hence the factor

        return covalent_radii

    def structure2graph(self, fileName: str) -> Data:

        """

        A function to turn atomic structure information imported from
        a  database file and convert it to torch_geometric Data graph. 
        In this particular class graphs are constructed in the following way:

        Chemically intuitive way: a node (atom) has edges only to
           other nodes that are at a distance that is up to alpha times the
           sum of their respective covalent radii away. In this mode
           edges will correspond to chemical bonds. To activate this
           mode it is necessary to pass the dictionary covalent_radii;
           if it is not passed or is set to None, the second mode is
           activated (see below).

        Args:

        :param: fileName (string): the path to the file where the structure
           information is stored in file.
        :type: str
        :return: graph representation of the structure contained in fileName
        :rtype: torch_geometric.data.Data

        """

        (
            molecule_id,
            n_atoms,
            labels,
            positions,
            properties,
            result
        ) = read_QM9_structure(fileName)

        # the total number of node features is given by the species features

        n_features = self.spec_features[labels[0]].size + \
                     self.bond_angle_features.n_features()
        node_features = torch.zeros((n_atoms, n_features), dtype=torch.float32)

        # atoms will be graph nodes; edges will be created for every
        # neighbour of i that is among the nearest
        # n_max_neighbours neighbours of atom i

        # first we loop over all pairs of atoms and calculate the matrix
        # of squared distances

        dij2 = np.zeros((n_atoms, n_atoms), dtype=float)

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):

                rij = positions[j, :] - positions[i, :]
                rij2 = np.dot(rij, rij)

                dij2[i, j] = rij2
                dij2[j, i] = rij2

        n_max = 12 # expected maximum number of neighbours bonded to node
        n_neighbours = np.zeros((n_atoms), dtype=int)
        neighbour_distance = np.zeros((n_atoms, n_max), dtype=float)
        neighbour_index = np.zeros((n_atoms, n_max), dtype=int)

        node0 = []
        node1 = []
        
        for i in range(n_atoms - 1):

            for j in range(i + 1, n_atoms):
                
                dcut = self.alpha * (
                    self.covalent_radii[labels[i]] + self.covalent_radii[labels[j]]
                )

                dcut2 = dcut * dcut

                if dij2[i, j] <= dcut2:

                    node0.append(i)
                    node1.append(j)

                    node0.append(j)
                    node1.append(i)

                    dij = np.sqrt(dij2[i, j])

                    neighbour_distance[i, n_neighbours[i]] = dij
                    neighbour_distance[j, n_neighbours[j]] = dij

                    neighbour_index[i, n_neighbours[i]] = j
                    neighbour_index[j, n_neighbours[j]] = i

                    n_neighbours[i] += 1
                    n_neighbours[j] += 1

        edge_index = torch.tensor([node0, node1], dtype=torch.long)
        
        _, num_edges = edge_index.shape

        # construct node geometric features; these will be appended at the end
        # of the features that are purely related to the species

        n_ba_features = self.bond_angle_features.n_features()

        for i in range(n_atoms):

            anglehist = np.zeros((n_ba_features), dtype=float)

            for n in range(n_neighbours[i] - 1):

                j = neighbour_index[i, n]

                rij = positions[j, :] - positions[i, :]
                dij = neighbour_distance[i, n]

                for m in range(n + 1, n_neighbours[i]):

                    k = neighbour_index[i, m]

                    rik = positions[k, :] - positions[i, :]
                    dik = neighbour_distance[i, m]

                    costhetaijk = np.dot(rij, rik) / (dij * dik)

                    anglehist += self.bond_angle_features.u_k(costhetaijk)

            node_total_features = np.concatenate(
            ( 
                self.spec_features[labels[i]], anglehist 
            ) 
            )

            node_features[i, :] = torch.from_numpy(node_total_features)

        # now, based on the edge-index information, we can construct the edge attributes

        bond_features = []

        for n in range(num_edges):

            i = edge_index[0, n]
            j = edge_index[1, n]

            dij = np.sqrt(dij2[i, j])

            bond_features.append(self.edge_features.u_k(dij))


        # it is apparently faster to convert numpy arrays to tensors than
        # to convert arrays of numpys, so let's do it this way

        features = np.array(bond_features)
        result = np.array(result)


        # now we can create a graph object (Data)

        edge_attr = torch.tensor(features, dtype=torch.float32)
        overall = torch.tensor(result, dtype = torch.float32)

        count_x = 0
        for i in node_features:
            count_y = 0
            for j in i:
                if math.isnan(j.item()):
                    node_features[count_x][count_y] = float(0)
                count_y += 1
            count_x += 1

        pos = torch.from_numpy(positions)

        structure_graph = Data(
            x=node_features.to(device), 
            y=overall.to(device), 
            edge_index=edge_index.to(device), 
            edge_attr=edge_attr.to(device), pos=pos
        )

        # we do not put pos in device at the moment, since this is not needed in the fitting

        return structure_graph

# register this derived class as subclass of AtomicStructureGraphs

AtomicStructureGraphs.register(QM9CovalentMolecularGraphs)
