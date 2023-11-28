
from abc import ABC, abstractmethod
import re
from typing import Dict, List, Tuple

import numpy as np
from torch_geometric.data import Data
from mendeleev import element

from device import device
from features import Features

class AtomicStructureGraphs(ABC):

    """

    :Class AtomicStructureGraphs:

    This is an Abstract Base Class (abc) that allows easy construction
    of derived classes implementing different strategies to turn
    atomic structural information into a graph. The base class
    implement helper function

    generate_node_features: sets up the arrays of node features for each
          chemical species

    Derived classes need to implement method structure2graph, that
    takes as input an input file containing the structural information
    and returns a torch_geometric.data Data object (graph) constructed
    according to the structure2graph implementation.

    """

    def __init__(
        self,
        species_list: List[str],
        edge_features: Features,
        bond_angle_features: Features,
        dihedral_features: Features = None,
        node_feature_list: List[str] = ['atomic_number'],
        n_total_node_features: int = 10,
        pooling: str = "add"
    ) -> None:

        """

        Initialises an instance of the class. It needs to be passed the list
        of chemical species that may be found in the training files, the
        chemical feature list of each node, which is a list of data
        that Mendeleev can understand (see below), and a total number for
        the node features; besides the chemical/physical features, nodes
        can be assigned initial numerical features that are specific to each
        species and that have no physico-chemical significance (see
        generate_node_features() below for details)

        Args:

        :param species_list List[str]: the chemical symbols of the species present
             in the atomic structures
        :param edge_features Features: an instance of the Features class to define
             edge features in the graph
        :param bond_angle_features Features: an instance of the Features class to define
             bond angle features
        :param dihedral_features Features (optional): an optional instance of the Features
             class to define dihedral angle features
        :param List[str] node_feature_list: (default empty) contains
             mendeleev data commands to select the required features
             for each feature e.g. node_feature_list = ['atomic_number',
             'atomic_radius', 'covalent_radius']
        :param int n_total_node_features: (default 10) the total number 
             of features per node, counting those listed in node_feature_list 
             (if any) and the numerical ones.
        :param str pooling: (default = 'add' ) indicates the type of pooling to be
             done over nodes to estimate the fitted property; usually
             this will be 'add', meaning that the property prediction
             is done by summing over nodes; the only other contemplated
             case is 'mean', in which case the prediction is given by
             averaging over nodes. WARNING: this must be done in a
             concerted way (the same) in the GNN model definition!

        """
        
        self.species = species_list
        
        self.edge_features = edge_features
        self.bond_angle_features = bond_angle_features
        self.dihedral_features = dihedral_features

        self.node_feature_list = node_feature_list

        self.n_total_node_features = n_total_node_features

        self.spec_features = self.generate_node_features()

        self.pooling = pooling

    def generate_node_features(self) -> Dict[str, float]:

        """

        This function generates initial node features for an atomic graph

        :return: It returns a dictionary where the keys are the chemical symbol and
        the values are an array of dimensions (n_node_features), such that
        initial nodes in the graph can have their features filled according
        to their species. The array of features thus created is later used
        in molecule2graph to generate the corresponding molecular graph.
        :rtype: Dict[str, float]

        """

        n_species = len(self.species)
        n_features = len(self.node_feature_list)
        n_node_features = self.n_total_node_features - n_features

        # generate an element object for each species

        spec_list = []

        for spec in self.species:
            spec_list.append(element(spec))

        x = np.pi * np.linspace(0.0, 1.0, n_node_features)

        factor = np.ones((n_features), dtype=float)

        for n in range(n_features):

            if re.search("radius", self.node_feature_list[n]):
                # if feature is a distance, convert from pm to Ang
                factor[n] = 0.01

        # now we can loop over each individual species, create its feature
        # vector and store it in spec_features

        # we want to have node features normalised in the range [-1:1]

        values = np.zeros((n_features, n_species), dtype=float)

        for n, spec in enumerate(spec_list):

            for m in range(n_features):

                command = "spec." + self.node_feature_list[m]
                values[m, n] = eval(command)

        # now detect the maximum and minimum values for each feature
        # over the list of species we have

        features_max = np.zeros(n_features, dtype=float)
        features_min = np.zeros(n_features, dtype=float)

        for m in range(n_features):
            features_max[m] = np.max(values[m, :])
            features_min[m] = np.min(values[m, :])

        # normalise values

        if n_species > 1:

           for n in range(n_species):
               for m in range(n_features):

                   values[m, n] = (
                       2.0
                       * (values[m, n] - features_min[m])
                       / (features_max[m] - features_min[m])
                       - 1.0
                   )

        else:

           values[:,0] = np.random.rand(n_features)

        spec_features = {}

        for n, spec in enumerate(spec_list):

            amplitude = 0.1 * float(spec.period)
            group = float(spec.group_id)

            vec2 = amplitude * np.sin(group * x)

            spec_features[spec.symbol] = np.concatenate((values[:, n], vec2))

        # we are done

        return spec_features

    @abstractmethod
    def structure2graph(self, file_name: str) -> Data:

        """

        This method must be implemented in derived classes. Its purpose
        is to take a file-name as input containing atomic structural
        information and return a Data (graph) object representing the
        same structure.

        Args:

        :param str file_name: path to file containing structure data

        :return: atomic graph (Data instance)
        :rtype: torch_geometric.data.Data

        """

        raise NotImplementedError