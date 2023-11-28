
from glob import glob
import random
import pickle

import torch
from torch_geometric.data import Data, Dataset

from atomic_structure_graphs import AtomicStructureGraphs

featureList = ["atomic_number"]

class GraphDataSet(Dataset):

    """
    :Class:

    Data set class to load molecular graph data
    """

    def __init__(
        self,
        database_dir: str,
        graphs: AtomicStructureGraphs,
        nMaxEntries: int = None,
        seed: int = 42,
        transform: object = None,
        pre_transform: object = None,
        pre_filter: object = None,
        file_extension: str = '.pkl'
    ) -> None:

        """

        Args:

        :param str database_dir: the directory where the data files reside

        :param AtomicStructureGraphs graphs: an object of class AtomicStructureGraphs 
                  whose function is to read each file in the data-base and 
                  return a graph constructed according to the particular way
                  implemented in the class object (see AtomicStructureGraphs
                  for a description of the class and derived classes)

        :param int nMaxEntries: optionally used to limit the number of clusters
                  to consider; default is all

        :param int seed: initialises the random seed for choosing randomly
                  which data files to consider; the default ensures the
                  same sequence is used for the same number of files in
                  different runs

        :param str file_extension: the extension of files in the database; default = .xyz

        """

        super().__init__(database_dir, transform, pre_transform, pre_filter)

        self.database_dir = database_dir

        self.graphs = graphs

        filenames = database_dir + "/*"+file_extension

        files = glob(filenames)

        self.n_structures = len(files)

        """
        filenames contains a list of files, one for each cluster in
        the database if nMaxEntries != None and is set to some integer
        value less than n_structures, then nMaxEntries clusters are
        selected randomly for use.
        """

        if nMaxEntries and nMaxEntries < self.n_structures:

            self.n_structures = nMaxEntries
            random.seed(seed)
            self.filenames = random.sample(files, nMaxEntries)

        else:

            self.n_structures = len(files)
            self.filenames = files

    def len(self) -> int:
        """
        :return: the number of entries in the database
        :rtype: int 
        """

        return self.n_structures

    def get(self, idx: int) -> Data:

        """
        This function loads from file the corresponding data for entry
        idx in the database and returns the corresponding graph read
        from the file
  
        Args:

        :param int idx: the idx'th entry in the database
        :return: the idx'th graph in the database
        :rtype: torch_geometric.data.Data

        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.filenames[idx]

        #structure_graph = self.graphs.structure2graph(file_name)
        
        with open(file_name, 'rb') as f:
            structure_graph = pickle.load(f)

        return structure_graph

    def get_file_name(self, idx: int) -> str:

        """
        Returns the cluster data file name
        
        :param int idx: the idx'th entry in the database
        :return: the filename containing the structure data corresponding
           to the idx'th entry in the database
        :rtype: str

        """

        return self.filenames[idx]
