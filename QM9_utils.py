
import re
from typing import List, Tuple
import pandas as pd

import numpy as np

def read_QM9_structure(
       file_name: str 
    ) -> Tuple[int, int, List[str], float, float, list[int]]:


    """

    This function opens a file of the GDB-9 database and processes it,
    returning the molecule structure in xyz format, a molecule identifier
    (tag), and a vector containing the entire list of molecular properties

    Args:

    :param str file_name: filename containing the molecular information

    :return: molecule_id (int): integer identifying the molecule number
        in the database n_atoms (int): number of atoms in the molecule
        species (List[str]): the species of each atom (len = n_atoms)
        coordinates (np.array(float)[n_atoms,3]): atomic positions
        properties (np.array(float)[:]): molecular properties, see
        database docummentation charge (np.array(float)[n_atoms]):
        Mulliken charges of atoms
    :rtype: Tuple[int, int, List[str], float, float, float]

    """
    
    with open(file_name, "r") as file_in:
        lines = file_in.readlines()

    n_atoms = int(lines[0])  # number of atoms is specified in 1st line

    words = lines[1].split()

    molecule_id = ''

    molecular_data = np.array(words[2:], dtype=float)

    count = 0
    for i in range(len(words[0])):
        if words[0][i+1] == '_':
            molecule_id += words[0][i]
            break
        else:
            count += 1
            molecule_id += words[0][i]


    result = []
    #for i in [1364, 1365, 1366, 1367, 1368]:
        #df = pd.read_csv('Database/ames_mutagenicity_data.csv', usecols=[i])
        #result.append(int(df.iloc[int(molecule_id)+1]))

    for i in [1364, 1365, 1366, 1367, 1368]:
        df = pd.read_csv('Database/ames_mutagenicity_data.csv', usecols=[i])
        if int(df.iloc[int(molecule_id)+1]) == 1:
            result.append(1.0)
        elif int(df.iloc[int(molecule_id)+1]) == 0:
            result.append(0.0)
        elif int(df.iloc[int(molecule_id)+1]) == -1:
            result.append(-1.0)
 
    #df = pd.read_csv('Database/ames_mutagenicity_data.csv', usecols=[1369])
    #result = (int(df.iloc[int(molecule_id)+1]))

    species = []  # species label
    coordinates = np.zeros((n_atoms, 3), dtype=float)  # coordinates in Angstrom
    # charge = np.zeros((n_atoms), dtype=float)  # Mulliken charges (e)

    # below extract chemical labels, coordinates and charges

    m = 0

    for n in range(2, n_atoms + 2):

        line = re.sub(
            r"\*\^", "e", lines[n]
        )  # this prevents stupid exponential lines in the data base

        words = line.split()

        species.append(words[0])

        x = float(words[1])
        y = float(words[2])
        z = float(words[3])

        # c = float(words[4])

        coordinates[m, :] = x, y, z

        # charge[m] = c
    
        m += 1

    # finally obtain the vibrational frequencies, in cm^-1

    #frequencies = np.array(lines[n_atoms + 2].split(), dtype=float)

    # we pack all the molecular data into a single array of properties


    properties = molecular_data

    return molecule_id, n_atoms, species, coordinates, properties, result

