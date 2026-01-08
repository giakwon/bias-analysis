'''
This program creates a  matrix by using that two pickle files created in setup.py.
'''

import csv
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def main():
    # read two pickle files created in setup.py 
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    f.close()
    with open('ethnicity.pkl', 'rb') as f:
        ethnicity = pickle.load(f)
    f.close()

    # read lastnames.csv file  
    csv_file_name = '/data/cs91r-s25/misc/lastnames.csv'
    with open(csv_file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        row1 = next(spamreader)
        data = []
        rows = []
        columns = []

        # for every file name, read the percentages
        for line in spamreader:
            name = line[0].upper()
            if name in names:
                row_index = names[name]
                # for the 6 races/ethnicities
                for i in range(6):
                    column_name = row1[5 + i]
                    percent = float(line[5 + i])
                    column_index = ethnicity[column_name]
                    rows.append(row_index)
                    columns.append(column_index)
                    data.append(percent)

    # create a coo_matrix 
    matrix = coo_matrix((data, (rows, columns)), shape=(len(names), len(ethnicity)))
    
    # convert coo_matrix to csr_matrix
    matrix_csr = matrix.tocsr()
    print(matrix_csr.toarray())

    # save csr_matrix to pickle file: matrix.pkl
    with open('matrix.pkl', 'wb') as f:
        pickle.dump(matrix_csr, f, pickle.HIGHEST_PROTOCOL)
    f.close()

main()