'''
This program creates two pickle files:
 1) maps file names to integers (names.pkl)
 2) ethnicities to integers (ethnicity.pkl)
'''

import csv
import pickle

'''
This function opens a pickle file for the names dictionary
and makes names.pkl
params:
    names_dict: dictionary of names
'''
def pickle_names(names_dict):
    with open('names.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(names_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()

'''
This function opens a pickle file for the ethnicity dictionary
and makes ethnicity.pkl
params:
    ethnicity_dict: dictionary of ethnicities
'''
def pickle_ethnicity(ethnicity_dict):
    with open('ethnicity.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(ethnicity_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def main():

    csv_file_name = '/data/cs91r-s25/misc/lastnames.csv'
    scrabble_words = '/data/cs91r-s25/scrabble/scrabble.txt'

    with open(scrabble_words) as f:
        scrabble_words = set(line.strip().upper() for line in f)
    
    # read lastnames.csv file  
    names_lst = [] 
    with open(csv_file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # make list of ids of text only written in english
        row1 = next(spamreader)
        for row in spamreader:
            name = row[0].split(',')[0]
            if name not in scrabble_words:
                names_lst.append(name)
    
    names_dict = {}
    for i in range(len(names_lst)):
        names_dict[names_lst[i]] = i
    # print(names_dict)

    ethnicity_dict = {}
    ethnicities = row1[5:]
    for i in range(len(ethnicities)):
        ethnicity_dict[ethnicities[i]] = i
    # print(ethnicity_dict)

    pickled_words = pickle_names(names_dict)
    pickled_books = pickle_ethnicity(ethnicity_dict)
        
   
main ()