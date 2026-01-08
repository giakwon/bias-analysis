"""
Program that allows the user to enter a query (space-separated words) 
and returns to the user a ranked list of the documents that match
the query.

Mode 1: Input a last name and returns a list of the 5 most similar last 
names by ethnic group

Mode 2: Input an ethnic group and returns the top 5 most common
last names within that ethnic group.
"""


from sklearn.metrics.pairwise import cosine_similarity 
import sys
import csv
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import json

def main():

    mode = sys.argv[1]
    # command line error
    if (len(sys.argv) != 2) or (mode != "1") and (mode != '2'):
        print("command-line argument needs to be added (1 or 2)")
        return
    
    # read in pkl files created in previous programs 
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    f.close()
    with open('ethnicity.pkl', 'rb') as f:
        ethnicities = pickle.load(f)
    f.close()
    with open('matrix.pkl', 'rb') as f:
        matrix = pickle.load(f)
    f.close()

    # create the tfidf_transformer
    tfidf_transformer = TfidfTransformer()

    # find the df values ("fit")
    tfidf_transformer.fit(matrix)

    # adjust the tfidf values to use the formulation we used in class
    if tfidf_transformer.idf_.min() > 1:
        tfidf_transformer.idf_ -= 1

    # apply tfidf to the matrix and save as tfidf_matrix
    tfidf_matrix = tfidf_transformer.transform(matrix)

    while True:

        if mode == "1":
            # ask user to enter a query (space-separated words)
            query_prompt = input("Enter a query(space-separated words) or 'c' to exit: ")

            if query_prompt == "c":
                break
            if query_prompt not in names:
                print("Last name not found (write in all caps).")
                continue

            index = names[query_prompt]
            query_vector = tfidf_matrix[index]        

            # use cosine similarity to compare the query to every row in the matrix
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            # print(similarities)
            ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)           

            reversed_names = {}
            for k, v in names.items():
                reversed_names[v] = k

            top_five_names = []
            top_five_ethnicities = []
            top_five_similarities = []

            # for visualize.py
            top_100 = {}
            for i in range(100):
                row, similarity = ranked[i+1]
                if row in reversed_names:
                    top_100[i] = reversed_names[row].lower(), similarity
            if query_prompt == "YODER":
                with open("yoder.json", "w") as f:
                    json.dump(top_100, f)
            if query_prompt == "WASHINGTON":
                with open("washington.json", "w") as f:
                    json.dump(top_100, f)
            if query_prompt == "ZHANG":
                with open("zhang.json", "w") as f:
                    json.dump(top_100, f)
            if query_prompt == "SAMPSON":
                with open("sampson.json", "w") as f:
                    json.dump(top_100, f)
            if query_prompt == "ALI":
                with open("ali.json", "w") as f:
                    json.dump(top_100, f)
            if query_prompt == "BARAJAS":
                with open("barajas.json", "w") as f:
                    json.dump(top_100, f)


            for i in range(5):
                row, similarity = ranked[i+1]
                # get top 5 similar names
                if (row in reversed_names): 
                    top_five_names.append(reversed_names[row].lower())
                    top_five_similarities.append(similarity)
                else:
                    pass

                # get the ethnicities for each name
                e_vector = matrix[row]
                e_array = e_vector.toarray()
                e_list = e_array[0]
                max = np.argmax(e_list)
                e = ""
                for k in ethnicities:
                    if ethnicities[k] == max: 
                        e = k
                        break
                top_five_ethnicities.append(e)

            print("top 5 most similar names to", query_prompt.lower())                   
            for i in range(len(top_five_names)):
                if top_five_ethnicities[i] == "PERCENT NON-HISPANIC OR LATINO WHITE ALONE":
                    top_five_ethnicities[i] = "white"
                if top_five_ethnicities[i] == "PERCENT NON-HISPANIC OR LATINO BLACK OR AFRICAN AMERICAN ALONE":
                    top_five_ethnicities[i] = "black/african american"
                if top_five_ethnicities[i] == "PERCENT NON-HISPANIC OR LATINO ASIAN AND NATIVE HAWAIIAN AND OTHER PACIFIC ISLANDER ALONE":
                    top_five_ethnicities[i] = "asian/pacific islander"
                if top_five_ethnicities[i] == "PERCENT NON-HISPANIC OR LATINO AMERICAN INDIAN AND ALASKA NATIVE ALONE":
                    top_five_ethnicities[i] = "american indian/alaska native"
                if top_five_ethnicities[i] == "PERCENT NON-HISPANIC OR LATINO TWO OR MORE RACES":
                    top_five_ethnicities[i] = "multiracial"
                if top_five_ethnicities[i] == "PERCENT HISPANIC OR LATINO ORIGIN":
                    top_five_ethnicities[i] = "hispanic or latino"
                print(f"{top_five_names[i]:<15} {top_five_ethnicities[i]:<30} {top_five_similarities[i]}")

        elif mode == "2":
            valid_ethnicities = ["white", "black", "african american", "asian", "pacific islander", "american indian", "alaska native", "multiracial", "hispanic", "latino"]
            print("valid ethnicities:", ", ".join(valid_ethnicities))

            # ask user to enter a query (space-separated words)
            query_prompt = input("Enter a query(space-separated words) or 'c' to exit: ")
            query_prompt_original = query_prompt

            if query_prompt == "c":
                break

            if query_prompt not in valid_ethnicities:
                print("ethnicity not in list (lowercase)")
                continue
            

            if query_prompt == "white":
                query_prompt = "PERCENT NON-HISPANIC OR LATINO WHITE ALONE"
            if query_prompt == "black" or query_prompt == "african american":
                query_prompt = "PERCENT NON-HISPANIC OR LATINO BLACK OR AFRICAN AMERICAN ALONE"
            if query_prompt == "asian" or query_prompt == "pacific islander":
                query_prompt = "PERCENT NON-HISPANIC OR LATINO ASIAN AND NATIVE HAWAIIAN AND OTHER PACIFIC ISLANDER ALONE"
            if query_prompt == "american indian" or query_prompt == "alaska native":
                query_prompt = "PERCENT NON-HISPANIC OR LATINO AMERICAN INDIAN AND ALASKA NATIVE ALONE"
            if query_prompt == "multiracial":
                query_prompt = "PERCENT NON-HISPANIC OR LATINO TWO OR MORE RACES"
            if query_prompt == "hispanic" or query_prompt == "latino":
                query_prompt = "PERCENT HISPANIC OR LATINO ORIGIN"

            index = ethnicities[query_prompt]
            array = matrix.toarray()
            e_lst = []
            for row in array:
                e_lst.append(row[index])


            e_lst = np.array(e_lst)
            sorted_lst = e_lst.argsort()
            top_five_names = sorted_lst[-5:]

            reversed_names = {}
            for k,v in names.items():
                reversed_names[v] = k
            
            print("top 5 most common names in", query_prompt_original, "ethnic group")
            for i in top_five_names:
                name = reversed_names.get(i).lower()
                percent = e_lst[i]
                print(f"{name:<15} {percent:.2f}%")

main()


