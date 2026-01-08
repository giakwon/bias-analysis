'''
This program visualizes the names from our pickle files 
as well as the top name from each ethinic group 
in comparison to its top 100 most similat names using pca and tsne. 
'''

import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

def perform_tsne(matrix):
    # we want to reduce these vectors to 2 dimensions
    model = TSNE(n_components=2, perplexity=5, random_state=0)
    # learn the transformation and apply it
    reduced = model.fit_transform(matrix)
    # convert it back into a dataframe with columns "x", "y", and matching labels
    df = pd.DataFrame(reduced, columns=["x", "y"], index=matrix.index)
    return df

def perform_pca(matrix):
    # we want to reduce these vectors to 2 dimensions
    model = PCA(n_components=2)
    # learn the transformation and apply it
    reduced = model.fit_transform(matrix)
    # convert it back into a dataframe with columns "x", "y", and matching labels
    df = pd.DataFrame(reduced, columns=["x", "y"], index=matrix.index)
    return df

def plot_relations(reduced_matrix, filename):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)

    # make first half of each relation one color, second half another color
    # two lists that stores indices:  even rows ; odd rows
    evens = []
    odds = []
    for i in range(len(reduced_matrix)):
        if i % 2 == 0:
            evens.append(i)
        else:
            odds.append(i)

    plt.scatter(reduced_matrix.iloc[evens]['x'], reduced_matrix.iloc[evens]['y'], c='r', s=50)
    plt.scatter(reduced_matrix.iloc[odds]['x'], reduced_matrix.iloc[odds]['y'], c='b', s=50)

    # add text labels to each of the dots
    # iterate over each of the row labels (the words) in the reduced_matrix 
        # and annotate that (x,y) coordinate with the label
    for i, label in enumerate(reduced_matrix.index):
        plt.annotate(label, (reduced_matrix['x'].iloc[i], reduced_matrix['y'].iloc[i]), color='k')

    # connect the query name to each of the other names 
    query_x = reduced_matrix['x'].iloc[0]
    query_y = reduced_matrix['y'].iloc[0]
    for i in range(1, len(reduced_matrix)):
        x_value = reduced_matrix['x'].iloc[i]
        y_value = reduced_matrix['y'].iloc[i]
        plt.plot([query_x, x_value], [query_y, y_value], color='lightgray', linewidth=1)

    plt.savefig(filename)

def choose_file(choice):
     
    if choice == "yoder":
        with open("yoder.json", "r") as f:
            return json.load(f)
    elif choice == "washington":
        with open("washington.json", "r") as f:
            return json.load(f)
    elif choice == "zhang":
        with open("zhang.json", "r") as f:
            return json.load(f)
    elif choice == "sampson":
        with open("sampson.json", "r") as f:
            return json.load(f)
    elif choice == "ali":
        with open("ali.json", "r") as f:
            return json.load(f)
    elif choice == "barajas":
        with open("barajas.json", "r") as f:
            return json.load(f)
    else:
        print("not one of the options!")
        c = input("choose name to visualize: ")
        return choose_file(c)
    
def main():

    parser = argparse.ArgumentParser(description="Visualize last names by ethnicity using PCA or TSNE")
    parser.add_argument("-o", "--output", default="plot.png", help=" path to save the plot (default is plot.png)")
    parser.add_argument("-tsne", action="store_true", help="perform t-SNE (default is PCA)")
    args = parser.parse_args()

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    f.close()
    with open('ethnicity.pkl', 'rb') as f:
        ethnicities = pickle.load(f)
    f.close()
    with open('matrix.pkl', 'rb') as f:
        matrix = pickle.load(f)
    f.close()
    choices = ["yoder", "washington", "zhang", "sampson", "ali", "barajas"]
    print("names:", ", ".join(choices))
    choice = input("choose name to visualize: ")
    similar_names = choose_file(choice)


    array = matrix.toarray()
    mappings = {}
    for k,v in names.items():
        mappings[v] = k.lower()
    
    index = []
    names_list = []
    for n in similar_names.values():
        name = n[0]
        if name.upper() in names:
            i = names[name.upper()]
            index.append(i)
            names_list.append(name.lower())
    new_array = array[index]

    df = pd.DataFrame(new_array, index=names_list)

    if args.tsne:
        reduced_matrix = perform_tsne(df)
    else:
        reduced_matrix = perform_pca(df)

    plot_relations(reduced_matrix, args.output)

main()