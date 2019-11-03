#!/usr/bin/env

""" This file contains code trying to predict Direction of Trade communities from World Bank country data """

import pickle
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import pprint
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import itertools


class NetworkLearning:

    def __init__(self):
        """
        Reads in network info from 'data/communities.pkl' to assign targets
        """

        # read in network info datafile
        with open('data/communities.pkl', 'rb') as f:
            self.network_info = pickle.load(f)

        # assign targets from data
        self.targets = self.create_country_targets()

    def plot_modularity_over_time(self):    #pragma: no cover
        """
        Saves plot of modularity of networks over time.

        :return: nothing, saves plot to 'plots/modularity.png'
        """

        years = self.network_info.keys()
        mods = [v['modularity'] for k, v in self.network_info.items()]

        plt.plot(years, mods)
        plt.xlabel('Year')
        plt.ylabel('Modularity')
        plt.title('Modularity through time')
        plt.savefig('plots/modularity.png')

    def plot_degrees_through_time(self):    #pragma: no cover
        """
        Saves plot of average degree of networks with number of nodes & edges over time.

        :return: nothing, saves plot to 'plots/degrees.png'
        """

        years = self.network_info.keys()
        out_deg = [float(v['Average out degree']) for k, v in self.network_info.items()]
        edges = [int(v['Number of edges']) for k, v in self.network_info.items()]
        nodes = [int(v['Number of nodes']) for k, v in self.network_info.items()]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Degree/Number nodes')
        ln1 = ax1.plot(years, out_deg, color='orange', label="In Degree")
        ln2 = ax1.plot(years, nodes, color='red', label="Nodes")
        ax1.set_yticks(np.arange(30, 250, 20))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('Number edges')
        ln3 = ax2.plot(years, edges, color='blue', label="Edges")
        ax2.set_yticks(np.arange(3000, 33000, 3000))

        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs)

        plt.title('Average degree and number of nodes/edges through time')
        plt.tight_layout()
        plt.savefig('plots/degrees.png')

    def create_country_targets(self):
        """
        Returns dictionary mapping countries, then years, to community number, from saved 'data/communities.pkl' file.

        :return: dictionary with country, then years, as keys, and community numbers as values
        """

        comm_dict = {}
        for year in self.network_info.keys():
            year_dict = {}
            # start community number at 0 (doesn't start from 0 in data file)
            community = 0
            for _, members in self.network_info[year]['communities'].items():
                # add country's community number to year_dict
                for country in members:
                    year_dict[country] = community

                community += 1

            # add year's data to community dictionary
            comm_dict[year] = year_dict

        return comm_dict

    def identify_country_name_mapping(self, feat_dict, year):
        """
        Matches World Bank country names to IMF country names; returns matches and unmatched names.

        :param feat_dict: dictionary of countries and their features
        :param year: integer year for which to do name mapping
        :return: 1. dictionary mapping country names from feature dataset to target dataset
                 2. list of target country names not matched
                 3. list of feature country names not matched
        """

        ignore_words = ['Rep.', 'of', 'North', 'South', 'Republic', 'Democratic', 'and', '&', 'P.R.:', 'Middle',
                        'Islands', 'Dem.', 'the', 'Arab', 'Asia', 'French', 'China', 'Islamic', 'Africa',
                        'Other', 'St.', 'The', 'Kingdom', 'Central', 'Europe', 'East', 'West', 'PDR', 'People\'s',
                        'middle', 'New', 'Northern']

        mapping = {}
        missing_features = []

        # map country names from features to targets
        for country in feat_dict.keys():
            if country in self.targets[year].keys():
                # look for perfect matches
                mapping[country] = country
            else:
                # look for word matches
                feature_words = [x.replace(',', '') for x in country.split(' ') if x not in ignore_words]
                target_words = {x: [w.replace(',', '') for w in x.split(' ') if w not in ignore_words]
                                for x in self.targets[year].keys()}
                matches = [k for k, v in target_words.items() if (any([w in feature_words for w in v])
                                                                  or any([w in v for w in feature_words]))]
                if len(matches) > 0:
                    mapping[country] = matches[0]
                else:
                    # if no matches found, add country to missing features
                    missing_features.append(country)

        # get countries from targets with no match in features
        missing_targets = [x for x in self.targets[year].keys()
                            if x not in mapping.values()]

        return mapping, missing_targets, missing_features

    def predict_all_years(self, years=np.arange(1960, 2020, 5)):
        """
        Trains classifier for each year of data separately and plots mean cross-val score over time.

        :param years: integer years for which to train classifier
        :return:
        """

        scores = []
        num_countries = []
        num_features = []
        for year in years:
            print('\n', year)
            results = self.predict_communities(year)
            scores.append(results[0])
            num_countries.append(results[1][0])
            num_features.append(results[1][1])

        # plot cross-val score
        plt.figure()
        plt.plot(years, scores)
        plt.xlabel('Year')
        plt.ylabel('Mean cross-val score')
        plt.ylim([0, 1])
        plt.title('Cross-validation score through time')
        plt.savefig('plots/cross_val.png')

        # plot num features and countries
        plt.figure()
        plt.plot(years, num_features, label='countries')
        plt.plot(years, num_countries, label='features')
        plt.xlabel('Year')
        plt.ylabel('Number')
        plt.title('Number of countries and features through time')
        plt.savefig('plots/num_feats.png')

    def plot_confusion_matrix(self, year, X, y, classes, normalize=False):
        """
        This function plots the confusion matrix from a random forest trained on X, y data.
        Normalization can be applied by setting `normalize=True`.

        :param year: integer year of interest
        :param X: np array of data features
        :param y: np array of data targets
        :param classes: list of class names
        :param normalize: optional boolean for whether or not to normalize the matrix
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        forest = RandomForestClassifier().fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix ' + str(year))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('plots/confusion_mat_' + str(year) + '.png')

    def plot_feature_importance(self, year, X, y, feat_names):
        """
        Plots feature importance from random forest classifier and prints to txt file.

        :param year: integer year of analysis
        :param X: features to train on
        :param y: targets
        :param feat_names: list of string names of features
        :return: nothing, saves feature importance plot and text to files
        """
        forest = RandomForestClassifier().fit(X, y)

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        with open('plots/feature_ranking_' + str(year) + '.txt', 'w') as file:
            file.write("Feature ranking:")
            for f in range(len(indices)):
                file.write("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]], feat_names[f]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances " + str(year))
        plt.bar(range(len(indices)), importances[indices],
                color="b", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), indices)
        plt.tight_layout()
        plt.savefig('plots/feature_imp_' + str(year) + '.png')

    def predict_communities(self, year):
        """
        Reads in features from year's pkl file and predicts communities using random forest.
        Runs plotting of feature importance and confusion matrices.

        :param year: year for which to do predictions
        :return: 1. mean k-fold cross-validation score of classifier with k=5
                 2. shape of features matrix (num of data pts, num features)
        """
        with open('data/world_bank_' + str(year) + ' [YR' + str(year) + '].pkl', 'rb') as f:
            feats = pickle.load(f)

        name_mapping, _, miss_feat_countries = self.identify_country_name_mapping(feats, year)

        year_feats = []
        year_targs = []
        countries = feats.keys()
        for country in countries:
            if country not in miss_feat_countries:
                feat_dict = feats[country]
                feat_list = [feat_dict[key].iloc[0] for key in sorted(feat_dict.keys())]
                year_feats.append(feat_list)

                target_name = name_mapping[country]
                year_targs.append(self.targets[year][target_name])

        X = np.array(year_feats)
        y = np.array(year_targs)

        # cross-val scores
        forest = RandomForestClassifier()
        scores = cross_val_score(forest, X, y, cv=5)

        # feature importance
        self.plot_feature_importance(year, X, y, sorted(feat_dict.keys()))

        # confusion matrix
        self.plot_confusion_matrix(year, X, y, classes=range(4))

        return np.mean(scores), X.shape


def main():
    nL = NetworkLearning()
    nL.predict_all_years()


if __name__ == "__main__":
    main()
