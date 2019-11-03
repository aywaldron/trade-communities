#!/usr/bin/env

""" This file contains code for reading in World Bank data files and saving to dictionary in pkl file """

import pickle
import pandas as pd
import numpy as np


def read_in_world_bank_data():
    """
    Reads data from file into dictionary organized by country, then year, then series name.

    :return: heirarchical dictionary with year -> country -> series name as keys
    """

    # read in datafiles
    df = pd.read_csv('data/World_Development_Indicators_Data.csv')

    # get list of all years available
    years = [col for col in df if (col.startswith('19') or col.startswith('20'))
             and int(col[:4]) in np.arange(2015, 2020, 5)]

    # make dictionary for this file's data
    data_dict = {}
    # organize data by year, then country, then series name
    for year in years:

        # initialize year dictionary with integer year
        int_year = int(year[:4])
        data_dict[int_year] = {}

        countries, features = get_good_data(df, year)
        print(year)
        print('\n%d countries, %d features' % (len(countries), len(features)))

        for country in countries:
            # skip irrelevant results
            if type(country) == float or 'Data from ' in country or 'Last updated' in country:
                continue

            # create dict for country's data
            country_dict = {}
            country_df = df.loc[df['Country Name'] == country]

            # skip countries with any missing values
            for feat in features:
                row = country_df.loc[country_df['Series Name'] == feat]
                country_dict[feat] = row[year]

            # update file_dict with data for country
            data_dict[int_year][country] = country_dict

        # save data dictionary to file
        with open('data/world_bank_' + str(year) + '.pkl', 'wb') as f:
            pickle.dump(data_dict[int_year], f, pickle.HIGHEST_PROTOCOL)


def which_countries_have_feat(df, feat, year):
    """
    Returns percentage of countries having feature for specified year and which countries have it.

    :param df: pandas dataframe of data
    :param feat: feature series name
    :param year: year in question (string column name)
    :return: 1. float percentage of countries having feature
             2. set of country names having feature
    """

    countries = df['Country Name'].unique()
    num_countries = len(countries)

    good_countries = set()
    for country in countries:
        row = df.loc[(df['Country Name'] == country) & (df['Series Name'] == feat)]
        if not row.empty and row[year].iloc[0] != '..':
            # no missing data
            good_countries.add(country)

    return len(good_countries) / num_countries, good_countries


def get_good_data(df, year):
    """
    Returns set of countries having all features in features dict also returned.

    :param df: pandas dataframe containing data
    :param year: year in question (string column name)
    :return: 1. set of country names
             2. set of feature series names
    """

    unique_feats = df['Series Name'].unique()

    percents = {y:0.5 for y in list(range(1960, 1967))}
    percents.update({y:0.6 for y in list(range(1967, 1977))})
    percents.update({y:0.7 for y in list(range(1977, 1987))})
    percents.update({y:0.8 for y in list(range(1987, 1997))})
    percents.update({y:0.85 for y in list(range(1997, 2007))})
    percents.update({y:0.9 for y in list(range(2007, 2017))})

    all_good_countries = {}
    for feat in unique_feats:
        percent, countries = which_countries_have_feat(df, feat, year)
        if percent > percents[int(year[:4])]:
            all_good_countries[feat] = countries

    # choose a random set of countries to initialize set intersection
    _, country_set = all_good_countries.popitem()
    # find countries that have all good features
    intersection = country_set.intersection(*[v for v in all_good_countries.values()])

    return intersection, all_good_countries.keys()


if __name__ == "__main__":
    read_in_world_bank_data()