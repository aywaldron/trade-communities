#!/usr/bin/env

""" This file contains code for reading Direction of Trade data from the IMF into a weighted, directed network
 and using it to find trade communities """

import networkx as nx
import pandas as pd
import math
from modularity_maximization import partition
from modularity_maximization.utils import get_modularity
import pickle


def create_network_dict(df, years):
    """
    Returns a dictionary of networks with the relevant years as keys.

    :param df: pandas dataframe of exports to use in creating networks
    :param years: iterable of integer years for which to create networks
    :return: dictionary of networkx graphs with integer years as keys
    """

    networks = {}
    for year in years:
        print('Creating network for %d...' % year)
        networks[year] = create_dot_network(df, str(year))

    return networks


def create_dot_network(df, year):
    """
    Returns networkx directed graph of international trade with country codes as nodes.

    :param df: pandas dataframe of trade exports with each row representing trade between 2 countries and columns for
               each year of data
    :param year: string year to create network for
    :return graph: newtorkx directed graph with exports in USD as edge weights
    """

    # extract only relevant data from dataframe
    data = df[['Country Code', 'Counterpart Country Code', year]]

    # initialize networkx DiGraph
    network = nx.DiGraph()
    # add edge to graph for each row of data not equal to NaN
    for index, row in data.iterrows():
        if not math.isnan(float(row[year])):
            network.add_edge(int(row['Country Code']), int(row['Counterpart Country Code']),
                             weight=row[year])

    return network


def extract_relevant_rows(df, column_name, column_value, not_equal=False):
    """
    Returns pandas dataframe consisting only of rows with specific values in a specific column.

    :param df: pandas dataframe to extract rows from
    :param column_name: name of column requiring specific value
    :param column_value: value required in column
    :param not_equal: boolean for whether to return rows equal to passed in values (False) or not
                      equal to passed in values (True)
    :return: pandas dataframe consisting only of desired rows
    """

    if not_equal:
        return df.loc[df[column_name] != column_value]

    return df.loc[df[column_name] == column_value]


def prepare_data(filename='data/DOT_timeSeries.csv'):
    """
    Reads in DOT datafile and filters for relevant information.

    :param filename: string path to csv file
    :return: pandas dataframe constructed from datafile and filtered for relevant rows
    """

    # read data file into pandas dataframe
    df = pd.read_csv(filename)

    # extract unwanted 'countries' from dataframe
    countries = ['Europe', 'Emerging and Developing Europe', 'Emerging and Developing Asia',
                 'Middle East, North Africa, and Pakistan', 'Export earnings: nonfuel',
                 'Sub-Saharan Africa', 'Export earnings: fuel', 'Western Hemisphere',
                 'World', 'Special Categories', 'Advanced Economies', 'CIS',
                 'Emerging and Developing Economies']
    for country in countries:
        df = extract_relevant_rows(df, column_name='Country Name', column_value=country, not_equal=True)
        df = extract_relevant_rows(df, column_name='Counterpart Country Name', column_value=country, not_equal=True)

    # extract exports only from data
    exports = extract_relevant_rows(df, column_name='Indicator Code', column_value='TXG_FOB_USD')
    # extract value attributes only from exports
    export_values = extract_relevant_rows(exports, column_name='Attribute', column_value='Value')

    return export_values


def create_country_code_dict(df):
    """
    Creates a dictionary of country names with country codes as keys from the passed in dataframe.

    :param df: pandas dataframe from which to extract country codes & names
    :return: dictionary with country codes as keys and country names as values
    """

    code_dict = {}

    # check both country and counterpart country columns for unique country codes
    for col in ['Country', 'Counterpart Country']:
        for code in df[col + ' Code'].unique():
            code_dict[int(code)] = df.loc[df[col + ' Code'] == code][col + ' Name'].values[0]

    return code_dict


def find_and_print_network_communities(G, code_dict=None):
    """
    Finds network communities through modularity maximization and returns dictionary of community
    members by country name with community numbers as keys.

    :param G: networkx Graph to find communities in
    :param code_dict: dictionary mapping country codes to names - if passed in, will use mappings for
                      recording community members
    :return: 1. dictionary with community numbers as keys and list of string country names as values
             2. modularity of discovered community partitions
    """

    comm_dict = partition(G)

    comm_members = {}
    for comm in set(comm_dict.values()):
        countries = [node for node in comm_dict if comm_dict[node] == comm]
        if code_dict is not None:
            countries = [code_dict[code] for code in countries]

        comm_members[comm] = countries

    return comm_members, get_modularity(G, comm_dict)


def get_network_info_dict(network):
    """
    Returns dictionary of network characteristics obtained from networkx.info method.

    :param network: network to get info on
    :return: dictionary mapping network characteristic name to value
    """
    info_str = nx.info(network)
    lines = info_str.split('\n')

    info_dict = {}
    for line in lines:
        pair = line.split(':')
        info_dict[pair[0]] = pair[1].strip()

    return info_dict


def save_all_community_information(networks, code_dict=None, filename='data/communities.pkl'):
    """
    Finds communities in each network and saves modularity, network info, and community members to file.

    :param networks: dictionary mapping integer years to networks
    :param code_dict: dictionary mapping country codes to names - if passed in, will use mappings for
                      recording community members
    :param filename: string name, including extension, of file to save info to
    :return: nothing, saves network info to 'communities.pkl'
    """

    save_dict = {}
    for year, network in networks.items():
        print('Finding communities for %d network...' % year)
        comms, mod = find_and_print_network_communities(network, code_dict)
        info_dict = get_network_info_dict(network)
        comm_dict = {'modularity': mod,
                     'communities': comms}
        save_dict[year] = {**info_dict, **comm_dict}

    with open(filename, 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)


def main():
    # clean data & create country code dictionary
    data = prepare_data()
    country_dict = create_country_code_dict(data)

    # create dictionary of networks with keys as years
    networks = create_network_dict(data, years=range(1948, 2018))

    # save community info for all networks
    save_all_community_information(networks, code_dict=country_dict)


if __name__ == "__main__":
    main()
