#!/usr/bin/env

""" This file contains tests for the dot_network.py file that creates DOT networks """

import pandas as pd
import dot_network as dot
import networkx as nx
import pickle


class test_dot_network:

    def setup(self):
        """ Setup method creates the test csv file and writes to data/test_DOT_files.csv """
        # extract test data from DOT data frame
        df = pd.read_csv('data/DOT_timeSeries.csv')
        test_df = df.loc[(df['Country Name'] == 'Angola') & (df['Counterpart Country Name'] == 'Colombia')]
        test_df = test_df.append(df.loc[(df['Country Name'] == 'Angola') &
                                        (df['Counterpart Country Name'] == 'Moldova')])
        test_df = test_df.append(df.loc[(df['Country Name'] == 'Moldova') &
                                        (df['Counterpart Country Name'] == 'Angola')])
        test_df = test_df.append(df.loc[(df['Country Name'] == 'World') &
                                        (df['Counterpart Country Name'] == 'Moldova')])

        self.filename = 'data/test_DOT_file.csv'
        self.test_data = test_df
        self.filtered_data = dot.prepare_data(self.filename)
        self.code_dict = dot.create_country_code_dict(self.filtered_data)
        self.network = dot.create_dot_network(self.filtered_data, '2007')
        self.network_dict = dot.create_network_dict(self.filtered_data, range(2007, 2010))

        # save test dataframe to file
        test_df.to_csv(self.filename)

    def test_extract_relevant_rows(self):
        """ Tests that extract_relevant_rows only returns the relevant rows """
        df = dot.extract_relevant_rows(self.test_data,
                                       column_name='Country Name',
                                       column_value='Angola')
        assert (df['Country Name'] == 'Angola').all()

    def test_extract_relevant_rows_not_equal(self):
        """ Tests that extract_relevant_rows filters out undesired rows when not_equal=True"""
        df = dot.extract_relevant_rows(self.test_data,
                                       column_name='Country Name',
                                       column_value='Angola',
                                       not_equal=True)
        assert not (df['Country Name'] == 'Angola').any()

    def test_prepare_data(self):
        """ Tests the prepare_data function by asserting that only relevant rows are returned """
        df = self.filtered_data
        assert len(df) == 3
        assert ((df['Country Name'] == 'Angola') & (df['Counterpart Country Name'] == 'Colombia')).any()
        assert ((df['Country Name'] == 'Angola') & (df['Counterpart Country Name'] == 'Moldova')).any()
        assert ((df['Country Name'] == 'Moldova') & (df['Counterpart Country Name'] == 'Angola')).any()

    def test_create_dot_network(self):
        """ Tests that create_dot_network returns the correct network """
        assert list(self.network.edges.data()) == [(614.0, 233.0, {'weight': 73172520.0}),
                                        (921.0, 614.0, {'weight': 263001.0})]

    def test_create_network_dict(self):
        """ Tests that create_network_dict returns a dictionary of networks """
        assert [type(self.network_dict[year]) == nx.DiGraph for year in range(2007, 2010)]

    def test_code_dict_creation(self):
        """ Tests that code dict created is correct """
        assert self.code_dict == {921: 'Moldova', 614: 'Angola', 233: 'Colombia'}

    def test_community_finding(self):
        """ Tests that network community finding function is working properly """
        comm, mod = dot.find_and_print_network_communities(self.network, code_dict=self.code_dict)
        assert comm == {0: ['Colombia', 'Angola', 'Moldova']}
        assert mod == 0.0

    def test_network_info_saving(self):
        """ Tests that network community info is correctly saved to file """
        dot.save_all_community_information(self.network_dict, code_dict=self.code_dict, filename='data/test.pkl')
        with open('data/test.pkl', 'rb') as f:
            loaded = pickle.load(f)

        assert loaded == {2008: {'communities': {0: ['Moldova', 'Angola']},
                                 'Average in degree': '1.0000',
                                 'Type': 'DiGraph',
                                 'Number of edges': '2',
                                 'Name': '',
                                 'Number of nodes': '2',
                                 'Average out degree': '1.0000',
                                 'modularity': 0.0},
                          2009: {'communities': {0: ['Moldova', 'Angola']},
                                 'Average in degree': '0.5000',
                                 'Type': 'DiGraph',
                                 'Number of edges': '1',
                                 'Name': '',
                                 'Number of nodes': '2',
                                 'Average out degree': '0.5000',
                                 'modularity': 0.0},
                          2007: {'communities': {0: ['Colombia', 'Angola', 'Moldova']},
                                 'Average in degree': '0.6667',
                                 'Type': 'DiGraph',
                                 'Number of edges': '2',
                                 'Name': '',
                                 'Number of nodes': '3',
                                 'Average out degree': '0.6667',
                                 'modularity': 0.0}}
