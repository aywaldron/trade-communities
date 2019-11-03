#!/usr/bin/env

""" This file contains code for creating txt files uploadable to mapchart.net """

import pickle
import numpy as np

to_ignore = ["Middle East not specified", "Africa not specified", "Asia not specified",
             "Countries & Areas not specified", "Middle East", "South African Common Customs Area (SACCA)",
             "Western Hemisphere not specified", "Countries & Areas not specified",
             "Other Countries not included elsewhere", "European Union", "Europe not specified",
             "Euro Area"]

string = {"groups":
              {"#cc3333":
                   {"div":"#box0","label":"",
                    "paths":["France","Greece","Hungary","Portugal","Norway","Austria","Denmark","Germany","Sweden","Bulgaria","Poland","Slovakia","Czechia","Finland","Morocco","Iceland","Ireland","Israel","Turkey","Croatia","Slovenia","Bosnia_and_Herzegovina","Serbia","Kosovo","Montenegro","FYROM","Romania","Russia","Tunisia"]},
               "#66c2a4":
                   {"div":"#box1","label":"",
                    "paths":["China","Myanmar","Hong_Kong","Mauritius","Indonesia","Pakistan","Philippines","Thailand","Sri_Lanka","India","Italy","Japan","DR_Congo","Angola","Kenya","Iran","Iraq","Jordan","Mozambique","Australia","New_Zealand","South_Africa","Syria","Sudan","Tanzania","Zimbabwe","Saudi_Arabia","Egypt","Zambia","Cyprus"]},
               "#4393c3":
                   {"div":"#box3","label":"",
                    "paths":["Cameroon","United_Kingdom","Albania","Ghana","Madagascar","Nigeria","Sierra_Leone","Djibouti","French_Polynesia"]},
               "#fdb462":
                   {"div":"#box4","label":"",
                    "paths":["Guatemala","Haiti","Honduras","Mexico","Nicaragua","Panama","Paraguay","Peru","Uruguay","Venezuela","Jamaica","Colombia","Netherlands","Switzerland","United_States","Trinidad_and_Tobago","Belgium","Ethiopia","Canada","Cuba","Spain","Argentina","Bolivia","Brazil","Chile","Costa_Rica","Dominican_Republic","Suriname","Ecuador","El_Salvador"]}
               },
          "title":"","hidden":[],"borders":"#000000"}

with open('data/communities.pkl', 'rb') as f:
    loaded = pickle.load(f)

years = np.arange(1950, 2020, 5)
for year in years:
    communities = loaded[year][communities]
    for key, community in zip(string['groups'].keys(), communities.values()):
        # remove irrelevant countries
        community = [x if x not in to_ignore for x in community]

        # replace old countries with new ones
        if "Yugoslavia, SFR" in comm_set:
            s.update(["Croatia","Slovenia","Bosnia_and_Herzegovina","Serbia","Kosovo","Montenegro","FYROM"])
            s.remove("Yugoslavia")

        if "Czechoslovakia" in comm_set:
            s.update(["Slovakia", "Czechia"])
            s.remove("Czechoslovakia")

        if "Congo, Democratic Republic of" in comm_set:
            s.update(["DR_Congo"])
            s.remove("Congo, Democratic Republic of")

        if "Syrian Arab Republic" in comm_set:
            s.remove("Syrian Arab Republic")
            s.update("Syria")

        if "China, P.R.: Hong Kong" in comm_set:
            s.remove("China, P.R.: Hong Kong")
            s.update("Hong_Kong")

        if "Venezuela, Republica Bolivariana de" in comm_set:
            s.remove("Venezuela, Republica Bolivariana de")
            s.update("Venezuela")

        if "Belgium-Luxembourg" in comm_set:
            s.remove("Belgium-Luxembourg")
            s.update(["Belgium", "Luxembourg"])

        if "China, P.R.: Mainland" in comm_set:
            s.remove("China, P.R.: Mainland")
            s.update("China")

        if "French Territories: French Polynesia" in comm_set:
            s.remove("French Territories: French Polynesia")
            s.update("French_Polynesia")

        if "U.S.S.R." in comm_set:
            s.remove("U.S.S.R")
            s.update("Russia")

        # replace spaces with underscores
        comm_set = set([x.replace(' ', '_') if ' ' in x else x for x in community])

        string["groups"][key] = community


