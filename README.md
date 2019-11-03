# trade-communities
Predicting international trade communities from world development indicators

## The problem

This project aimed to determine the predictive power of world
development indicators for international trade communities, while inspecting changes and trends in
these communities over time. It was hypothesized that world development indicators may contain the
right balance of both sociologically and economically-driven information to predict international trade
communities.

In order to test this hypothesis, a weighted, directed network was created using export FOB data
from the International Monetary Fund’s (IMF) Direction of Trade (DOT) statistics. Communities were
found in this network using the modularity maximization algorithm explained in [1]. These community
classifications were then predicted from the World Bank’s World Development Indicators using a
random forest classifier. 

## References

1. Leicht EA, Newman ME. “Community structure in directed networks”. Phys Rev Lett. 2008
Mar 21; 100(11):118703. Epub 2008 Mar 21.
