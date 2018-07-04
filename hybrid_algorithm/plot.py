"""
plot figures
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from hybrid_algorithm.utilities import plot_bar
#labels
label = ["Without RV", "With RV"]
#xticks
xticks = ["CNN", "IGCNN", "IGSCNN", "HIGSCNN", "HIGSECNN"]
#xlabel
xlabel = "Classifier"
#ylabel
ylabel = "Accuracy"
#accuracy for single fault
mean1 = np.array([[0.774509804,0.797919168,0.826130452,0.849939976,0.951380552],[0.917166867,0.87555022,0.911564626,0.928171269,0.988795518]])

#accuracy for multiple-fault (2-fault)
mean2 = np.array([[0.361,0.3144,0.5302,0.5366,0.8584],[0.475,0.455,0.574,0.5632,0.8808]])

plot_bar(mean1, label, xlabel=xlabel, ylabel=ylabel, xticks=xticks)

plot_bar(mean2, label, xlabel=xlabel, ylabel=ylabel, xticks=xticks)
