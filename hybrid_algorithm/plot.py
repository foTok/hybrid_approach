"""
plot figures
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from hybrid_algorithm.utilities import plot_bar

#accuracy
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

#time consuming
mean3 = np.array([[3.84300334,0.61936619,0.74386624,1.15460043,1.33133282],\
[1.4617041239343882 + 3.84300334,1.4499365234945687+0.61936619,1.393627318075585+0.74386624,1.3627067191584157+1.15460043,1.3445167731641323+1.33133282]])

ylabel3 = "Time/s"

plot_bar(mean1, label, ylim=(0.4, 1.15), xlabel=xlabel, ylabel=ylabel, xticks=xticks)

plot_bar(mean2, label, xlabel=xlabel, ylabel=ylabel, xticks=xticks)

plot_bar(mean3, label, xlabel=xlabel, ylabel=ylabel3, xticks=xticks)
