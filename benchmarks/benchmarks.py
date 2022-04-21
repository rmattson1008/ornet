import os
import numpy as np
import pickle

# for now just limiting by the final fricking representation files... 29,31,54.  

# before working on this doc:
# save models (in workspace ig) -> drivers
# hook representations, save in wkspace ig. 

# load cnn feature dicts, cnn+roi feature dicts
# load cnn+lstm feature dicts (exactly where should we hook them?)
# load intermediate feature dicts

# Figure out how comparable they are
# figure out what we doing with intermediates -> same as Neelima?? 

# Run a few classification models on all. like svm, random forest, etc
# delany should do this. 

# get deep features 
with open('/home/rachel/representations/cnn/BaseCnn_embeddings10.pkl', 'rb') as f:
            cnn_feature_dict = pickle.load(f)

# print(feature_dict.keys())

control_features = cnn_feature_dict["control"]
mdivi_features = cnn_feature_dict["mdivi"]
llo_features = cnn_feature_dict["llo"]



to_data = "/data/ornet/gmm_intermediates"

llos = []
mdivis = []
controls = []

path = os.path.join(to_data, "llo")
for file in os.listdir(path):
    if 'normalized' in file:
        name = os.path.join(path, file)
        llos.append(np.load(name))

path = os.path.join(to_data, "mdivi")
for file in os.listdir(path):
    if 'normalized' in file:
        name = os.path.join(path, file)
        llos.append(np.load(name))

path = os.path.join(to_data, "control")
for file in os.listdir(path):
    if 'normalized' in file:
        name = os.path.join(path, file)
        llos.append(np.load(name))

# Neelima took some averages... this format might be gross for that
# check work on local
# we can easily change cnn final decision layer size