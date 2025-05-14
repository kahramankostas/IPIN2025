<center><h1>Huawei University Challenge Competition 2021</h1></center> 
<center><h1>Data Science for Indoor positioning</h1></center> 

## 2.2 Full Mall Graph Clustering

### Train

The sample training data for this problem is a set of 106981 fingerprints (`task2_train_fingerprints.json`) and some edges between them. We have provided files that indicate three different edge types, all of which should be treated differently. 

`task2_train_steps.csv` indicates edges that connect subsequent steps within a trajectory. These edges should be highly trusted as they indicate a certainty that two fingerprints were recorded from the same floor.

`task2_train_elevations.csv` indicate the opposite of the steps. These elevations indicate that the fingerprints are almost definitely from a different floor. You can thus extrapolate that if fingerprint $N$ from trajectory $n$ is on a different floor to fingerprint $M$ from trajectory $m$, then all other fingerprints in both trajectories $m$ and $n$ must also be on seperate floors.

`task2_train_estimated_wifi_distances.csv` are the pre-computed distances that we have calculated using our own distance metric. This metric is imperfect and as such we know that many of these edges will be incorrect (i.e. they will connect two floors together). We suggest that initially you use the edges in this file to construct your initial graph and compute some solution. However, if you get a high score on task1 then you might consider computing your own wifi distances to build a graph.

Your graph can be at one of two levels of detail, either trajectory level or fingerprint level, you can choose what representation you want to use, but ultimately we want to know the **trajectory clusters**. Trajectory level would have every node as a trajectory and edges between nodes would occur if fingerprints in their trajectories had high similiraty. Fingerprint level would have each fingerprint as a node. You can lookup the trajectory id of the fingerprint using the `task2_train_lookup.json` to convert between representations. 

To help you debug and train your solution we have provided a ground truth for some of the trajectories in `task2_train_GT.json`. In this file the keys are the trajectory ids (the same as in `task2_train_lookup.json`) and the values are the real floor id of the building.

### Test

The test set is the exact same format as the training set (for a seperate building, we weren't going to make it that easy ;) ) but we haven't included the equivalent ground truth file. This will be withheld to allow us to score your solution.

Points to consider
- When doing this on real data we do not know the exact number of floors to expect, so your model will need to decide this for itself as well. For this data, do not expect to find more than 20 floors or less than 3 floors.
- Sometimes in balcony areas the similarity between fingerprints on different floors can be deceivingly high. In these cases it may be wise to try to rely on the graph information rather than the individual similarity (e.g. what is the similarity of the other neighbour nodes to this candidate other-floor node?)
- To the best of our knowledge there are no outlier fingerprints in the data that do not belong to the building. Every fingerprint belongs to a floor



## 2.3 Loading the data

In this section we will provide some example code to open the files and construct both types of graph.


```python
import os
import json
import csv
import networkx as nx
from tqdm import tqdm

path_to_data = "task2_for_participants/train"

with open(os.path.join(path_to_data,"task2_train_estimated_wifi_distances.csv")) as f:
    wifi = []
    reader = csv.DictReader(f)
    for line in tqdm(reader):
        wifi.append([line['id1'],line['id2'],float(line['estimated_distance'])])
        
with open(os.path.join(path_to_data,"task2_train_elevations.csv")) as f:
    elevs = []
    reader = csv.DictReader(f)
    for line in tqdm(reader):
        elevs.append([line['id1'],line['id2']])        

with open(os.path.join(path_to_data,"task2_train_steps.csv")) as f:
    steps = []
    reader = csv.DictReader(f)
    for line in tqdm(reader):
        steps.append([line['id1'],line['id2'],float(line['displacement'])]) 
        
fp_lookup_path = os.path.join(path_to_data,"task2_train_lookup.json")
gt_path = os.path.join(path_to_data,"task2_train_GT.json")

with open(fp_lookup_path) as f:
    fp_lookup = json.load(f)

with open(gt_path) as f:
    gt = json.load(f)
   
```

### Fingerprint graph
This is one way to construct the fingerprint-level graph, where each node in the graph is a fingerprint. We have added edge weights that correspond to the estimated/true distances from the wifi and pdr edges respectively. We have also added elevation edges to indicate this relationship. You might want to explicitly enforce that there are *none* of these edges (or any valid elevation edge between trajectories) when developing your solution. 


```python
G = nx.Graph()

for id1,id2,dist in tqdm(steps):
    G.add_edge(id1, id2, ty = "s", weight=dist)
    
for id1,id2,dist in tqdm(wifi):
    G.add_edge(id1, id2, ty = "w", weight=dist)
    
for id1,id2 in tqdm(elevs):
    G.add_edge(id1, id2, ty = "e")
```

### Trajectory graph
The trajectory graph is arguably not as simple as you need to think of a way to represent many wifi connections between trajectories. In the example graph below we just take the mean distance as a weight, but is this really the best representation?


```python
B = nx.Graph()

# Get all the trajectory ids from the lookup
valid_nodes = set(fp_lookup.values())

for node in valid_nodes:
    B.add_node(node)

# Either add an edge or append the distance to the edge data
for id1,id2,dist in tqdm(wifi):
    if not B.has_edge(fp_lookup[str(id1)], fp_lookup[str(id2)]):
        
        B.add_edge(fp_lookup[str(id1)], 
                   fp_lookup[str(id2)], 
                   ty = "w", weight=[dist])
    else:
        B[fp_lookup[str(id1)]][fp_lookup[str(id2)]]['weight'].append(dist)
        
# Compute the mean edge weight
for edge in B.edges(data=True):
    B[edge[0]][edge[1]]['weight'] = sum(B[edge[0]][edge[1]]['weight'])/len(B[edge[0]][edge[1]]['weight'])
        
# If you have made a wifi connection between trajectories with an elev, delete the edge
for id1,id2 in tqdm(elevs):
    if B.has_edge(fp_lookup[str(id1)], fp_lookup[str(id2)]):
        B.remove_edge(fp_lookup[str(id1)], 
                      fp_lookup[str(id2)])
```
