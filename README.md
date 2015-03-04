Learning Distance Metrics
=========================
*learning_dist_metrics* is a Python module to implementate the supervised learning distance metrics algorithm introduced in 
the [paper](http://ai.stanford.edu/~ang/papers/nips02-metric.pdf), which is co-authored by [Eric P. Xing](), [Andrew Y. Ng](), [Micheal I. Jordan]() and [Start Russell](). To deliver a consistent programming style, the algorithm is developed by following [scikit-learn](http://orbi.ulg.ac.be/bitstream/2268/154357/1/paper.pdf) API design protocols.

### Algorithm details:

### Modified Objective Functions:


How to Install: 
===============
The algorithm is still in packaging process. To use it, download the module from the repository and include the folder under the root directory of your application.


How to use it:
==============
```
from learning_dist_metrics.datasets import load_data
from laerning_dist_metrics.ldm import LDM


sample_data = load_data.load_sample_data()

ldm = LDM() 

ldm.fit(sample_data.data, sample_data.sim_pairs, sample_data.diff_pairs)
```