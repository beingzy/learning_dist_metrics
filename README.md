Learning Distance Metrics
=========================
*learning_dist_metrics* is a Python module to implement the supervised learning distance metrics algorithm introduced in 
the [publication](http://ai.stanford.edu/~ang/papers/nips02-metric.pdf), which is co-authored by [Eric P. Xing](), [Andrew Y. Ng](), [Micheal I. Jordan]() and [Start Russell](). To deliver a consistent programming style, the algorithm is developed by following [scikit-learn](http://orbi.ulg.ac.be/bitstream/2268/154357/1/paper.pdf) API design protocols.

The learning distance metrics learns weights of features of subjects by maximizing the distances between subjects from different class and minimizing the ditances between subjects of a same class. The implemented version makes assumption on the transformation matrix, **A**, is diagonal. In applications, a user has to provide the subject profile data, a list of pairs considered same and a list of pairs considered different. If the list of different classes has not been provided, the algorithm will consrtuct the list by including all of the possible pairs which are formed by subjects mentioned in the profile data and not listed in the same class list. As a result, the learned transformation matrix, **A**, will be produced.

* Distnace Definion:
```math
d(x, y) = d_A(x, y) = ||x - y||_A = \sqrt{(x - y)^{T} A (x - y)}
```
* Objective function for Minimization:
```math
min(A) = \frac{sum_{(x, y) \in S}d(x, y)}{sum_{(x, y) \in D}d(x, y)}
```
Where, **S** is the set of subject pairs considered coming from a same class. **D** is the set of subject pairs considered from a different class.



How to Install: 
===============
The algorithm is still in packaging process. To use it, download the module from the repository and include the folder under the root directory of your application.


How to use it:
==============
```python
from learning_dist_metrics.datasets import load_data
from laerning_dist_metrics.ldm import LDM


sample_data = load_data.load_sample_data()

ldm = LDM() 

ldm.fit(sample_data.data, sample_data.sim_pairs, sample_data.diff_pairs)
```
![3D Scatterplots of 2 Clusters in the original space](/images/2clusters_3d_origin.png)
![3D Scatterplots of 2 Clusters in the transformed Space](/images/2clusters_3d_fitted.png)