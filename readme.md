# Masters Project

Rawser Spicer

Code and work related to my master project relate to landscape Changes
in permafrost regions. More specifically using Random Forests to 
train thermokarst initiation models.

## notes on model/ Random Forest naming
* **OM:** this refers to the original model created using 
find_initiation_areas_vpdm in initiation_ares.py
* **RF_e[N]\_rs[M]\_mln[Q]\_...\_v[z]:** Refers to the various iteration of the random forest model. An example is **RF_e10_rs_42_mln100_v1**. Each of the naming descriptors, except v which refers to the version, correspond to a sckit-learn Random Forest parameter (see below).
  * *e[N]:* refers to the num_estimators.
  * *rs[N]:* refers to the random_sate.
  * *mln[Q]:* refers to the max_lean_nodes.
  * *msl[R]:* refers to the min_sample_leaf.
  * *mss[S]:* refers to the min_sample_split.
  * *mf[T]:* refers to the max_features.
  * *V[Z]:* version of model if trained multiple times





## Repo Directory
* **forestpy:** Tool for training random forest models with varying parameters, 
and ranking them by how closely they match the original model. 
* **notebooks:** Jupyter-notebooks
* **notes:** notes on meeting and other things
* **presentations:** presentations given on project.
* **research:** notes on research.
* **scripts:** scripts used that are not part of any particular tool 
* **writing:** write up paper and other related writings



