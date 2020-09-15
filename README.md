The repository provides basic formulae and a sample code to instantiate "Privacy at risk" for the Laplace mechanism.
The details are part of the paper in PoPETS 2021. The preprint of the paper can be found at: https://arxiv.org/abs/2003.0097.

# Privacy-at-risk

Illustratiions in the Section 6 of the paper are made using following files:
  1. formule.py: It contains formulae for privacy at risk for the Laplace mechanism.
  2. composition.py: It contains composition theorem with privacy at risk.
  3. new_expt.py: It contains the simulation for the illustration in the Section 6 of the paper. 
  
The illustration makes use of a cleaned sample from US 2000 dataset (https://usa.ipums.org/usa/). We have removed the original IDs from the dataset for the purpose of anonymisation.

# Comparison with PATE

PATE folder contains the code that implements Private Aggregation of Teacher Ensemble mode proposed by Papernot et al. (https://arxiv.org/abs/1610.05755). It contains three files.
  1. teachers.py: It trains the ensemble of teachers on MNIST data. Currently, it implements teachers using logistic regression as the classifier.
  2. student.py: It builds a synthetic training dataset for training the student model using the trained ensemble of teachers.
  3. analysis.py: It uses the moment accountant to compute the data-dependent privacy level for the ensemble of teachers. We incorporate Privacy at Risk in the computation of the moment accountant.
