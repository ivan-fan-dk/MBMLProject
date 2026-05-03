# MBMLProject
CM -> LCCM -> GP-LCCM 

Description of swissmetro dataset: https://transp-or.epfl.ch/biogeme-2.5/swissmetro.pdf

They interviewed regular train and car travelers and asked them to choose between their current mode of transport and the hypothetical Swissmetro in various price and travel time scenarios.

All respondents have train and SW availability, and 84.3% have car availability. 
Alternatives: Respondents chose between three modes: Train, Car, and Swissmetro.
Size: It typically contains around 10,728 observations from about 1,192 individuals.


From the last run of gp_lccm.py: 
Best number of latent classes:

Best by prediction_LL (primary predictive metric): K=9
Best by test_accuracy: K=5
Interpretation of the best predictive model (K=9):

The segmentation is highly heterogeneous, with a few large classes and several very small classes.
Largest share class is Class 7 (about 0.41), car-preferring (positive ASC Car), and strongly sensitive to both time and cost.
Class 3 (about 0.25) is very time-sensitive but has near-zero cost coefficient, so its VOT is extremely large and unstable.
Several tiny-share classes (for example around 0.00 to 0.03 share) indicate possible over-segmentation for interpretability, even though prediction_LL is highest.
Practically:
If you optimize pure predictive LL, choose K=9.
If you want stronger behavioral interpretability/stability with best accuracy, K=5 is a very defensible choice.
If you want, I can add one more auto-export file that picks a recommended K using a combined rule (for example highest prediction_LL among models with no tiny-share classes below 3%).