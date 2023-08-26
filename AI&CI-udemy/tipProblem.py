
'''
Antecedents (inputs) - service, food quality
Consequents (outputs) - tip
Rules:
    - IF the service was good or the food quality was good, THEN the tip will be high.
    - IF the service was average, THEN the tip will be medium.
    - If the service was poor and the food quality was poor THEN the tip will be low.
Usage
''' 
'''
service = 9.8
quality = 6.5

How much would we recommend the tip?

'''

# Importing the libraries
import numpy as np
import skfuzzy as fuzz

# Define the variables
x_qual = np.arange(0,11,1)
x_serv = np.arange(0,11,1)
x_tip = np.arange(0,26,1)

# Membership functions
qual_lo = fuzz.trimf(x_qual, [0,0,5])
qual_md = fuzz.trimf(x_qual, [0,5,10])
qual_hi = fuzz.trimf(x_qual, [5,10,10])

serv_lo = fuzz.trimf(x_serv, [0,0,5])
serv_md = fuzz.trimf(x_serv, [0,5,5])
serv_hi = fuzz.trimf(x_serv, [0,10,10])

tip_lo = fuzz.trimf(x_tip, [0,0,13])
tip_md = fuzz.trimf(x_tip, [0,13,25])
tip_hi = fuzz.trimf(x_tip, [13,25,25])


'''

Define the fuzzy relationship between input and output variables by defining the RULES:
1. IF the food is bad OR the service is poor,
2. IF the service is acceptable, then the tip will be medium.
3. IF the food is great OR the service is amazing then the tip will be high

'''
# RULES:

qual_level_lo = fuzz.interp_membership(x_qual, qual_lo, 6.5)
qual_level_md = fuzz.interp_membership(x_qual, qual_md, 6.5)
qual_level_hi = fuzz.interp_membership(x_qual, qual_hi, 6.5)

serv_level_lo = fuzz.interp_membership(x_serv, serv_lo, 9.8)
serv_level_md= fuzz.interp_membership(x_serv, serv_md, 9.8)
serv_level_hi = fuzz.interp_membership(x_serv, serv_hi, 9.8)

# Rule 1
active_rule1 = np.fmax(qual_level_lo, serv_level_lo)
tip_activation_lo = np.fmin(active_rule1, tip_lo)

# Rule 2
tip_activation_md = np.fmin(serv_level_md, tip_md)

# Rule 3
active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, tip_hi)

# Defuzzification
aggregated = np.fmax(tip_activation_lo, np.fmax(tip_activation_md, tip_activation_hi))

tip = fuzz.defuzz(x_tip, aggregated, 'centroid')

print(tip)

# END

