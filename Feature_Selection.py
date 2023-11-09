from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np

CO2_df = pd.read_csv('rearranged_CO2_data.csv')

def joint_entropy_nd(args, bins):
    
    binned_dist, _ = np.histogramdd(args, bins=bins)

    
    probs = binned_dist / np.sum(binned_dist)

    
    probs = probs[np.nonzero(probs)]

    
    joint_entropy = -np.sum(probs * np.log2(probs))

    return joint_entropy

your_bin_value = 11


input_features = np.array(CO2_df.iloc[:, :-1].values.T)  
target_variable = CO2_df.iloc[:, -1].values


S = set()

for i in range(len(input_features)):
    current_feature = input_features[i]

    args_to_check_1 = [input_features[j] for j in S] + [current_feature, target_variable]
    args_to_check_2 = [input_features[j] for j in S] + [current_feature]
    args_to_check_3 = [input_features[j] for j in S] + [target_variable]
    args_to_check_4 = [input_features[j] for j in S]

    result_1 = joint_entropy_nd(args_to_check_1, bins=your_bin_value) - joint_entropy_nd(args_to_check_2, bins=your_bin_value)

    if i == 0:
        result_2 = joint_entropy_nd(args_to_check_3, bins=your_bin_value)
    else:
        result_2 = joint_entropy_nd(args_to_check_3, bins=your_bin_value) - joint_entropy_nd(args_to_check_4, bins=your_bin_value)

    result = result_2 - result_1
    print(result)

    if result != 0:
        S.add(i) 

print(S)