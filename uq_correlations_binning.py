import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns


save_dir = "/data/diag/leahheil/saved/polish/"

# feature = "FC_thickness"
feature = "lipid_arc"

for uncertainty_type in ["relative"]:#, "absolute"]:

    if uncertainty_type == "absolute":
        info = pd.read_csv(save_dir + "metrics_mc10_info_polish_abs.csv") 
    else:
        info = pd.read_csv(save_dir + "metrics_mc10_info_polish_test.csv") 
    info_lipid = info[(info['lipid_present_y'] == 1) & (info['lipid_present_x'] == 1)] 
    info_calcium = info[(info['calcium_present_y'] == 1) & (info['calcium_present_x'] == 1)] 

    for metric in ['lipid_en', 'lipid_MI', 'lipid_brier', 'total_en', 'total_MI', 'total_brier', 'total_en_raw', 'total_MI_raw', 'total_brier_raw']:
            print(metric)
    
            # first is for relative difference, second for absolute difference
            if uncertainty_type == "absolute":
                arc_groups = [0, 15, 35, 60, 360]
            else:
                arc_groups = [0, 0.4, 0.8, 1.2, 1.6] # arc
                # arc_groups = [0, 0.2, 0.4, 0.7, 2] # FCT
                
            arc_groups_dict = {}

            # plt.figure(figsize=(4,4))
            # plt.hist(info_lipid[metric])
            # plt.savefig(save_dir + f"histogram_{metric}.png", dpi=300, bbox_inches='tight')
            # plt.close()

            for i in range(len(arc_groups)-1):
                arc_groups_dict[i] = info_lipid[(info_lipid[feature] > arc_groups[i]) & (info_lipid[feature] < arc_groups[i+1])][metric].values
            
            df = pd.DataFrame(columns=["group", "metric"])

            for key, value in arc_groups_dict.items():
                group_df = pd.DataFrame({'Group': [str(key) + f" (n = {round(len(value), 3)})"]*len(value),
                                        'Metric': value})
                print(key, len(group_df))
                df = pd.concat([df, group_df], ignore_index = True)

            plt.figure(figsize=(8 * 1.2, 4))
            sns.violinplot(x='Group', y='Metric', data=df, color='tab:blue', cut=0,
                        linewidth=1, legend=False)
            
            stat, p_value = kruskal(arc_groups_dict[0], arc_groups_dict[1], arc_groups_dict[2], arc_groups_dict[3])
            print(stat, p_value)


            # for i in range(len(arc_groups_dict)):
            #     for j in range(len(arc_groups_dict)):
            #         if i < j:
            #             stat, p_value = mannwhitneyu(arc_groups_dict[i], arc_groups_dict[j])
            #             significance = p_value <= 0.05
                        
                        # if significance:
                        #     print(i, j, stat, p_value)
                        #     y_pos = max(df[(df['group'] == i) | (df['group'] == j)]['metric']) * (1 + 0.06*i)
                        #     plt.plot([i, j], [y_pos, y_pos], color='black', lw=1)
                        #     if 0.05 > p_value > 0.01:
                        #         p_symbol = '*'
                        #     elif 0.01 > p_value > 0.001:
                        #         p_symbol = '**'
                        #     elif p_value < 0.001:
                        #         p_symbol = '***'
                        #     plt.text((i+j)/2, y_pos, p_symbol, fontsize=12, ha='center', va='bottom', color='black')
                        # else:
                        #     print("not significant: ", i, j, stat, p_value)
            
            plt.tight_layout()
            feature_name = feature.split("_")[1]
            print(feature_name)
            if uncertainty_type == "absolute":
                plt.savefig(save_dir + f"plot_{feature_name}_abs_{metric}", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(save_dir + f"plot_{feature_name}_rel_{metric}", dpi=300, bbox_inches='tight')



    # feature = "calcium_arc"
    # feature = "calcium_depth"
    # feature = "calcium_thickness"


    # for metric in ['calcium_en', 'calcium_MI', 'calcium_brier']:#, 'total_en', 'total_MI', 'total_brier', 'total_en_raw', 'total_MI_raw', 'total_brier_raw']:
    #         print(metric)
    
    #         # first is for relative difference, second for absolute difference
    #         if uncertainty_type == "absolute":
    #             arc_groups = [0, 15, 35, 60, 360]
    #         else:
    #             # arc_groups = [0, 0.1, 0.25, 0.5, 2] # arc
    #             # arc_groups = [0, 0.15, 0.25, 0.5, 2] # depth
    #             arc_groups = [0, 0.05, 0.12, 0.32, 2] # thickness
                
    #         arc_groups_dict = {}

    #         for i in range(len(arc_groups)-1):
    #             arc_groups_dict[i] = info_calcium[(info_calcium[feature] > arc_groups[i]) & (info_calcium[feature] < arc_groups[i+1])][metric].values
            
    #         df = pd.DataFrame(columns=["group", "metric"])

    #         for key, value in arc_groups_dict.items():
    #             group_df = pd.DataFrame({'group': [key]*len(value),
    #                                     'metric': value})
    #             print(key, len(group_df))
    #             df = pd.concat([df, group_df], ignore_index = True)

    #         plt.figure(figsize=(5 * 1.2, 4))
    #         sns.violinplot(x='group', y='metric', data=df, color='tab:blue', cut=0,
    #                     linewidth=1, legend=False)

    #         for i in range(len(arc_groups_dict)):
    #             for j in range(len(arc_groups_dict)):
    #                 if i < j:
    #                     stat, p_value = mannwhitneyu(arc_groups_dict[i], arc_groups_dict[j])
    #                     significance = p_value <= 0.05
                            
    #                     if significance:
    #                         print(i, j, stat, p_value)
    #                         y_pos = max(df[(df['group'] == i) | (df['group'] == j)]['metric']) * (1 + 0.06*i)
    #                         plt.plot([i, j], [y_pos, y_pos], color='black', lw=1)
    #                         if 0.05 > p_value > 0.01:
    #                             p_symbol = '*'
    #                         elif 0.01 > p_value > 0.001:
    #                             p_symbol = '**'
    #                         elif p_value < 0.001:
    #                             p_symbol = '***'
    #                         plt.text((i+j)/2, y_pos, p_symbol, fontsize=12, ha='center', va='bottom', color='black')
            
    #         plt.tight_layout()
    #         feature_name = feature.split("_")[1]
    #         if uncertainty_type == "absolute":
    #             plt.savefig(save_dir + f"plot_calcium_{feature_name}_abs_{metric}", dpi=300, bbox_inches='tight')
    #         else:
    #             plt.savefig(save_dir + f"plot_calcium_{feature_name}_rel_{metric}", dpi=300, bbox_inches='tight')