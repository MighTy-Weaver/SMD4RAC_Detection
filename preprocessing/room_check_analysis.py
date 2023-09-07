import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

replace_2016 = [714, 503, 1012, 235, 520, 735, 220, 335, 619, 817, 807, 202, 424, 801, 211, 402, 201, 326, 306, 429,
                414, 715, 311, 330]
replace_2017 = [432, 802, 227, 231, 733, 210, 315, 427, 430, 612, 613, 626, 630, 704, 914, 123, 307, 903]
replace_2018 = [219, 516, 417, 605, 816, 703, 803, 818, 915, 122, 207, 310, 320, 824, 518, 530, 913]
replace_2019 = [822, 730, 608, 617, 708, 825, 204, 216, 413, 703, 725, 810, 410, 830, 523, 618, 415, 328, 1007, 821,
                332]
replace_2020 = [808, 819, 403, 716, 303, 334, 832, 401, 622]
replace_2021 = [604, 702, 735, 217, 517, 710]

replace_dict = {2016: replace_2016, 2017: replace_2017, 2018: replace_2018, 2019: replace_2019, 2020: replace_2020,
                2021: replace_2021}


def calculate_vapor_density(temp: float):
    return 5.018 + 0.32321 * temp + 8.1847 * pow(10, -3) * pow(temp, 2) + 3.1243 * pow(10, -4) * pow(temp, 3)


room_volume = 30  # m^3

room_data = pd.read_csv('../data/room_check_original.csv', index_col=None)
pd.set_option("display.max_columns", None)

print(room_data)
print(list(room_data))
room_data['room'] = room_data['room'].astype(int)
room_data['Occupancy'].replace({math.nan: 0, 'Y': 1})

for i in range(len(room_data)):
    if math.isnan(room_data.loc[i, 'Indoor_after']):
        room_data = room_data.drop(i, axis=0)
    try:
        if math.isnan(room_data.loc[i, 'Indoor_RH_init']):
            room_data = room_data.drop(i, axis=0)
    except TypeError:
        pass
    room_data.loc[i, 'replaced_year'] = 0000
    for key in replace_dict.keys():
        if room_data.loc[i, 'room'] in replace_dict[key]:
            room_data.loc[i, 'replaced_year'] = key

    # calculate stabilize time
    time_list = room_data.loc[i, 'Time before stabilization'].split('min')
    room_data.loc[i, 'Time_stabilize_in_sec'] = int(time_list[0]) * 60 + int(time_list[1].replace('s', '')) if \
        time_list[1] != '' else int(time_list[0]) * 60

    # handle characters
    room_data.loc[i, 'Indoor_RH_init'] = float(room_data.loc[i, 'Indoor_RH_init'].replace('%', ''))
    room_data.loc[i, 'Indoor_RH_after'] = float(room_data.loc[i, 'Indoor_RH_after'].replace('%', ''))
    room_data.loc[i, 'Env RH'] = float(room_data.loc[i, 'Env RH'].replace('%', ''))
    room_data.loc[i, 'Rated Power'] = float(room_data.loc[i, 'Rated Power'].replace('W', ''))
    if len(room_data.loc[i, 'Time before stabilization'].split('min')) == 1:
        room_data.loc[i, 'Time before stabilization'] = \
            int(room_data.loc[i, 'Time before stabilization'].replace('s', '').split('min')[0]) * 60 + \
            int(room_data.loc[i, 'Time before stabilization'].replace('s', '').split('min')[1])
    else:
        room_data.loc[i, 'Time before stabilization'] = int(
            room_data.loc[i, 'Time before stabilization'].replace('s', '').split('min')[0]) * 60
    room_data.loc[i, 'Velocity'] = room_data.loc[i, 'Velocity'].replace('m/s', '')

    # add some new columns
    room_data.loc[i, 'Indoor_T_diff'] = -round(room_data.loc[i, 'Indoor_after'] - room_data.loc[i, 'Indoor_init'], 5)
    room_data.loc[i, 'Indoor_RH_diff'] = -round(room_data.loc[i, 'Indoor_RH_after'] -
                                                room_data.loc[i, 'Indoor_RH_init'], 5)
    room_data.loc[i, 'T_air_diff'] = round(room_data.loc[i, 'Tair_init'] -
                                           room_data.loc[i, 'Stabilized Tair'], 5)
    room_data.loc[i, 'init_saturate_vd'] = calculate_vapor_density(room_data.loc[i, 'Indoor_init'])
    room_data.loc[i, 'after_saturate_vd'] = calculate_vapor_density(room_data.loc[i, 'Indoor_after'])
    room_data.loc[i, 'init_actual_vd'] = room_data.loc[i, 'Indoor_RH_init'] * 0.01 * room_data.loc[
        i, 'init_saturate_vd']
    room_data.loc[i, 'after_actual_vd'] = room_data.loc[i, 'Indoor_RH_after'] * 0.01 * room_data.loc[
        i, 'after_saturate_vd']

# unite data type
room_data['Indoor_RH_init'] = room_data['Indoor_RH_init'].astype('float64')
room_data['Indoor_RH_after'] = room_data['Indoor_RH_after'].astype('float64')
room_data['Env RH'] = room_data['Env RH'].astype('float64')
room_data['Stabilized Tair'] = room_data['Stabilized Tair'].astype('float64')
room_data['Time before stabilization'] = room_data['Time before stabilization'].astype('int64')
room_data['Velocity'] = room_data['Velocity'].astype('float64')
room_data['replaced_year'] = room_data['replaced_year'].astype('int64')

# calculate latent heat (water)
room_data['init_vapor_mass'] = room_data['init_actual_vd'] * room_volume
room_data['after_vapor_mass'] = room_data['after_actual_vd'] * room_volume
room_data['vapor_mass_diff'] = room_data['init_vapor_mass'] - room_data['after_vapor_mass']  # gram
room_data['vapor_latent_heat'] = room_data['vapor_mass_diff'] * 2260  # joule

# calculate specific heat (air)
room_data['air_specific_heat'] = 1.012 * 1225 * room_volume * room_data['Indoor_T_diff']

# calculate efficiency
room_data['Efficiency'] = (room_data['vapor_latent_heat'] + room_data['air_specific_heat']) / (
        room_data['Rated Power'] * room_data['Time_stabilize_in_sec'])

room_data[room_data.Occupancy == 'Y'].to_csv('../occupancy.csv', index=False)
room_data[room_data.Efficiency > 1].to_csv('../Larger_than_1.csv', index=False)

room_data['prev_Efficiency'] = (1.012 * (room_data['Velocity'] * 1225) * room_data['T_air_diff'] + 2260 * room_data[
    'Velocity'] * (room_data['Indoor_RH_diff'] * 0.01) * 40) * 0.02 / room_data['Rated Power']
# # 40是绝对湿度下的含水量，1.012是空气比热容，1225是空气密度，2260是水的latent heat，0.02是出风口面积0.02平方米

room_data.to_csv('../data/room_check_processed.csv', index=False)

# 1.012 * (room_data['Velocity'] * 1225) * room_data['Indoor_T_diff'] * 1
# s * S
# c
# 速度 * 密度
# 温差ΔT
#
# 2260 * room_data['Velocity'] * (room_data['Indoor_RH_diff'] * 0.01) * 40)

# make the plot
room_data['Efficiency'] = room_data['Efficiency'].astype('float64')
room_data['prev_Efficiency'] = room_data['prev_Efficiency'].astype('float64')
Efficiency = list(room_data['Efficiency'])
room_data.sort_values(by=['Efficiency'], inplace=True, ascending=False)
Efficiency = sorted(Efficiency)

# print(room_data.corr())

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 9, 7
plt.rcParams["figure.autolayout"] = True

# normalize efficiency
# Efficiency = [(i - min(Efficiency)) / (max(Efficiency) - min(Efficiency)) for i in Efficiency]
room_data['Efficiency'] = [(i - min(Efficiency)) / (max(Efficiency) - min(Efficiency)) for i in room_data['Efficiency']]
sns.histplot(data=room_data, x="Efficiency")
plt.xlabel("Efficiency", fontsize=23)
plt.ylabel("Count", fontsize=23)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.axvline(x=0.71, color='r', linestyle='--')
plt.savefig('./efficiency_distribution.jpg', dpi=600, bbox_inches='tight')
plt.clf()

# efficiency_dict = {room_data.loc[i, 'room']: room_data.loc[i, 'Efficiency']
#                    for i in range(len(room_data))}
# np.save('../data/efficiency_dict.npy', efficiency_dict)

# high_class = [int(i.split('\\')[-1].split('.')[0]) for i in
#               glob.glob(
#                   '../../UROP 2 - Zhongming Lu/Inefficient-AC-detection/20210713191456_cate3_KMeans_22_33/a/*.png')] + [
#                  int(j.split('\\')[-1].split('.')[0]) for j in
#                  glob.glob(
#                      './20210713191456_cate3_KMeans_22_33/c/*.png')]
# high_class = [i for i in high_class if i in efficiency_room_list]
# medium_class = [int(j.split('\\')[-1].split('.')[0]) for j in glob.glob(
#     '../../UROP 2 - Zhongming Lu/Inefficient-AC-detection/20210713191456_cate3_KMeans_22_33/c/*.png')]
# medium_class = [i for i in medium_class if i in efficiency_room_list]
# low_class = [int(i.split('\\')[-1].split('.')[0]) for i in glob.glob(
#     '../../UROP 2 - Zhongming Lu/Inefficient-AC-detection/20210713191456_cate3_KMeans_22_33/b/*.png')]
# low_class = [i for i in low_class if i in efficiency_room_list]
#
# efficiency_room_list = [i for i in efficiency_room_list if i in high_class + medium_class + low_class]
# print(high_class, medium_class, low_class)
# a = high_class + medium_class
# print('TP', len([i for i in efficiency_room_list[:len(a)] if i in a]))
# print('TN', len([i for i in efficiency_room_list[len(a):] if i in low_class]))
# print('FP', len([i for i in efficiency_room_list[:len(a)] if i not in a]))
# print('FN', len([i for i in efficiency_room_list[len(a):] if i not in low_class]))
#
# TP = len([i for i in efficiency_room_list[:len(a)] if i in a])
# TN = len([i for i in efficiency_room_list[len(a):] if i in low_class])
# FP = len([i for i in efficiency_room_list[:len(a)] if i not in a])
# FN = len([i for i in efficiency_room_list[len(a):] if i not in low_class])

# print('Accuracy = {}'.format(round((TP + TN) / (len(efficiency_room_list)), 7)))
# print('Recall = {}'.format(round(TP / (TP + FN), 7)))
# print('Precision = {}'.format(round(TP / (TP + FP), 7)))
# print('F_1 score = {}'.format(round(2 * TP / (2 * TP + FP + FN), 7)))

# plt.hist(Efficiency, bins=50, density=False)
# plt.show()
# room_data.to_csv('./data/room_check.csv', index=False)
#
# room_data = room_data.drop(['Occupancy', 'Machine Mode', 'Rated Power', 'Facing', 'recent_status', 'weather'], axis=1)
# print(list(room_data))
# print(room_data.dtypes)
# print(room_data.describe())
