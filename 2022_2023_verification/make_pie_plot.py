import os

import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = 10,5
plt.rcParams.update({'font.size': 18})
# Data for the first plot
total_racs_2021_2022 = 5
consistently_inefficient_racs = 3

# Data for the second plot
normal_to_inefficient_racs = 4
reduced_efficiency_racs = 3

# Pie chart for the first plot
labels = ['Detected', 'Failed']
sizes = [consistently_inefficient_racs, total_racs_2021_2022 - consistently_inefficient_racs]
colors = ['#ff9999', '#c2c2f0']
plt.subplot(1, 2, 1)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.xlabel('RACs with Consistent Inefficiency\nin 2021 and 2022', loc='left')

# Pie chart for the second plot
labels = ['Detected', 'Failed']
sizes = [reduced_efficiency_racs, normal_to_inefficient_racs - reduced_efficiency_racs]
colors = ['#ff9999', '#c2c2f0']
plt.subplot(1, 2, 2)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.xlabel('RACs with Decreased Inefficiency\nfrom 2021 to 2022', loc='left')

plt.suptitle("Model Verification Results on 2022 Data")
# Display the plots
plt.tight_layout()
plt.savefig('pie_plot.png', bbox_inches='tight', dpi=800)
