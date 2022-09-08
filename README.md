# Learn from smart electricity meter to evaluate the air conditioner’s efficiency

This is the official code repository for the paper ***Learn from smart electricity meter to evaluate the air
conditioner’s efficiency***, which is to be submitted to [Nature Sustainability](https://www.nature.com/natsustain/).

## 1. Citing this work

Please use the Bibtex below for citation of this work:

```
To Be Updated
```

## 2. Environment Setup

Experiments are conducted under Windows 10 with Python 3.7/3.8 as the developing environment.

Use the following code segment to install all the required packages:

```commandline
pip install -r requirements.txt
```

## 3. Data Compilation

***Due to the privacy issues, the dataset will not be made open to public.***

However, we provide a 200
lines [sample version](https://github.com/MighTy-Weaver/SMD4RAC_Detection/blob/main/sample_data.csv) of the
full dataset to demonstrate the formation of our experimenting data, and you can check
the `preprocessing/data_compilation.py` for how our data is compiled from different categories of data.

*Remarks: Please notice that the `Location` in `sample_data.csv` are set to 0 for privacy.*

## 4. Model Training & Evaluation

![setting1](./demo/SettingI_all.jpg)
![setting2](./demo/SettingII_all.jpg)

![setting1_model](./demo/SettingI_model.jpg)
![setting2_model](./demo/SettingII_model.jpg)

## 5. Energy Saving Result

![energy_saving](./preprocessing/TOTAL_comparison.png)

## 6. Acknowledgement

To Be Updated.

## 7. Contact

If you have any question, feel free to email me at `1874240442@qq.com`. This email will be active all the time. 