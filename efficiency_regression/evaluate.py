import glob
import os
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

if not os.path.exists('./truth_pred_plot/'):
    os.mkdir('./truth_pred_plot')
if not os.path.exists('./model_plot/'):
    os.mkdir('./model_plot')

plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"
checkpoints = glob.glob('./*rat0.8/')

gs_choices = [5, 10, 25, 50, 75, 100]
model_choices = ['lstm', 'bilstm', 'transformer', 'lstm-transformer', 'bilstm-transformer']

model_dict = {mc: {g: None for g in gs_choices} for mc in model_choices}


def find_latest_record(files: list):
    if not files:
        return None
        # return {'trn_loss': [], 'val_loss': [], 'trn_r2': [], 'val_r2': []}
    maximum_num = max(int(f.split('.')[-2][7:11]) for f in files)
    for f in files:
        if str(maximum_num) in f:
            return np.load(f, allow_pickle=True).item()
    raise Exception(f"Maximum not found in {files}")


for f in tqdm(checkpoints):
    info = f.split('_')
    model_version = info[0].replace("./", '')
    epoch_num = int(info[3].replace('e', ''))
    gs = int(info[6].replace('gs', ''))
    preds = list(chain(*np.load(f'{f}/best_pred.npy', allow_pickle=True).tolist()))
    labels = list((np.load(f'{f}/best_label.npy', allow_pickle=True).tolist()))

    plt.scatter(x=labels, y=preds, label="(Label, Prediction)")
    real_range = np.linspace(min(list(preds + labels)), max(list(preds + labels)))
    plt.plot(real_range, real_range, color='m', linestyle="-.", linewidth=1, label="Identity Line (y=x)")
    plt.xlabel(f"Truth label\n$R^2 Score\t{round(r2_score(labels, preds), 5)}$")
    plt.ylabel("Prediction")
    plt.title(f"Model: {model_version}\tGroup Size: {gs}")
    plt.legend()
    plt.savefig(f'./truth_pred_plot/{model_version}_{gs}.png', bbox_inches='tight')
    plt.clf()

    record = find_latest_record(glob.glob(f'{f}/epoch*.npy'))
    model_dict[model_version][gs] = record
    if record:
        pass

for m in model_choices:
    g_choices = [g for g in gs_choices if model_dict[m][g]]
    trn_loss_min = [min(model_dict[m][g]['trn_loss']) for g in gs_choices if model_dict[m][g]]
    val_loss_min = [min(model_dict[m][g]['val_loss']) for g in gs_choices if model_dict[m][g]]
    trn_r2_max = [max(model_dict[m][g]['trn_r2']) for g in gs_choices if model_dict[m][g]]
    val_r2_max = [max(model_dict[m][g]['val_r2']) for g in gs_choices if model_dict[m][g]]

    plt.plot(g_choices, trn_loss_min, label="Min Training Loss")
    plt.plot(g_choices, val_loss_min, label="Min Validation Loss")
    plt.xlabel("Group Size")
    plt.ylabel("Loss")
    plt.title(f"Model: {m} Loss - GroupSize Curve")
    plt.legend()
    plt.savefig(f'./model_plot/{m}_loss.png', bbox_inches='tight')
    plt.clf()

    plt.plot(g_choices, trn_r2_max, label="Max Training $R^2$")
    plt.plot(g_choices, val_r2_max, label="Max Validation $R^2$")
    plt.xlabel("Group Size")
    plt.ylabel("$R^2$")
    plt.title(f"Model: {m} $R^2$ - GroupSize Curve")
    plt.legend()
    plt.savefig(f'./model_plot/{m}_r2.png', bbox_inches='tight')
    plt.clf()
