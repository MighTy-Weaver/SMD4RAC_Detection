import pandas as pd

record=pd.read_csv('')


# import os
#
# from matplotlib import pyplot as plt
#
# if not os.path.exists('./truth_pred_plot/'):
#     os.mkdir('./truth_pred_plot')
# if not os.path.exists('./model_plot/'):
#     os.mkdir('./model_plot')
#
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["savefig.bbox"] = "tight"
# # plt.scatter(x=labels, y=preds, label="(Label, Prediction)")
# real_range = np.linspace(min(list(preds + labels)), max(list(preds + labels)))
# seaborn.regplot(x=labels, y=preds, label="(Label, Prediction)", scatter=True)
# plt.plot(real_range, real_range, color='m', linestyle="-.", linewidth=1, label="Identity Line (y=x)")
# plt.xlabel(f"Truth label\n$Accuracy Score\t{round(r2_score(labels, preds), 5)}$")
# plt.ylabel("Prediction")
# plt.title(f"Model: {model_version}\tGroup Size: {gs}")
# plt.legend()
# plt.savefig(f'./truth_pred_plot/{model_version}_{gs}.png', bbox_inches='tight')
# plt.clf()
#
# if record:
#     length = range(1, len(record['trn_loss']) + 1)
#     plt.plot(length, record['trn_loss'], label='Training Loss')
#     plt.plot(length, record['val_loss'], label='Validation Loss')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"Model {model_version} Loss Curve")
#     plt.legend()
#     plt.savefig(f'./truth_pred_plot/{model_version}_{gs}_loss.png', bbox_inches='tight')
#     plt.clf()
#
#     plt.plot(length, record['trn_acc'], label='Training Acc')
#     plt.plot(length, record['val_acc'], label='Validation Acc')
#     plt.xlabel("Epoch")
#     plt.ylabel("Acc")
#     plt.title(f"Model {model_version} Acc Curve")
#     plt.legend()
#     plt.savefig(f'./truth_pred_plot/{model_version}_{gs}_acc.png', bbox_inches='tight')
#     plt.clf()
#
# for m in model_choices:
#     g_choices = [g for g in gs_choices if model_dict[m][g]]
#     trn_loss_min = [min(model_dict[m][g]['trn_loss']) for g in gs_choices if model_dict[m][g]]
#     val_loss_min = [min(model_dict[m][g]['val_loss']) for g in gs_choices if model_dict[m][g]]
#     trn_acc_max = [max(model_dict[m][g]['trn_acc']) for g in gs_choices if model_dict[m][g]]
#     val_acc_max = [max(model_dict[m][g]['val_acc']) for g in gs_choices if model_dict[m][g]]
#
#     print(m, max(val_acc_max), g_choices[val_acc_max.index(max(val_acc_max))])
#
#     plt.plot(g_choices, trn_loss_min, label="Min Training Loss")
#     plt.plot(g_choices, val_loss_min, label="Min Validation Loss")
#     plt.xlabel("Group Size")
#     plt.ylabel("Loss")
#     plt.title(f"Model: {m} Loss - GroupSize Curve")
#     plt.legend()
#     plt.savefig(f'./model_plot/{m}_loss.png', bbox_inches='tight')
#     plt.clf()
#
#     plt.plot(g_choices, trn_acc_max, label="Max Training Accuracy")
#     plt.plot(g_choices, val_acc_max, label="Max Validation Accuracy")
#     plt.xlabel("Group Size")
#     plt.ylabel("Accuracy")
#     plt.title(f"Model: {m} Accuracy - GroupSize Curve")
#     plt.legend()
#     plt.savefig(f'./model_plot/{m}_acc.png', bbox_inches='tight')
#     plt.clf()
