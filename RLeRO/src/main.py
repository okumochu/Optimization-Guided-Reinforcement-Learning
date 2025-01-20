# %%
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from tqdm import tqdm
from experiments import deterministic_exp, robust_exp, A2C_train, A2C_inference, RLOR_train, RLOR_inference  # A2C_exp_bk
from rlmodel import A2C
from rlormodel import RLOR
from evaluator import Evaluator
from datahandler import DataHandler
from ormodel import ORModel

# D:\projects\chemical-scheduling-rlro> python -u .\main.py > debug.txt
# %%

# os.environ["GRB_LICENSE_FILE"] = os.path.join("utils", "gurobi.lic")
random.seed(880203)

# %%
a2c = A2C()
a2c_train_ep_rs, a2c_train_objs, a2c_entropies, dhs = A2C_train(a2c=a2c,
                                                                train_episode=5000,
                                                                display=[])
# %%
plt.title("train_entrs")
plt.plot(a2c_entropies, label="a2c")
plt.legend()

# %%
plt.title("train_ep_rs")
plt.plot(a2c_train_ep_rs, label="a2c")
plt.legend()

# %%
plt.title("train_objs")
plt.plot(a2c_train_objs, label="a2c")
plt.legend()

# %%
rho = 1.25  # 10: 2.20

# %%
rldm = RLOR()

rldm_train_ep_rs, rldm_train_objs, rldm_entropies = RLOR_train(rlor=rldm,
                                                               dhs=dhs,
                                                               train_episode=5000,
                                                               display=[],
                                                               rho=rho,
                                                               oracle="DM")

# %%
rlrm = RLOR()
rlrm_train_ep_rs, rlrm_train_objs, rlrm_entropies = RLOR_train(rlor=rlrm,
                                                               dhs=dhs,
                                                          train_episode=5000,
                                                          display=[],
                                                          rho=rho,
                                                          oracle="RM",
                                                          robustness=0.05)

# %%
plt.figure(figsize=(20, 5))

plt.title("train_entrs")
plt.plot(a2c_entropies, marker='.', color="blue",
         label="a2c", alpha=0.2, linewidth=0.5)
plt.plot(rldm_entropies, marker='^', color="orange",
         label="rldm", alpha=0.2, linewidth=0.5)
plt.plot(rlrm_entropies, marker='1', color="green", label="rlrm", alpha=0.2, linewidth=0.5)

rldm_entropies_ = pd.Series(rldm_entropies)
rlrm_entropies_ = pd.Series(rlrm_entropies)
# signal = entropies_[entropies_ < entropies_.rolling(50).quantile(0.1)]
rldm_signal = rldm_entropies_[rldm_entropies_ < rho]
rlrm_signal = rlrm_entropies_[rlrm_entropies_ < rho]

plt.scatter(rldm_signal.index, rldm_signal,
            color="red", label="rldm_signal", alpha=0.5)
plt.scatter(rlrm_signal.index, rlrm_signal, color="green", label="rlrm_signal", alpha=0.5)
plt.legend()


# %%
plt.title("train_ep_rs")
plt.plot(a2c_train_ep_rs, label="a2c")
plt.plot(rldm_train_ep_rs, label="rldm")
plt.plot(rlrm_train_ep_rs, label="rlrm")
plt.legend()
# %%
plt.title("train_objs")
plt.plot(a2c_train_objs, label="a2c")
plt.plot(rldm_train_objs, label="rldm")
plt.plot(rlrm_train_objs, label="rlrm")
plt.legend()

# %%
dm_results = {
    "objective": list(),
    "sales_profit": list(),
    "stockout_cost": list(),
    "transition_cost": list(),
    "inventory_cost": list(),
    "inference_time": list()
}
rm_results = {
    "objective": list(),
    "sales_profit": list(),
    "stockout_cost": list(),
    "transition_cost": list(),
    "inventory_cost": list(),
    "inference_time": list()
}
rl_results = {
    "objective": list(),
    "sales_profit": list(),
    "stockout_cost": list(),
    "transition_cost": list(),
    "inventory_cost": list(),
    "inference_time": list()
}
rldm_results = {
    "objective": list(),
    "sales_profit": list(),
    "stockout_cost": list(),
    "transition_cost": list(),
    "inventory_cost": list(),
    "inference_time": list()
}
rlrm_results = {
    "objective": list(),
    "sales_profit": list(),
    "stockout_cost": list(),
    "transition_cost": list(),
    "inventory_cost": list(),
    "inference_time": list()
}


# %%
experiment_cnt = 5
all_res = {
    "dh": list(),
    "dm": list(),
    "rm": list(),
    "rl": list(),
    "rldm": list(),
    "rlrm": list()
}

for e in range(experiment_cnt):
    print(f"experiemnt: {e}")

    dh = DataHandler()
    # # dh.export(f"ex{ex}_conf", save_dir="params")
    dm_result, nmis = deterministic_exp(experiment=e,
                                        dh=dh,
                                        timelimit=None,
                                        solver_log=False,
                                        plot_demand=False,
                                        export_mi=False)

    # rm_result = robust_exp(experiment=e,
    #                        normal_mis=nmis,
    #                        dh=dh,
    #                        robustness=0.05,
    #                        timelimit=None,
    #                        solver_log=False,
    #                        export_mi=False)

    # rl_result = A2C_inference(a2c=a2c,
    #                           dh=dh,
    #                           normal_mis=nmis,
    #                           display=[])  # [i for i in range(100)]

    # rldm_result = RLOR_inference(rlor=rldm,
    #                              dh=dh,
    #                              normal_mis=nmis,
    #                              display=[])

    # rlrm_result = RLOR_inference(rlor=rlrm,
    #                              dh=dh,
    #                              normal_mis=nmis,
    #                              display=[])
    
    all_res['dh'].append(dh)
    all_res['dm'].append(dm_result)
    # all_res['rm'].append(rm_result)
    # all_res['rl'].append(rl_result)
    # all_res['rldm'].append(rldm_result)
    # all_res['rlrm'].append(rlrm_result)

    # train_display = [] # [i for i in range(50000) if i % 1000 == 0]
    # inference_display = [] # inference_display = [i for i in range(50)]
    # rl_result = A2C_exp(train_episode=40000, normal_mis=nmis,
    #                 dh=dh,
    #                 train_display=train_display,
    #                 inference_display=inference_display)
    dm_results["objective"].append(dm_result["obj"])
    dm_results["sales_profit"].append(dm_result["sales_profit"])
    dm_results["stockout_cost"].append(dm_result["stockout_cost"])
    dm_results["transition_cost"].append(dm_result["transition_cost"])
    dm_results["inventory_cost"].append(dm_result["inventory_cost"])
    dm_results["inference_time"].append(dm_result["inference_time"])

    # rm_results["objective"].append(rm_result["obj"])
    # rm_results["sales_profit"].append(rm_result["sales_profit"])
    # rm_results["stockout_cost"].append(rm_result["stockout_cost"])
    # rm_results["transition_cost"].append(rm_result["transition_cost"])
    # rm_results["inventory_cost"].append(rm_result["inventory_cost"])
    # rm_results["inference_time"].append(rm_result["inference_time"])

    # rl_results["objective"].append(rl_result["obj"])
    # rl_results["sales_profit"].append(rl_result["sales_profit"])
    # rl_results["stockout_cost"].append(rl_result["stockout_cost"])
    # rl_results["transition_cost"].append(rl_result["transition_cost"])
    # rl_results["inventory_cost"].append(rl_result["inventory_cost"])
    # rl_results["inference_time"].append(rl_result["inference_time"])

    # rldm_results["objective"].append(rldm_result["obj"])
    # rldm_results["sales_profit"].append(rldm_result["sales_profit"])
    # rldm_results["stockout_cost"].append(rldm_result["stockout_cost"])
    # rldm_results["transition_cost"].append(rldm_result["transition_cost"])
    # rldm_results["inventory_cost"].append(rldm_result["inventory_cost"])
    # rldm_results["inference_time"].append(rldm_result["inference_time"])

    # rlrm_results["objective"].append(rlrm_result["obj"])
    # rlrm_results["sales_profit"].append(rlrm_result["sales_profit"])
    # rlrm_results["stockout_cost"].append(rlrm_result["stockout_cost"])
    # rlrm_results["transition_cost"].append(rlrm_result["transition_cost"])
    # rlrm_results["inventory_cost"].append(rlrm_result["inventory_cost"])
    # rlrm_results["inference_time"].append(rlrm_result["inference_time"])


# %%
dm_results = pd.DataFrame(dm_results)
# rm_results = pd.DataFrame(rm_results)
# rl_results = pd.DataFrame(rl_results)
# rldm_results = pd.DataFrame(rldm_results)
# rlrm_results = pd.DataFrame(rlrm_results)

# %%
display(dm_results.describe())
# display(rm_results.describe())
# display(rl_results.describe())
# display(rldm_results.describe())
# display(rlrm_results.describe())


# %%



case_ = 0
product = 3
mod = "dm"


prd = [all_res[mod][case_]['x'][x] for x in all_res[mod][case_]['x'] if x[0] == product]
yld = [all_res['dh'][case_].A_hat[a] for a in all_res['dh'][case_].A_hat if a[0] == product]


plt.plot([all_res['dh'][case_].D_hat[d] for d in all_res['dh'][case_].D_hat if d[0] == product], label="demand")
plt.plot([x*a for x, a in zip(prd, yld)], label='production')
plt.plot([all_res[mod][case_]['s'][s] for s in all_res[mod][case_]['s'] if s[0] == product], label='inventory')
plt.plot([all_res[mod][case_]['l'][l] for l in all_res[mod][case_]['l'] if l[0] == product], label='stockout')

plt.legend()
#%%
plt.figure()
for product in range(4):
    # plt.plot([all_res['dh'][case_].D_hat[d] for d in all_res['dh'][case_].D_hat if d[0] == product], label="demand")
    # plt.plot([all_res[mod][case_]['s'][s] for s in all_res[mod][case_]['s'] if s[0] == product], label='inventory')
    plt.plot([all_res[mod][case_]['l'][l] for l in all_res[mod][case_]['l'] if l[0] == product], label='stockout')



#%%

# print(rl_result["obj"])
# print(dm_result["obj"])
# print(rm_result["obj"])
# env.reset()
# state, reward, done, _ = env.step([1, 0, 0, 0])
# state, reward, done, _ = env.step([0, 0, 1, 0])
# state, reward, done, _ = env.step([0, 0, 1, 0])
# state, reward, done, _ = env.step([1, 0, 0, 0])
# state, reward, done, _ = env.step([1, 0, 0, 0])

# dm_results["objective"].append(dm_result["obj"])
# dm_results["sales_profit"].append(dm_result["sales_profit"])
# dm_results["stockout_cost"].append(dm_result["stockout_cost"])
# dm_results["transition_cost"].append(dm_result["transition_cost"])
# dm_results["inventory_cost"].append(dm_result["inventory_cost"])

# rm_results["objective"].append(rm_result["obj"])
# rm_results["sales_profit"].append(rm_result["sales_profit"])
# rm_results["stockout_cost"].append(rm_result["stockout_cost"])
# rm_results["transition_cost"].append(rm_result["transition_cost"])
# rm_results["inventory_cost"].append(rm_result["inventory_cost"])


# dm_results = pd.DataFrame(dm_results)
# rm_results = pd.DataFrame(rm_results)

# %%
# display(dm_results.head())
# display(rm_results.head())
# display(dm_results.describe())
# display(rm_results.describe())


# %%


# %%
# display(determined)
# display(robust)


# # %%
# determined.describe().round().astype(int).drop("count")
# # %%
# robust.describe().round().astype(int).drop("count")
# # %%
# plt.scatter(range(experiment_cnt), determined["obj"], label="det eval")
# plt.scatter(range(experiment_cnt), robust["obj"], label="ro eval")
# # plt.title("")
# plt.legend()


# # %%
# plt.hist(determined["obj"], label="det objs", alpha=0.9)
# plt.hist(robust["obj"], label="ro objs", alpha=0.9)
# plt.title("objective value distribution")
# plt.legend()

# %%
# with open("debug.txt", 'w') as f:
#     f.write("P:\t")
#     for p in m.mi.P:
#         f.write(f"{p}\t")
#     f.write("\n")

#     for i in m.mi.I:
#         f.write(f"x{i}:\t")
#         for p in m.mi.P:
#             v = round(sols["x"][i, p])
#             f.write(f"{v}\t")
#         f.write("\n")

#     f.write("\n")

# %%
# data = dict()
# data["p"] = e.ei.P_hat
# for i in m.mi.I:
#     data[f"x{i}"] = [round(det_result["x"][i, p]) for p in e.ei.P_hat]
# for i in m.mi.I:
#     data[f"D^{i}"] = [round(e.ei.D_hat[i, p]) for p in e.ei.P_hat]
# for i in m.mi.I:
#     data[f"s{i}"] = [round(det_result["s"][i, p]) for p in e.ei.P_hat]
# for i in m.mi.I:
#     data[f"l{i}"] = [round(det_result["l"][i, p]) for p in e.ei.P_hat]
# pd.DataFrame(data)


# # %%
# data = dict()
# data["p"] = e.ei.P_hat
# for i in m.mi.I:
#     data[f"x{i}"] = [round(ro_result["x"][i, p]) for p in e.ei.P_hat]
# for i in m.mi.I:
#     data[f"D^{i}"] = [round(e.ei.D_hat[i, p]) for p in e.ei.P_hat]
# for i in m.mi.I:
#     data[f"s{i}"] = [round(ro_result["s"][i, p]) for p in e.ei.P_hat]
# for i in m.mi.I:
#     data[f"l{i}"] = [round(ro_result["l"][i, p]) for p in e.ei.P_hat]
# pd.DataFrame(data)


# # %%
# valid = pd.DataFrame(data)
# for i in m.mi.I:
#     valid[f"x{i}'"] = valid[f"x{i}"] * m.mi.A_star[i]

# valid.to_csv("debug.csv", index=False)
# # %%

# %%
# fig, ax = plt.subplots(dh.product_cnt, 1, figsize=(
#     20, 12), constrained_layout=True)
# fig.suptitle("Demand for each products", fontsize=16)

# for i in dh.I:
#     actual = [dh.D_hat[i, p] for p in dh.P_hat]
#     predict = [mi.D_star[i, p] for p in dh.P_hat]
#     error = [abs(predict[p] - actual[p])
#              for p in range(len(dh.P_hat))]

#     ax[i].set_title(f'Product {i}')
#     ax[i].plot(actual, label="Actual")
#     ax[i].plot(predict, label="Expected")
#     ax[i].plot(error, label="Error")
#     ax[i].legend()

# %%
import plotly.figure_factory as ff
mod = "dm"
case_ = 1
all_res[mod][case_]['x']


dmdf = [
    dict(Task=f"Product {pd_[0]}", Start=pd_[1], Finish=pd_[1] + 1)
    for pd_ in all_res[mod][case_]['x'] if all_res[mod][case_]['x'][pd_] == 1
]

# df = [dict(Task="Job A", Start=1, Finish=5.5),
#       dict(Task="Job B", Start=2, Finish=3),
#       ]

fig = ff.create_gantt(dmdf, index_col='Task', show_colorbar=True,
                      group_tasks=True)
fig.update_layout(xaxis_type='linear')
fig.show()



# %%
import plotly.figure_factory as ff
mod = "dm"
case_ = 0
all_res[mod][case_]['x']


rmdf = [
    dict(Task=f"Product {pd_[0]}", Start=pd_[1], Finish=pd_[1] + 1)
    for pd_ in all_res[mod][case_]['x'] if all_res[mod][case_]['x'][pd_] == 1
]

# df = [dict(Task="Job A", Start=1, Finish=5.5),
#       dict(Task="Job B", Start=2, Finish=3),
#       ]

fig = ff.create_gantt(rmdf, index_col='Task', show_colorbar=True,
                      group_tasks=True)
fig.update_layout(xaxis_type='linear')
fig.show()
# %%
import plotly.figure_factory as ff
mod = "dm"
case_ = 2
all_res[mod][case_]['x']


rldf = [
    dict(Task=f"{pd_[0]}", Start=pd_[1], Finish=pd_[1] + 1)
    for pd_ in all_res[mod][case_]['x'] if all_res[mod][case_]['x'][pd_] == 1
]

# df = [dict(Task="Job A", Start=1, Finish=5.5),
#       dict(Task="Job B", Start=2, Finish=3),
#       ]

fig = ff.create_gantt(rldf, index_col='Task', show_colorbar=True,
                      group_tasks=True)
fig.update_layout(xaxis_type='linear')
fig.show()

# %%
dmdf = pd.DataFrame(dmdf)
rmdf = pd.DataFrame(rmdf)
rldf = pd.DataFrame(rldf)
# %%
(dmdf['Task'] == rmdf['Task']).sum() / dmdf.shape[0]

# %%
(dmdf['Task'] == rldf['Task']).sum() / dmdf.shape[0]

# %%
case = 0
models = ['dm', 'rm', 'rl', 'rldm', 'rlrm']

res = dict()
for mod in models:
    df = [
        dict(Task=f"{pd_[0]}", Start=pd_[1], Finish=pd_[1] + 1)
        for pd_ in all_res[mod][case_]['x'] if all_res[mod][case_]['x'][pd_] == 1
    ]
    res[mod] = pd.DataFrame(df)


gan = pd.DataFrame()
for mod in models:
    for mod_ in models:
        a = (res[mod]['Task'] == res[mod_]['Task']).sum() / res[mod_].shape[0]
        # print(mod, mod_, a)
        gan.loc[mod, mod_] = a




    
    



# %%
