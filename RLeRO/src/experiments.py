from datahandler import DataHandler
from ormodel import ORModel
from evaluator import Evaluator
from utils.ormodelinputs import ORModelInputs
from utils.jspenv import JSPEnv
from rlmodel import A2C
from rlormodel import RLOR

import numpy as np
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
import time


def deterministic_exp(experiment: int, dh: DataHandler,
                      timelimit=None, solver_log=False,
                      plot_demand=False, export_mi=False,
                      ) -> Tuple[Dict[str, float], ORModelInputs]:

    decisions = list()
    sols = None
    nmis = list()
    dh.current_period = 0  # reset dh state

    # solving
    start_time = time.time()
    for i in range(dh.experiment_len):
        nmi = dh.generate_normal_model_inputs(sols)

        if plot_demand:
            dh.plot_demand(nmi)
        if export_mi:
            nmi.export(name=f"det_ex{experiment}_win{i}", save_dir=os.path.join(
                "params", "deterministic"))
        nmis.append(nmi)

        m = ORModel(nmi)
        m.optimize(timelimit=timelimit, log=solver_log)
        sols = m.get_sols()
        decisions.append(sols["x"])
    inference_time = time.time() - start_time
    print(f"do solve time {inference_time}")
    # evaluateing
    ei = dh.generate_evaluator_input(decisions)
    result = Evaluator(ei).evaluate()
    result["inference_time"] = inference_time
    return result, nmis


def robust_exp(experiment: int, normal_mis: List[ORModelInputs], dh: DataHandler,
               robustness=0.1, timelimit=None, solver_log=False, export_mi=False
               ) -> Dict[str, float]:
    decisions = list()
    sols = None
    dh.current_period = 0  # reset dh state
    start_time = time.time()
    # solving
    for i in range(dh.experiment_len):
        rmi = dh.generate_robust_model_input(
            sols, normal_mis[i], robustness=robustness)

        if export_mi:
            rmi.export(name=f"ro_ex{experiment}_win{i}",
                       save_dir=os.path.join("params", "robust"))

        m = ORModel(rmi)
        m.optimize(timelimit=timelimit, log=solver_log)
        sols = m.get_sols()
        decisions.append(sols["x"])
    inference_time = time.time() - start_time
    print(f"ro solve time {inference_time}")
    # evaluateing
    ei = dh.generate_evaluator_input(decisions)
    result = Evaluator(ei).evaluate()
    result["inference_time"] = inference_time
    return result


def A2C_train(a2c: A2C, train_episode: int, display):
    E_CNT = train_episode  # 3000
    sols = dict()
    ep_rs = list()
    entropies = list()
    objs = list()
    ep_r = 0
    dh = DataHandler()
    dhs = [dh]
    traverse = 0
    for episode in tqdm(range(E_CNT)):
        mi = dh.generate_RL_model_input(last_sols=sols, normal_input=None, inference=False)
        env = JSPEnv(init_mi=mi)
        s = env.reset()
        if episode in display:
            print(f"episode: {episode}")
            env.display()
        while True:
            a, is_random, entr_ = a2c.choose_action(s)
            s_, r, done, _ = env.step(a)
            if episode in display:
                env.display()
                print(f"reward: {r}")
                # print(env.state.to_numpy(), len(env.state.to_numpy()))
            a = np.argmax(a)
            # print(a, end='\t')
            ep_r += r
            entropies.append(entr_)
            # if r < 0:
            #     print(f"t: {env.state.t}")
            # print(is_random, r)

            a2c.learn(s, a, r, s_)

            if done:
                break
            s = s_

        sols["s"] = env.state.inv
        sols["x"] = env.state.schedule

        # TODO: index out of range error
        is_last = episode % (dh.experiment_len - dh.plan_horizon + 1) == 0
        if is_last:
            traverse += 1
            if traverse % 10 == 0: # 同一個環境 traverse 超過 10 次
                if len(dhs) <= 30: # 如果環境數 < 30 創立新環境
                    dh = DataHandler()
                    dhs.append(dh)
                    dh_idx = len(dhs)
                else:
                    # dh_idx = random.randrange(len(dhs)) # 隨機抽取
                    dh_idx = traverse // 10 % 30 # cyclical 的 traverse 環境
                    dh = dhs[dh_idx] 
                print(f"traverse: {traverse}, using dh pool [{dh_idx}]")

            # if traverse % 10 == 0:
            # # if traverse == 10:
            #     # print("new DataHandler!")
            #     dh = DataHandler()
            #     dhs.append(dh)
            #     # traverse = 0
            ep_rs.append(ep_r)
            objs.append(env.obj)
            dh.current_period = 0
            ep_r = 0

    return ep_rs, objs, entropies, dhs


def A2C_inference(a2c: A2C, dh: DataHandler, normal_mis: List[ORModelInputs], display):
    dh.current_period = 0  # reset dh state
    sols = dict()
    decisions = list()
    start_time = time.time()
    for episode in range(dh.experiment_len):
        # TODO: index out of range error
        # if episode % (dh.experiment_len - dh.plan_horizon) == 0:
        #     dh.current_period = 0
        mi = dh.generate_RL_model_input(
            sols, normal_mis[dh.current_period])
        env = JSPEnv(init_mi=mi)
        s = env.reset()
        if episode in display:
            print(f"episode: {episode}")
            env.display()
        ep_r = 0
        while True:
            a, is_random, entr_ = a2c.choose_action(s, inference=True)
            s_, r, done, _ = env.step(a)
            if episode in display:
                env.display()
                print(f"reward: {r}")
                # print(env.state.to_numpy(), len(env.state.to_numpy()))
            a = np.argmax(a)
            # print(a, end='\t')
            ep_r += r
            # if r < 0:
            #     print(f"t: {env.state.t}")
            # print(is_random, r)

            if done:
                break
            s = s_

        sols["s"] = env.state.inv
        sols["x"] = env.state.schedule
        decisions.append(sols["x"])

    inference_time = time.time() - start_time
    print(f"rl solve time {inference_time}")
    ei = dh.generate_evaluator_input(decisions)
    results = Evaluator(ei).evaluate()
    results["inference_time"] = inference_time
    return results


def RLOR_train(rlor: RLOR, dhs: List[DataHandler], train_episode: int, rho: float, display, oracle="DM", robustness=0.1):
    entropies = list()
    E_CNT = train_episode  # 3000
    sols = dict()
    ep_rs = list()
    objs = list()
    ep_r = 0
    # dh = DataHandler()
    dh = dhs[0]
    dh.current_period = 0
    traverse = 0
    for episode in tqdm(range(E_CNT)):
        mi = dh.generate_RL_model_input(last_sols=sols, normal_input=None, inference=False)
        env = JSPEnv(init_mi=mi)
        s = env.reset()
        if episode in display:
            print(f"episode: {episode}")
            env.display()
        while True:
            a, is_random, entr_ = rlor.choose_action(s)
            if not is_random and entr_ < rho: # OR invoked
                # print("OR invoked!", f"{entr_} < {rho}")
                if oracle == "DM":
                    ormi = dh.generate_DM_oracle_input(env.state)
                elif oracle == "RM":
                    ormi = dh.generate_RM_oracle_input(env.state, robustness=robustness)
                else:
                    raise ValueError("Invalid oracle value")
                # ormi.export("debug", ".")
                m = ORModel(ormi)
                m.optimize(timelimit=None, log=False)
                x = m.get_sols()["x"]
                a = [round(x[i, env.mi.P[-1]]) for i in env.mi.I]

            s_, r, done, _ = env.step(a)
            if episode in display:
                env.display()
                print(f"reward: {r}")
                # print(env.state.to_numpy(), len(env.state.to_numpy()))
            a = np.argmax(a)
            # print(a, end='\t')
            ep_r += r
            entropies.append(entr_)
            # if r < 0:
            #     print(f"t: {env.state.t}")
            # print(is_random, r)

            rlor.learn(s, a, r, s_)

            if done:
                break
            s = s_

        sols["s"] = env.state.inv
        sols["x"] = env.state.schedule

        # TODO: index out of range error
        is_last = episode % (dh.experiment_len - dh.plan_horizon + 1) == 0
        if is_last:
            # print(dh.current_period)
            traverse += 1
            # dh = dhs[traverse // 10]
            # dh_idx = random.randrange(len(dhs)) # 隨機抽取
            dh_idx = traverse // 10 % 30 # cyclical 的 traverse 環境
            dh = dhs[dh_idx] 
            dh.current_period = 0

            ep_rs.append(ep_r)
            objs.append(env.obj)
            ep_r = 0

    return ep_rs, objs, entropies


def RLOR_inference(rlor: RLOR, dh: DataHandler, normal_mis: List[ORModelInputs], display):
    dh.current_period = 0  # reset dh state
    sols = dict()
    decisions = list()
    start_time = time.time()
    for episode in range(dh.experiment_len):
        # TODO: index out of range error
        # if episode % (dh.experiment_len - dh.plan_horizon) == 0:
        #     dh.current_period = 0
        mi = dh.generate_RL_model_input(
            sols, normal_mis[dh.current_period])
        env = JSPEnv(init_mi=mi)
        s = env.reset()
        if episode in display:
            print(f"episode: {episode}")
            env.display()
        ep_r = 0
        while True:
            a, is_random, entr_ = rlor.choose_action(s, inference=True)
            s_, r, done, _ = env.step(a)
            if episode in display:
                env.display()
                print(f"reward: {r}")
                # print(env.state.to_numpy(), len(env.state.to_numpy()))
            a = np.argmax(a)
            # print(a, end='\t')
            ep_r += r
            # if r < 0:
            #     print(f"t: {env.state.t}")
            # print(is_random, r)

            if done:
                break
            s = s_

        sols["s"] = env.state.inv
        sols["x"] = env.state.schedule
        decisions.append(sols["x"])

    inference_time = time.time() - start_time
    print(f"rlor solve time {inference_time}")
    ei = dh.generate_evaluator_input(decisions)
    results = Evaluator(ei).evaluate()
    results["inference_time"] = inference_time
    return results

def A2C_exp_bk(train_episode, normal_mis: List[ORModelInputs], dh: DataHandler, train_display, inference_display):
    def train(a2c: A2C):
        E_CNT = train_episode  # 3000
        dh.current_period = 0  # reset dh state
        sols = dict()
        ep_rs = list()
        objs = list()
        ep_r = 0
        for episode in tqdm(range(E_CNT)):
            mi = dh.generate_RL_model_input(
                sols, normal_mis[dh.current_period])
            env = JSPEnv(init_mi=mi)
            s = env.reset()
            if episode in train_display:
                print(f"episode: {episode}")
                env.display()
            while True:
                a, is_random = a2c.choose_action(s)
                s_, r, done, _ = env.step(a)
                if episode in train_display:
                    env.display()
                    print(f"reward: {r}")
                    # print(env.state.to_numpy(), len(env.state.to_numpy()))
                a = np.argmax(a)
                # print(a, end='\t')
                ep_r += r
                # if r < 0:
                #     print(f"t: {env.state.t}")
                # print(is_random, r)

                a2c.learn(s, a, r, s_)

                if done:
                    break
                s = s_

            sols["s"] = env.state.inv
            sols["x"] = env.state.schedule

            # TODO: index out of range error
            is_last = episode % (dh.experiment_len - dh.plan_horizon + 1) == 0
            if is_last:
                ep_rs.append(ep_r)
                objs.append(env.obj)
                dh.current_period = 0
                ep_r = 0

        return ep_rs, objs

    def inference(a2c: A2C):
        dh.current_period = 0  # reset dh state
        sols = dict()
        # returns = np.zeros(E_CNT)
        # objs = np.zeros(E_CNT)
        decisions = list()
        for episode in range(dh.experiment_len):
            # TODO: index out of range error
            # if episode % (dh.experiment_len - dh.plan_horizon) == 0:
            #     dh.current_period = 0
            mi = dh.generate_RL_model_input(
                sols, normal_mis[dh.current_period])
            env = JSPEnv(init_mi=mi)
            s = env.reset()
            if episode in inference_display:
                print(f"episode: {episode}")
                env.display()
            ep_r = 0
            while True:
                a, is_random = a2c.choose_action(s, inference=True)
                s_, r, done, _ = env.step(a)
                if episode in inference_display:
                    env.display()
                    print(f"reward: {r}")
                    # print(env.state.to_numpy(), len(env.state.to_numpy()))
                a = np.argmax(a)
                # print(a, end='\t')
                ep_r += r
                # if r < 0:
                #     print(f"t: {env.state.t}")
                # print(is_random, r)

                if done:
                    break
                s = s_

            sols["s"] = env.state.inv
            sols["x"] = env.state.schedule
            decisions.append(sols["x"])
        
        return decisions

    a2c = A2C()
    results = dict()
    results["train_ep_rs"], results["train_objs"] = train(a2c)
    start_time = time.time()
    decisions = inference(a2c)
    print(f"rl inference time {time.time() - start_time}")
    ei = dh.generate_evaluator_input(decisions)
    results["inference_eval"] = Evaluator(ei).evaluate()
    return results
