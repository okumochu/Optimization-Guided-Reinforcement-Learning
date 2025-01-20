## Uncertainty explained
- $A^*_{i}$：產品$i$一個時間時間週期的預期生產量，$i \in I$  
    - $\hat{A}_{ip}$：實際生產量（非模型輸入），$\text{uniform}(90, 110)$
- $D^*_{ip}$：在時間週期$p$結束前預期需求產品$i$的量，$i \in I,~p\in P$
  - 近$H$天內需求量是確定的，$H+1$天後為預測值
  - $\hat{D}_{ip}$：實際需求（非模型輸入），$\text{uniform}(80, 120)$
  <!-- - Normal distribution -->


# Model
## Sets
$I$：產品種類集合  
> $\{0, 1, 2, 3\}$

$\hat{P}$：總時間週期集合  
> $\{0, 1, ..., 179\}$

$P$：單一模型時間週期集合，$P$ 在 $\hat{P}$ 中滾動  
> $\{0, 1, ..., 19\} \rightarrow \{1, 2, ..., 20\} \rightarrow \{2, 3, ..., 21\} \rightarrow ...$

$F$：生產規劃定案的時間週期集合
> $\{0, 1, ..., 4\} \rightarrow \{1, 2, ..., 5\} \rightarrow \{2, 3, ..., 6\} \rightarrow ...$

## Parameters
$C^T_{ij}$：生產由產品$i$轉換至產品$j$的成本，$i \in I,~j \in I$  
> $V_j \cdot A^*_j \cdot 0.2$

$C^S_{i}$：產品$i$的倉儲成本，$i \in I$  
> $V_i \cdot 0.2$

$C^L_{i}$：產品$i$的延遲交貨成本，$i \in I$  
> $V_i \cdot 0.3$

$V_{i}$：產品$i$的單位毛利，$i \in I$  
> $\text{uniform}(80, 100)$

$S^I_{i}$：產品$i$的初始倉儲，$i \in I$  
> $\text{uniform}(90, 110)$

$X_{if}$：在時間週期$f$對產品$i$的定案生產規劃，$i \in I, f \in F \cup\{P_{-1}\}$  
> 沿襲上期規劃

$D^*_{ip}$：在時間週期$p$結束前需求產品$i$的量，$i \in I,~p\in P$  
> $\hat{D}_{ip} + (\text{noise} \propto p)$

$A^*_{i}$：產品$i$一個時間週期的生產量，$i \in I$  
> $100$

<!-- $Q^{\max}_{i}$：產品$i$一個時間週期的最大生產量，$i \in I$  -->

## Variables
### Decisions variables
$x_{ip}$：在時間週期$p$是否生產產品$i$，$i \in I, p \in P \cup\{P_{-1}\}$  
<!-- $q_{ip}$：在時間週期$p$生產產品$i$數量，$i \in I, p \in P \cup\{P_{-1}\}$   -->
### Derived variables
$s_{ip}$：產品$i$在時間週期$p$結束時的存貨量，$i \in I, p \in P \cup \{P_{-1}\}$  
$l_{ip}$：產品$i$在時間週期$p$結束時的延遲交貨量，$i \in I, p \in P$  
$z_{ijp}$：在時間週期$p$時是否從上一期生產產品$i$轉換至產品$j$，$i \in I, j \in I, p \in P$

## Objective
$\max.$ 銷售產品毛利 - 倉儲成本 - 延遲交貨成本 - 換線成本  
銷售產品毛利 = $\displaystyle{\sum_{i \in I}\sum_{p \in P}V_{i}A_{i}x_{ip}}$  
倉儲成本 = $\displaystyle{\sum_{i \in I}\sum_{p \in P}C^S_i s_{ip}}$  
延遲交貨成本 = $\displaystyle{\sum_{i \in I}C^L_{i} l_{ip}}$  
換線成本 = $\displaystyle{\sum_{i \in I} \sum_{j \in I, j \neq i} \sum_{p \in P}C^T_{ij} z_{ijp}}$
## Constraints

### 起始狀態定義
<!-- $s_{iP_0} = S^I_{i}, \forall i \in I$   -->
$s_{iP_{-1}} = S^I_i, \forall i \in I$  
$x_{if} = X_{if}, \forall i \in I, f \in F \cup \{P_{-1}\}$  
### 生產、倉儲、需求平衡
<!-- $s_{ip} = s_{ip-1} + A^*_i x_{ip} - D^*_{ip} + l_{ip}, \forall i \in I, p \in P \setminus\{P_0\}$ -->
$s_{ip} = s_{i(p-1)} + A^*_i x_{ip} - D^*_{ip} + l_{ip}, \forall i \in I, p \in P$
### 產品轉換
$\displaystyle{\sum_{i \in I}z_{ijp} = x_{jp}, \forall j \in I, p \in P}$  
$\displaystyle{\sum_{j \in I}z_{ijp} = x_{i(p-1)}, \forall i \in I, p \in P}$
### 生產限制
$\displaystyle{\sum_{i \in I}x_{ip} = 1, \forall p \in P}$  
<!-- $q_{ip} \leq Q^{\max}_i x_{ip}, \forall i \in I, p \in P$ -->

---
## Actor-Critic
- Description
  - 利用 Actor 產生策略，Critic 評價策略好壞，前者學習「看到什麼狀態要做什麼動作（答案）」，後者學習「做下這步動作過後的期望總回報（分數）」
- Termimologies
  - 軌跡：trajectory
  - 動作：action
  - 狀態：state
  - 價值：value
  - 更新梯度：gradient
  - 回報：reward
  - 總回報：return
- Notations
  - $N$：取樣軌跡 $\tau$ 數量
  - $Q(s, a | \theta_q)$：用 $\theta_q$ 所估計的狀態 $s$ 下，動作 $a$ 的價值（動作價值函數）
  - $\tau_n$：第 $n$ 個取樣出來的軌跡
  - $\pi(a | s, \theta)$：在給定狀態 $s$ 的情況下，$\theta$ 這組參數輸出動作 $a$ 的機率（策略函數）
  - $p_\theta(a | s)$：同 $\pi(a | s, \theta)$
  - $T_n$：軌跡 $\tau_n$ 的總步數
  - $s^{(n)}_t$：軌跡 $\tau_n$ 在第 $t$ 期的狀態
  - $a^{(n)}_t$：軌跡 $\tau_n$ 在第 $t$ 期的動作
  - $r^{(n)}_{t}$：軌跡 $\tau_n$ 在第 $t$ 期的回報
- Actor
  - $\pi(a | s, \theta)$ 或 $p_\theta(a | s)$
  - 參數：$\theta$
  - Actor 的更新梯度
    - 抽樣 $N$ 筆 $\tau$ 過後，對每一步 $t$ 計算 action-value 乘以 $\theta$ gradient 的值
    - 可以理解為在狀態 $s$ 當一個動作 $a$ 的 $Q$ 值越高時，就更新 $\theta$ 使得在狀態 $s$ 下產生 $a$ 的機率上升，反之，當 $Q$ 值很低則不鼓勵 $\pi$ 再次選擇此動作
    - $\displaystyle{\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} Q(s^{(n)}_t, a^{(n)}_t) \cdot \nabla_\theta\log p_\theta(a^{(n)}_t | s^{(n)}_t)}$
    - 此處的 $Q$ 是固定的，Actor 不會學習 $Q$
- Critic
  - $Q(s, a |\theta_q)$
  - 拿來估計 $\pi(a | s, \theta)$ 的 $Q$ 值，也就是未來的預期總回報
  - 參數：$\theta_q$
  - Critic 的更新梯度
    - 抽樣 $N$ 筆 $\tau$ 過後，對每一步 $t$ 計算 $Q$ 值估計的 TD-error，也就是把第 $t$ 期的回報加上 $t+1$ 期的最大估計 $Q$ 值減去第 $t$ 期的估計 $Q$ 值
    - 這個 TD-error 越小越好，所以要用梯度更新 $\theta_q$ 使其趨向最小值
    - $\displaystyle{\nabla_{\theta_q} \{\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} [r^{(n)}_t + \max_{a^{(n)}_{t+1}} Q(s^{(n)}_{t+1}, a^{(n)}_{t+1})] - Q(s^{(n)}_{t}, a^{(n)}_{t})\}^2}$

## Advantage Actor Critic
- Description
  - 先前學習的 Actor-Critic 方法中，Actor 動作的更新依據是利用估計的 $Q$ 值進行加權，由於各個狀態、動作的 $Q$ 值的 scale 可能都不一樣，會影響到收斂穩定性，所以引入 $A$ 取代 $Q$，利用把給定狀態下，各個動作的 $Q$ 值減去平均值 baseline（類似 Normalization）來改善收斂的穩定性。
- Termimologies
  - 軌跡：trajectory
  - 動作：action
  - 狀態：state
  - 價值：value
  - 更新梯度：gradient
  - 回報：reward
  - 總回報：return
- Notations
  - $N$：取樣軌跡 $\tau$ 數量
  - $K$：累計梯度採樣個數
  - $Q(s, a)$：狀態 $s$ 下，動作 $a$ 的價值（動作價值函數）
  - $V(s | \theta_v)$：狀態 $s$ 下，使用 $\theta_v$ 所估計出來的期望總回報（狀態價值函數）
  - $A(s, a)$：優勢函數，即 $Q(s, a) - V(s)$
  - $A^\prime(s, a)$：$A(s, a)$ 的近似
  - $\tau_n$：第 $n$ 個取樣出來的軌跡
  - $\pi(a | s, \theta)$：在給定狀態 $s$ 的情況下，$\theta$ 這組參數輸出動作 $a$ 的機率（策略函數）
  - $p_\theta(a | s)$：同 $\pi(a | s, \theta)$
  - $T_n$：軌跡 $\tau_n$ 的總步數
  - $s^{(n)}_t$：軌跡 $\tau_n$ 在第 $t$ 期的狀態
  - $a^{(n)}_t$：軌跡 $\tau_n$ 在第 $t$ 期的動作
  - $r^{(n)}_{t}$：軌跡 $\tau_n$ 在第 $t$ 期的回報
  - $R^{(n)}_t$：軌跡 $\tau_n$ 在第 $t$ 期後的總回報
  - $\gamma$：回報衰退率
- Reductions
  - 因為引入了 $A(s^{(n)}_t, a^{(n)}_t) = Q(s^{(n)}_t, a^{(n)}_t) - V(s^{(n)}_t)$，導致需要三個 networks ($Q$, $V$, $\pi$)，這會導致訓練複雜度太高，所以我們拋棄 $Q$，使用 $r^{(n)}_{t} + \gamma V(s^{(n)}_{t+1})$ 來近似 $Q(s^{(n)}_t, a^{(n)}_t)$
  - 雖然這個近似會有點誤差，但比起訓練三個模型還是比較好
  - $A(s^{(n)}_t, a^{(n)}_t) = Q(s^{(n)}_t, a^{(n)}_t) - V(s^{(n)}_t) \approx r^{(n)}_{t} + \gamma V(s^{(n)}_{t+1}) - V(s^{(n)}_t) = A^\prime(s^{(n)}_t, a^{(n)}_t)$
  <!-- - 又 $A^\prime(s^{(n)}_t, a^{(n)}_t) = r^{(n)}_{t} + \gamma V(s^{(n)}_{t+1}) - V(s^{(n)}_t)$ 是一個遞迴式，可以被化簡為一般式：
    - $A^\prime(s_{k-1}, a_{k-1}) = r_{k-1} + \gamma V(s_{k}) - V(s_{k-1})$
    - $A^\prime(s_{k-2}, a_{k-2}) = r_{k-2} + \gamma r_{k-1}  + \gamma^2 V(s_{k}) - V(s_{k-2})$
    - ...
    - $A^\prime(s_0, a_0) = r_0 + \gamma r_1 + ... + \gamma^k V(s_k) - V(s_{0})$
    - $A^\prime(s_t, a_t) = R^{(n)}_t - V(s_t)$ -->
  - 在訓練 Actor, Critic 的時候，都使用 $A^\prime(s^{(n)}_t, a^{(n)}_t)$  

- Actor
  - 使用累計梯度更新（提高運行效率）
  - 更新梯度
    - $\displaystyle{\sum_{t=1}^{K} A^\prime(s_t, a_t) \nabla_\theta \log \pi(a_t, s_t)}$
    - 把資料的 gradient 算出來後總和再一次更新


> TODO: 優化變數和常數 notation
---
$\displaystyle{\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} [Q(s^{(n)}_t, a^{(n)}_t) - V(s^{(n)}_t, \theta)] \cdot \nabla_\theta\log p_\theta(a^{(n)}_t | s^{(n)}_t)}$
  - $V(s)$：狀態價值 (state-value) 函數
---
  
    

**Algorithm 1** Robust optimization embedded A2C learning algorithm

---
**Require:** 
- A differentiable policy parameterization $\pi(a | s, \theta_\pi)$
- A differentiable state-value parameterization $V(s|\theta_V)$
- Select step-size hyper-parameters $0 < \alpha_\pi, \alpha_V \leq 1$

**Body**  


1. Initialize the parameters $\theta_\pi, \theta_V$.
2. **for** $N$ episodes **do**:  
   Initialize the episode with $s_0$

$\theta_{\pi} ∶=\theta_{\pi}+\alpha_{\pi}\ (\nabla_{\theta_{\pi}}\mathcal{L}(\theta_{\pi})+\beta \nabla _{\theta_{\pi}}H)$

$\displaystyle{\nabla_{\theta_V}\sum_{t=1}^{T_n}\frac{1}{T_n}A^\prime(s^{(n)}_t, a^{(n)}_t| \theta_V)^2}$