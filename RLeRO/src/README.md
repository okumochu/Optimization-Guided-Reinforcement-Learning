# chemical-scheduling-rlro


## Uncertainty explained

- $A^*_{i}$：產品$i$一個時間時間週期的生產量，$i \in I$  
  - try:
    - RO & SO & deterministic on Uniform distribution  
    - RO & SO & deterministic on Normal distribution  
- $D^*_{ip}$：在時間週期$p$結束前需求產品$i$的量，$i \in I,~p\in P$
  - 近$H$天內需求量是確定的，$H+1$天後為預測值
  - Uniform distribution
  - Normal distribution




# Model
## Sets
$I$：產品種類集合  
$\hat{P}$：總時間週期集合  
$P$：單一模型時間週期集合，$P$ 在 $\hat{P}$ 中滾動  
$F$：生產規劃定案的時間週期集合

## Parameters
$C^T_{ij}$：生產由產品$i$轉換至產品$j$的成本，$i \in I,~j \in I$  
$C^S_{i}$：產品$i$的倉儲成本，$i \in I$  
$C^L_{i}$：產品$i$的延遲交貨成本，$i \in I$  
$V_{i}$：產品$i$的單位毛利，$i \in I$  
$S^I_{i}$：產品$i$的初始倉儲，$i \in I$  
$X_{if}$：在時間週期$f$對產品$i$的定案生產規劃，$i \in I, f \in F \cup\{P_{-1}\}$  
$D^*_{ip}$：在時間週期$p$結束前需求產品$i$的量，$i \in I,~p\in P$  
$A^*_{i}$：產品$i$一個時間週期的生產量，$i \in I$  
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
銷售產品毛利 = $\displaystyle{\sum_{i \in I}\sum_{p \in P}V_{i}(D^*_{ip}-l_{ip})}$  
倉儲成本 = $\displaystyle{\sum_{i \in I}\sum_{p \in P}C^S_i s_{ip}}$  
缺貨成本 = $\displaystyle{\sum_{i \in I}\sum_{p \in P}C^L_{i} l_{ip}}$  
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