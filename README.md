Problem Motivation

Distributed denial of service attacks generate large volumes of abnormal network traffic that overwhelm services and infrastructure. Traditional detection methods rely on static thresholds or supervised classifiers that may struggle to adapt to evolving traffic patterns. Reinforcement learning provides a framework where an agent learns decision policies through interaction with data and reward feedback, enabling adaptive detection strategies.

Dataset and Preprocessing

The system uses the CICDDoS2019 network traffic dataset which contains labeled network flow records representing both benign activity and multiple DDoS attack types. Numeric flow features are extracted and missing values are removed. To ensure that features contribute comparably during analysis, the data is standardized using z score normalization where each feature is transformed to have zero mean and unit variance.

Dimensionality Reduction with PCA

Network flow datasets often contain many correlated features. Principal Component Analysis transforms the standardized feature space into a smaller set of orthogonal components that capture the dominant variance of the data.

The covariance matrix of the centered data matrix $X$ is

Σ
=
1
𝑛
𝑋
𝑇
𝑋
Σ=
n
1
	​

X
T
X

Principal components are obtained by solving the eigenvalue problem

Σ
𝑣
𝑖
=
𝜆
𝑖
𝑣
𝑖
Σv
i
	​

=λ
i
	​

v
i
	​


where
$v_i$ represents the eigenvectors and
$\lambda_i$ represents the variance explained along each direction.

The dataset is projected into the reduced space as

𝑍
=
𝑋
𝑉
𝑘
Z=XV
k
	​


where $V_k$ contains the top $k$ eigenvectors.

Because the covariance matrix is symmetric, its eigenvectors are orthogonal. This means each principal component captures an independent direction of variance in the data.

Quantum Entropy Feature Extraction

Each PCA feature vector $v$ is normalized and interpreted as a state vector. A density matrix representation is constructed as

𝜌
=
𝑣
𝑣
𝑇
ρ=vv
T

The von Neumann entropy of the density matrix is then computed

𝑆
(
𝜌
)
=
−
Tr
(
𝜌
log
⁡
𝜌
)
S(ρ)=−Tr(ρlogρ)

This entropy value represents the uncertainty or disorder in the state representation derived from the network flow features.

Q Learning Update Rule

The reinforcement learning agent updates its action value estimates using

𝑄
(
𝑠
,
𝑎
)
←
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)←Q(s,a)+α[r+γ
a
′
max
	​

Q(s
′
,a
′
)−Q(s,a)]

Reinforcement Learning Formulation

The classification problem is modeled as a sequential decision process. Each state consists of discretized features derived from entropy values and the leading principal component. The agent selects an action representing a prediction of benign or malicious traffic. The environment provides feedback through a reward signal that reflects prediction correctness.

Q Learning Training Procedure

The agent learns a policy using the Q learning update rule

𝑄
(
𝑠
,
𝑎
)
=
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)=Q(s,a)+α[r+γ
a
′
max
	​

Q(s
′
,a
′
)−Q(s,a)]

where 
𝑠
s represents the current state, 
𝑎
a the selected action, 
𝑟
r the reward, and 
𝑠
′
s
′
 the next state. The learning rate 
𝛼
α controls the update magnitude while the discount factor 
𝛾
γ determines the importance of future rewards. An epsilon greedy policy balances exploration and exploitation during training.

Evaluation and Observations

Training progress is monitored by tracking cumulative reward per episode. Increasing reward over time indicates that the agent is learning to correctly classify traffic patterns while accounting for entropy based uncertainty in the state representation.
