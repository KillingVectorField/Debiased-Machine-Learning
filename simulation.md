### Model

$$
Y=D\theta_0+(DX^*)^T\eta_0+g_0(X)+U
$$
$$
D=m_0(X) + V
$$
$$
E[U|X, D]=0, E[V|X]=0, E[UV|X]=0
$$

Neyman Orthogonal Score:
$$
\psi(X;\beta,\gamma) = (Y-A^T\beta-E[Y-A^T\beta|X])(A-E[A|X])
$$
where $A^T = (D, D{X^*}^T), \beta^T=(\theta, \eta^T)$ and $\gamma$ denotes nuisance parameters.

#### DML1

Let $\hat{h_k}(x)$ and $\hat{l_k}(x)$ denote the ML estimators for $A$ and $Y$ respectively from $I_k^c$. For each fold $I_k$
$$
\hat{\beta_k} =\left(\sum_{i\in I_k}\left[a_i-\hat{h_k}(x_i)\right]\left[a_i-\hat{h_k}(x_i)\right]^T\right)^{-1}\left(\sum_{i\in I_k}\left[y_i-\hat{l_k}(x_i)\right]\left[a_i-\hat{h_k}(x_i)\right]\right)\\
\hat{\beta} = \frac{1}{K}\sum_{k=1}^{K}\hat{\beta_k}
$$

#### DML2

$$
\hat{\beta} =\left(\sum_{k=1}^{K}\sum_{i\in I_k}\left[a_i-\hat{h}_k(x_i)\right]\left[a_i-\hat{h}_k(x_i)\right]^T\right)^{-1}\left(\sum_{k=1}^{K}\sum_{i\in I_k}\left[y_i-\hat{l}_k(x_i)\right]\left[a_i-\hat{h}_k(x_i)\right]\right)
$$

#### Covariance

$$
\begin{align}
\hat{J_0}=\frac{1}{n}\sum_{k=1}^{K}\sum_{i\in I_k}[a_i-\hat{h}_k(x_i)][a_i-\hat{h}_k(x_i)]^T
$$
$$
\hat{\sigma}^2=\hat{J_0}^{-1}\frac{1}{n}\sum_{k=1}^{K}\psi(x_i;\hat{\beta},\hat{h}_k, \hat{l}_k)\psi(x_i;\hat{\beta},\hat{h}_k, \hat{l}_k)^T[{\hat{J_0}^{-1}}]^T

$$

### Simulation Design

$$
\begin{align}
&y_i = d_i + (d_ix_i^*)'\mathbf{1}+x_i'(c_y\theta_0)+u_i\\
&d_i = \frac{\exp\{x_i'(c_d\theta_0)\}}{1+\exp\{x_i'(c_d\theta_0)\}} +v_i
\end{align}
$$

where $d_ix_i^*$ denotes the interaction term.

