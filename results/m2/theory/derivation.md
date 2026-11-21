# Mean-field theory of Model 2 (M2): derivation

Lane m2/theory, part "meanfield". Written 2026-11-09. Code: `code/m2_mf_core.py` (map, closures, linearisation),
`code/m2_mf_run.py` (GPU runs), `code/m2_mf_glin.py`, `code/m2_mf_chain.py`, `code/m2_mf_analyze.py`,
`code/m2_mf_predict.py`. Pre-registration: `plan_meanfield.md`. Predictions with their numbers:
`meanfield_predictions.json`. All formulas use the notation of `results/m2/MODEL2.md`. Notation:
- $P = 1 - e^{-\Delta t} = 0.18127$ is the off-probability per step ($\Delta t = 0.2$);
- $\alpha \equiv \alpha_\mathrm{eff} = P/\Delta t = 0.9063$;
- $q = e^{-\Delta t/\tau}$ is the per-step resource recovery factor, with $\tau$ = TAU;
- $f(s) = 1 - \exp(-\beta \tanh(s)_+ \Delta t)$ is the on-probability per step at input $s$ ($\beta = 1$).

---------------------------------------------------------------------------------------------------------------------

## 1. One neuron, exactly: the per-spike Tsodyks–Markram resource in the synchronous update

**The update (code/ald_sim.py, `--std_mode spike`).** In each step, using the old state:
- the input is $s_i = g\sum_j W_{ij} a_j e_j + h_i$;
- an active unit turns off with probability $P$, and a quiescent unit turns on with probability $p_i = f(s_i)$;
- every resource recovers: $x \to \tilde x = 1 - q(1-x)$;
- at an onset ($0 \to 1$) the epoch's efficacy is set to $e \leftarrow \tilde x$, and then $x \leftarrow (1-U)\tilde x$.

The output of $j$ during an epoch is the constant $e_j$.

**Closed moment map.** Take one neuron with a given (deterministic) on-probability sequence $p(t)$. Its four first moments
$$\rho = E[a],\qquad y = E[a\,e],\qquad m_1 = E[a\,x],\qquad m_0 = E[(1-a)\,x]$$
obey a closed linear map, because an onset depends only on $p$ and not on $x$:
$$A = q\,m_1 + (1-q)\rho,\qquad B = q\,m_0 + (1-q)(1-\rho),$$
$$\rho' = (1-P)\rho + p(1-\rho),\qquad y' = (1-P)\,y + p\,B,$$
$$m_1' = (1-P)A + p(1-U)B,\qquad m_0' = P A + (1-p) B .$$
Here $A = E[a\tilde x]$ and $B = E[(1-a)\tilde x]$. The map is exact for one unit.
- Test T2 (`tests_meanfield.json`): a Monte Carlo of ald_sim.py itself (20,000 isolated units, decay start and a
  switched drive with clamps) matches it to $\le 0.1\%$ in $\rho$, in the efficacy of active units and in the mean resource,
  with $|z| \le 1.4$.
- For a non-depressing neuron ($U = 0$, $q = 1$) the map gives $y = \rho$, $m_1 = \rho$, $m_0 = 1 - \rho$ exactly.

**Stationary efficacy (renewal).** Hold $p$ constant. The epochs ($L \ge 1$ steps) and the gaps ($Q \ge 1$ steps) are
geometric with parameters $P$ and $p$, and they are independent of the resource. Let $d = 1 - e$ be the deficit at an onset.
Between two onsets $d' = q^{L+Q}\,(U + (1-U)d)$. So $E[d] = MU/(1-(1-U)M)$, with $M = E[q^L]\,E[q^Q]$, and
$$\bar e(p) = \frac{1 - M}{1 - (1-U)M},\qquad E[q^L] = \frac{Pq}{1-(1-P)q},\quad E[q^Q] = \frac{pq}{1-(1-p)q}.$$
The stationary point of the moment map gives the same value ($y^*/\rho^* = \bar e$ to $10^{-10}$, test T1).

**Poisson limit: the "x* = 1/(1 + U τ ρ)" form, derived for the synchronous update.**
- For $\Delta t \ll \tau$, $1 - M \approx (\Delta t/\tau)E[L+Q]$, and $E[L+Q] = 1/P + 1/p = 1/(P\rho)$ (since $\rho = p/(p+P)$).
- Hence
$$\bar e \approx \frac{1}{1 + U\,\tau\,\alpha\,\rho} = \frac{1}{1 + U\,\nu\,\tau},\qquad \nu = \alpha\rho = \text{onsets per model unit}.$$
- In model units: one epoch is one spike, the epoch rate is $\alpha\rho$, and the recovery time is $\tau$ units.
- The exact discrete formula differs from this limit by 0.4–0.5% at $\rho \le 0.05$ for $\tau = 157$, and by 1.7% at
  $\rho = 0.14$ for $\tau = 30$ (T1).

**Transmission function and its scale.**
- The mean transmitted output of a depressing neuron is $\phi(\rho) = \rho\,\bar e(\rho)$.
- In the Poisson form $\phi \approx \rho/(1 + \rho/\rho_\mathrm{dep})$, with
$$\rho_\mathrm{dep} = \frac{1}{U\tau\alpha}.$$
- M2: $\rho_\mathrm{dep} = 0.0281$. KW09: 0.0164. N15: 0.0564. TAU 30: 0.147.
- So an excitatory neuron's output saturates at about $\rho_\mathrm{dep}$, whatever its own activity. At the maximal
  $\rho = 0.5$ ($p = P$), M2 has $\phi_\max = 0.027$, against 0.5 without depression.
- Examples (M2, exact): $\bar e = 0.41$ at $\rho = 0.04$ (an isolated ORN), 0.22 at 0.1, 0.053 at 0.5
  (`meanfield_analytic.json`).

## 2. The exact linearisation about the silent state: g_lin is unchanged

In the sparse limit, the on-probability of $i$ when only $j$ is active is $f(g E_{ij} e_j)$, where $E$ is the
excitatory part of $W$. Inhibitory inputs contribute 0 at first order because of the rectifier. So the exact
(Möbius, first-order) branching operator about silence, with the resources frozen at their current values, is
$$B_{ij}(g; x) = f(g E_{ij} x_j)/\Delta t,$$
where $x_j$ is the resource that $j$'s next onset will use. Three consequences:
1. **Rested silence.** At rest $x \equiv 1$, so $B(g;1)$ is M1's operator, and $g_\mathrm{lin}$, defined by
   $\lambda_{PF}(B(g_\mathrm{lin})) = \alpha$, is **exactly M1's: 0.16315**.
   - Numerical check (T6): $\lambda_{PF}(B(0.16315))/\alpha = 0.99998$.
   - g_lin for the variants and nulls is in `meanfield_glin.json`.
   - Spike-mode depression only acts after a first onset. The old rate mode instead multiplied $B$ by
     $\varphi \approx 0.8$.
2. **Uniformly depleted resources.** If every resource is frozen at the same value $x$, then $B(g;x) = B(gx;1)$, because
   $f$ depends on $g$ and $x$ only through their product. So **$g_\mathrm{lin}(x) = g_\mathrm{lin}/x$ exactly**.
   Numerical check: $\sigma(2g_\mathrm{lin}, x = 0.5) = 1$ (`meanfield_glin.json`).
3. **No r dependence.** Inhibition enters only at second order, through coincidences of an excitatory and an inhibitory
   input in the same step. $g_\mathrm{lin}$ is therefore independent of $r$, as in M1.

## 3. The network mean field with resources

**The map.** Each neuron carries $(\rho_i, y_i, m_{1,i}, m_{0,i})$ and is updated with the map of §1, with $p_i = f(\bar s_i)$.
Two closures for $\bar s_i$, the only approximation:
- **QMF** (M1's `theory_qmf.py` closure, mean input): $\bar s_i = g\sum_j W_{ij}\,y_j + h_i$.
  - Independence and Jensen: $E f(s) \to f(E s)$.
  - With $U = 0$ this is exactly M1's map (test T4).
- **CQMF** (coincidence-corrected; new). Write $s = A - gS_I$ with $A = h + gS_E \ge 0$ ($h \ge 0$ the external drive) and
  $S_E, S_I \ge 0$. Then $E[s_+] = E[A] - E[\min(A, gS_I)]$. For independent $A$ and $S_I$,
  $E[\min(A, gS_I)] \le E[A\,1(S_I > 0)] = \bar A\,\pi_I$, and likewise $\le g\mu_I\,P(A > 0)$. CQMF uses
  $$\bar s_i = \bar A_i - \min\big(\bar A_i\,\pi_{I,i},\ g\mu_{I,i}\,\pi_{A,i}\big),\qquad \bar A_i = h_i + g\mu_{E,i},$$
  $$\pi_{X,i} = 1 - \prod_{j \in X(i)}(1 - \rho_j),\qquad \pi_{A,i} = 1 \text{ if } h_i > 0 \text{ else } \pi_{E,i},\qquad \mu_{E,i} = \sum_j E_{ij} y_j,\qquad \mu_{I,i} = \sum_j |W^-_{ij}|\,y_j .$$
  (Correction C1 in `plan_meanfield.md`: the first version omitted $h$ from $A$, so driven neurons could not be inhibited.
  Only the driven runs were affected, and they were redone.)
  Properties:
  - Its drive is a lower bound on $E[(S_E - S_I)_+]$ that is tighter than QMF's $(\mu_E - \mu_I)_+$.
  - It equals QMF at $r = 0$ and when activity is dense ($\pi \to 1$).
  - Near silence (h = 0) its inhibition is of second order ($\mu_E\pi_I = O(\rho^2)$), as in the exact linearisation.
  - QMF instead subtracts the full mean inhibition at any activity, which is why M1's QMF silent edge moved with $r$
    (0.159 → 0.198) while the true $g_\mathrm{lin}$ does not.
- Which closure is primary was decided on M1 before any M2 number was looked at (validation V1, section 9).

**Fixed points.** $\rho_i = p_i/(p_i + P)$ and $y_i = \rho_i\,\bar e_i(p_i)$, with $p_i = f(\bar s_i[y])$. A depressing
neuron feeds the network $\phi(\rho_i) \le \phi_\max$ (§1).

**Linear stability.** The Jacobian of the map at a state (exact JVP for QMF, `M.jvp`; central differences for CQMF;
both tested, T5). Per neuron, with $\delta p_i = f'(\bar s_i)\,\delta\bar s_i$:
$$\delta\rho' = (1-P-p)\,\delta\rho + (1-\rho)\,\delta p,\qquad \delta y' = (1-P)\,\delta y + p\,\delta B + B\,\delta p,$$
$$\delta m_1' = (1-P)\,\delta A + (1-U)(p\,\delta B + B\,\delta p),\qquad \delta m_0' = P\,\delta A + (1-p)\,\delta B - B\,\delta p,$$
with $\delta A = q\,\delta m_1 + (1-q)\,\delta\rho$ and $\delta B = q\,\delta m_0 - (1-q)\,\delta\rho$.
- The rectifier is linearised with its right derivative at $\bar s = 0$ (conservative).
- Resources of silent neurons ($p = 0$) decouple, with eigenvalue $q$, and never feed back.
- The spectrum near 1 is therefore crowded by $\approx 10^5$ slow resource modes. So the slow modes that the
  population activity shows are measured by a **ring-down**: propagate a kick through the linear map, then fit damped
  modes to the population sums by the matrix pencil (`m2_mf_analyze.modes`).

## 4. Near the onset, depression shrinks the self-sustained amplitude by a factor of about 36

Take a one-mode ansatz $\rho = A v$ with left vector $u$ and the rectified Perron root $\sigma_0 = g\mu/\alpha$, and keep the
leading nonlinearity. M1's leading saturations (from $1-\rho$ and from $f$) are $O(1)$ per unit activity. Depression
adds $\bar e \approx 1 - \rho/\rho_\mathrm{dep}$, which is 36 times larger. The fixed point
$$A^* \approx \frac{\sigma_0 - 1}{\sigma_0}\;\frac{\langle u, v\rangle}{\langle u, v^2\rangle}\;\Big(\rho_\mathrm{dep}^{-1} + c_{M1}\Big)^{-1}$$
is supercritical (continuous), with a slope about $\rho_\mathrm{dep}\,c_{M1} \approx 1/36$ of M1's. The mean field
therefore still has a continuous onset at the silent edge. But its sustained state just above the edge is very small, and a
small state is exactly what fluctuations extinguish. This is the mean-field reason the M2 onset lies far above $g_\mathrm{lin}$.

## 5. Slow-fast (activity–resource) reduction: no Hopf bifurcation from depression alone

Let $n$ be the mode activity and $X$ the mode-weighted efficacy, with $\sigma_0 = \lambda(g)/\alpha$ the undepressed branching
ratio. In continuous time (the corrections are $O(\Delta t/\tau)$):
$$\dot n = \alpha(\sigma_0 X - 1)\,n - \alpha c\,n^2,\qquad \dot X = \frac{1 - X}{\tau} - U\alpha\,n X .$$
- **Fixed point.** $X^* = (1 + c n^*)/\sigma_0$ and $n^* = \rho_\mathrm{dep}\big(\sigma_0/(1 + cn^*) - 1\big) \approx \rho_\mathrm{dep}(\sigma_0 - 1)$.
- **Jacobian.**
$$J = \begin{pmatrix} -\alpha c n^* & \alpha\sigma_0 n^* \\ -U\alpha X^* & -(1/\tau)(1 + n^*/\rho_\mathrm{dep}) \end{pmatrix},$$
  with $\operatorname{tr}J = -\alpha c n^* - \sigma_0/\tau < 0$ and $\det J > 0$.
- **So the fixed point is always stable: depression alone gives no Hopf bifurcation** in this reduction. It gives a
  stable focus.
  - Its frequency satisfies $\omega_0^2 \approx \alpha(\sigma_0 - 1)/\tau$, and its damping rate is $\gamma = \sigma_0/(2\tau) + \alpha c n^*/2$.
  - The period is $P_\mathrm{osc} = 2\pi/\sqrt{\omega_0^2 - \gamma^2}$; with $c = 0$, $P_\mathrm{osc} \approx 2\pi\sqrt{\tau/(\alpha(\sigma_0-1))}$.
    For M2: 185, 117, 83, 59, 41 and 27 units at $\sigma_0$ = 1.2, 1.5, 2, 3, 5 and 10.
  - The damping per period, $C(2P)/C(P) = e^{-\gamma P_\mathrm{osc}}$ for a single mode, is largest at $\sigma_0 = 2$ and is
    bounded by
$$e^{-\gamma P_\mathrm{osc}} \le \exp\!\big(-2\pi/\sqrt{\alpha\tau}\big) = 0.59\ (\text{M2}),\ 0.65\ (\text{KW09}),\ 0.50\ (\text{N15}),\ 0.30\ (\tau = 30).$$
  - Any saturation ($c > 0$) or network heterogeneity only adds damping.
- **Prediction: depression cannot by itself produce an OSC-strong rhythm** (MODEL2 threshold 0.7). That would need
  $\tau \ge 4\pi^2/(\alpha \ln^2(1/0.7)) = 342$ model units.
- Limit cycles in the network mean field need another mechanism, such as the E–I loops behind M1's non-stationary QMF
  at $r \ge 0.5$ (pd3 MT3).

## 6. The crash after a fully active start (setting a)

- **Setup.** At $t = 0$ every unit is active with $e = 1$ and $x = 1-U$. Within a few tens of units the high activity
  depletes the resources of the depressing neurons to $x_c \approx \bar e(\rho\sim 0.3\text{–}0.5) \approx 0.05\text{–}0.1$.
- **Decay and recovery.** Activity then decays at the rate $\alpha(1 - \sigma_0 x(t))$, while the resources recover as
  $x(t) = 1 - (1 - x_c)e^{-t/\tau}$.
- **Depth of the trough.** The one-mode count falls to a minimum at $\sigma_0 x(t_m) = 1$, i.e. at
  $t_m = \tau\ln\!\big(\sigma_0(1-x_c)/(\sigma_0-1)\big)$, with depth
$$\ln\frac{n_0}{n_\min} = \alpha\tau\Big[1 - \sigma_0 x_c - (\sigma_0 - 1)\ln\frac{\sigma_0(1-x_c)}{\sigma_0 - 1}\Big]\qquad(\sigma_0 x_c < 1).$$
- **Extinction during the crash** happens if $n_\min < 1$, which requires $\ln n_0 \lesssim 10$ against $\alpha\tau = 142$.
  - With $\ln n_0 \approx 10$ the crash alone kills activity up to $\sigma_0 \approx 4.7$ ($x_c = 0.05$) or 3.7 ($x_c = 0.1$) (table in `meanfield_analytic.json`).
  - At $\tau = 30$ the bracket is multiplied by $\alpha\tau = 27$ instead of 142, so the crash kills far less. This is the
    mean-field reason that the shorter recovery time gives a smaller onset shift (aldef: ×2.5 instead of ×3.3–4.8).
- **In the network.** This is computed without the one-mode reduction: from the decay start, with $n(t) = \sum_i\rho_i(t)$ and
  a cutoff $n_c = 1$ (0.1 and 10 as sensitivity).
  - The mean field cannot see fluctuation-driven extinction of a small state that survives the crash.
  - **Hence the mean-field S and onset lines (§8) are lower bounds** for the simulated ones. The bound uses
    $P(\text{alive at } t) \le E[n(t)]$, and is expected to hold for CQMF; for QMF at $r > 0$, over-inhibition can break it.

## 7. The branching chain with resources (theory_branch extended)

- **Inputs.** M1's onset theory uses one mode $(u, v)$ of $B(g)$ and a logistic birth–death chain $b(n) = \lambda n(1 - n/K)$,
  $d(n) = \alpha n$, with $K = K_\mathrm{eff}(g, r)$. Resources multiply the birth rate by the mode-weighted efficacy
  $X(n) = \sum_j u_j v_j\,\bar e_j(\rho_j = n\langle u^2,v\rangle v_j)/\sum_j u_j v_j$.
- **Two limits.**
  - Frozen: $X = X^* = X(n^*)$ with $\lambda X(n^*)(1-n^*/K) = \alpha$. The resource is slow compared with the escape.
  - Adiabatic: $X = X(n)$.
- **Lifetime.** The exact mean extinction time from the top state; S and onset at $\tau = 200$ and 3000.
- **K.** $K = \kappa_r K_\mathrm{eff}$, with $\kappa_r$ fitted per $r$ so that the undepressed chain reproduces the M1 simulated
  onset (the aldef procedure, now for all 23 $r$). $\kappa = 1$ is reported too.
- **Caveat.** The mode is M1's leading mode, the antennal-lobe (AL) module. So the chain gives the lifetime of the AL mode
  alone. When other modes sustain activity first (aldef: with depression the first sustained state is brain-wide), the
  network onset lies below the chain's. **The chain onset is an upper bound on the network onset** (if the AL mode alone
  sustains, so does the network); the cutoff mean field (§6) is a lower bound.

## 8. Driven setting (b)

- **Protocol.** The ORN drive $h_\mathrm{ORN}$ is applied to all 2,639 ORNs from a quiet start, and the ORNs are clamped
  off at $T_1 = 1500$. The map is the same, with $p_\mathrm{ORN} = 0$ after the clamp; the clamp moves the active units'
  resources to the quiescent pool.
- **Measured ORN rate.** The mean field captures the drop of the ORN rate below the isolated 0.04: recurrent inhibition
  reaches the ORN terminals (MODEL2 §1).
- **Gain.** $G = \rho_\mathrm{uPN}/\rho_\mathrm{ORN}$ is a direct prediction.
- **Removal.** The removal class uses the same cutoff as §6 on $n_\mathrm{nonORN}$, and $S_1$ and $S_2$ as in MODEL2 §5b.
  So the DA→DT and DT→SS lines are lower bounds, for the same reason as in §6.
- **Bistability.** The quiet start is compared with a fully active start under the same drive.

## 9. Assumptions, in one list

1. **Independence.** Independence of inputs, and $E f \to f(\cdot)$ of the closure's drive, i.e. the mean-field closure.
   QMF over-inhibits at $r > 0$; CQMF bounds the inhibition by coincidence probabilities, but still assumes independent inputs.
2. **No fluctuations in the network mean field.** Extinction enters only through the cutoff $n_c$, so the S and onset lines
   are lower bounds. The chain (§7) adds fluctuations of one mode only.
3. **Rectifier.** Its derivative at exactly zero input is taken one-sided.
4. **One-mode reductions (§4–6).** These are for orientation only. Every network number comes from the full
   167,216-neuron map on the simulator's operator (built with ald_sim.py's own functions; nnz 24,562,483 at $r = 1$, test T3).
5. **Electrical coupling, adaptation and facilitation** are absent, as in the model.
6. **Time mapping.** 1 unit = 10 ms only matters for the variants' TAU values (MODEL2).
