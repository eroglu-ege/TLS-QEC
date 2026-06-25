# Jaynes-Cummings Model: Qubit + Linear Resonator

## 1. What is this and why does it differ from the TLS model?

In the qubit+TLS model we built earlier, both systems are two-level:

```
qubit+TLS:       H_q ⊗ H_TLS     →  4-dim Hilbert space
```

In the Jaynes-Cummings (JC) model, the second system is a **harmonic oscillator** (linear resonator) with infinitely many levels, truncated at `N_max` photons:

```
qubit+resonator: H_q ⊗ H_res     →  2*(N_max+1) dim Hilbert space
```

The resonator can hold 0, 1, 2, ... photons. This is physically closer to a real microwave resonator or transmission line in a circuit QED experiment.

---

## 2. The Hamiltonian

$$H = \frac{\omega_q}{2}\sigma_z + \omega_r a^\dagger a + g(a\sigma_+ + a^\dagger\sigma_-)$$

| Term | Operator | Physical meaning |
|------|----------|-----------------|
| $\frac{\omega_q}{2}\sigma_z$ | Pauli Z on qubit | Qubit energy splitting |
| $\omega_r a^\dagger a$ | Photon number | Resonator energy |
| $g\cdot a^\dagger\sigma_-$ | Creates photon, lowers qubit | Qubit emits into resonator |
| $g\cdot a\sigma_+$ | Destroys photon, raises qubit | Resonator re-excites qubit |

The exchange interaction conserves total excitation number $N_{exc} = \frac{1}{2}(\sigma_z + 1) + a^\dagger a$ — exactly as in the TLS model, but now the resonator side can hold multiple photons.

---

## 3. Initialization and the fast limit

The code sets the initial state instantaneously at $t=0$:

```python
psi0 = |e, 0>  =  qubit excited  ⊗  resonator vacuum
```

This represents the **fast initialization limit**: a $\pi$-pulse was applied to the qubit on a timescale $t_{pulse} \ll 1/g$, so the resonator had no time to respond. In circuit QED this is achievable since pulse durations (~10 ns) are much shorter than $1/g$ (~100 ns for typical couplings).

If initialization were slow ($t_{pulse} \sim 1/g$), photons would build up in the resonator during the pulse itself — requiring a driven, time-dependent Hamiltonian.

---

## 4. How photons are created

Starting from $|e, 0\rangle$, the interaction term $g \cdot a^\dagger\sigma_-$ acts as:

$$a^\dagger\sigma_- |e, 0\rangle = |g, 1\rangle$$

The qubit falls from $|e\rangle$ to $|g\rangle$ while the resonator jumps from vacuum $|0\rangle$ to one photon $|1\rangle$. The resonator then leaks this photon to the transmission line at rate $\kappa$:

```
t=0:    |e, 0>   qubit excited, resonator empty
         ↓ g
t~1/g:  |g, 1>   photon created in resonator
         ↓ κ
t~1/κ:  |g, 0>   photon leaks out → ADC signal
```

The **emitted photon flux** is $\kappa\langle a^\dagger a\rangle(t)$ — what an ADC or homodyne detector measures.

---

## 5. Dissipation channels

| Jump operator | Rate | Physical process |
|--------------|------|-----------------|
| $\sqrt{\gamma(1+n_{th}^q)}\,\sigma_-$ | qubit decay | $T_1$ relaxation |
| $\sqrt{\gamma\,n_{th}^q}\,\sigma_+$ | thermal excitation | bath drives qubit up |
| $\sqrt{\kappa(1+n_{th}^r)}\,a$ | photon loss | resonator emits to ADC |
| $\sqrt{\kappa\,n_{th}^r}\,a^\dagger$ | thermal photon injection | bath adds photons |
| $\sqrt{\gamma_\phi/2}\,\sigma_z$ | pure dephasing | $T_2 < 2T_1$ |

The key new channel vs the TLS model: $\sqrt{\kappa}\,a$ — the resonator decaying by emitting a photon to the outside world. $\kappa$ plays the role of $\gamma_t$ from the TLS model.

---

## 6. Tracked observables

| Observable | Formula | Physical meaning |
|------------|---------|-----------------|
| `P_e_q` | $(\langle\sigma_z\rangle + 1)/2$ | Qubit excited state probability |
| `n_photons` | $\langle a^\dagger a\rangle(t)$ | Mean photons inside resonator |
| `photon_flux` | $\kappa\langle a^\dagger a\rangle$ | Photons/time leaving resonator |
| `photons_emitted` | $\int_0^t \kappa\langle n\rangle\,dt'$ | Total photons emitted |
| `coherence_qr` | $\lvert\langle a\sigma_+\rangle\rvert$ | Qubit-resonator coherence |
| `Purcell_rate` | $g^2\kappa/(\Delta^2 + (\kappa/2)^2)$ | Enhanced decay rate (analytic) |

**Why cumulative photons matters:** should approach 1.0 at long times for $|e,0\rangle$ initial state with no thermal effects. Deviations indicate energy leakage.

---

## 7. Physical regimes

### Bad cavity / Purcell ($g/\kappa \ll 1$)

Resonator leaks faster than qubit fills it. Qubit sees an additional decay channel:

$$\gamma_{\text{Purcell}} = \frac{g^2\kappa}{\Delta^2 + (\kappa/2)^2}$$

ADC sees: single clean exponential pulse. Solomon is exact here.

### Threshold ($g/\kappa \sim 1$)

Coupling and loss timescales comparable. Onset of non-exponential photon flux. Solomon begins to break down — same physics as $g/\gamma_t \sim 1$ in the TLS model.

### Strong coupling ($g/\kappa \gg 1$)

Multiple qubit-resonator exchanges before photon escapes. Photon flux shows **oscillations** at vacuum Rabi frequency $\Omega_R \approx g$. ADC sees multiple photon bursts. Solomon fails completely.

---

## 8. Connection to Solomon breakdown

| TLS model | JC model | Role |
|-----------|----------|------|
| TLS | Resonator | Second quantum system |
| $\gamma_t$ | $\kappa$ | Decay rate of second system |
| $g/\gamma_t$ | $g/\kappa$ | Key dimensionless ratio |
| Solomon valid: $g/\gamma_t \lesssim 0.5$ | Solomon valid: $g/\kappa \lesssim 0.5$ | Same threshold |

The Solomon equations describe the bad-cavity / Purcell regime. The full Lindblad JC model captures the crossover to strong coupling.

---

## 9. N_max truncation warning

Fock space is truncated at `N_max`. Rule of thumb: `N_max >= 3 * max(<a†a>)`. The code warns if `n_photons` approaches `N_max`.

---

## 10. Code structure

```
jaynes_cummings.py
├── build_operators(N_max)     — qubit ⊗ resonator operators
├── build_H_JC(wq, wr, g)      — Jaynes-Cummings Hamiltonian  
├── initial_state(qubit, n)    — product state |qubit, n_photons>
├── purcell_rate(g, kappa, Δ)  — analytic Purcell rate
└── evolve(...)                — full master equation solver
```

Import in analysis scripts:
```python
from qubit_tls.jaynes_cummings import evolve, purcell_rate
```
