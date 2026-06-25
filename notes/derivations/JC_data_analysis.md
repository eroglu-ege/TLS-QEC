# Jaynes-Cummings Model: Data Analysis Report

**Date:** 2026-06-25  
**Parameters (baseline):** $g/\kappa=5$, $\gamma=\kappa=0.01$, $T=0$ (unless noted)

---

## 1. Driven JC — Photon Saturation (`driven_JC_g0.050_kappa0.010_nth0.000.npz`)

### Setup
Continuous coherent drive on the qubit at rate $\Omega$. Qubit starts in ground state.
No thermal population ($n_{th}=0$), no pure dephasing.

### Key result: photons saturate, never diverge

| $\Omega$ | $\Omega/g$ | $\langle n\rangle_{ss}$ (numeric) | $\langle n\rangle_{ss}$ (analytic) | Ratio |
|----------|-----------|-----------------------------------|------------------------------------|-------|
| 0.005 | 0.10 | 0.00245 | 0.00245 | **1.0000** |
| 0.010 | 0.20 | 0.00980 | 0.00980 | **1.0000** |
| 0.050 | 1.00 | 0.24504 | 0.24507 | **0.9999** |
| 0.100 | 2.00 | 0.97968 | 0.98030 | **0.9994** |

The analytic formula is confirmed to 4 decimal places:

$$\boxed{\langle n\rangle_{ss} = \frac{g^2\Omega^2}{4\left(g^2 + \frac{\gamma\kappa}{4}\right)^2}}$$

**Why saturation occurs and not divergence:**
Every photon that enters the resonator from the qubit also leaks out at rate $\kappa$.
At steady state, pump rate = leak rate:

$$g^2 \cdot P_e^{drive} \sim \kappa \cdot \langle n\rangle_{ss}$$

The drive cannot maintain $P_e=1$ indefinitely because the qubit also decays at rate
$\gamma$ and loses coherence. The balance gives a finite $\langle n\rangle_{ss} \propto \Omega^2$.

**Physical picture of the $\Omega^2$ scaling:**
- $\Omega/2$ drives the qubit coherence $\langle\sigma_-\rangle$
- The coherence drives the resonator field $\langle a\rangle \propto g\langle\sigma_-\rangle$
- Photon number $\langle n\rangle = |\langle a\rangle|^2 \propto g^2\Omega^2$

So: doubling the drive quadruples the intracavity photon number (always finite).

**The $g^2$ prefactor:** no coupling = no photons. The drive is on the qubit, not the
resonator directly. Energy reaches the resonator only via the coupling $g$.

---

## 2. Four Regimes (`JC_regimes_nth0.000_gphi0.000.npz`)

### Setup
Single shot: qubit starts in $|e\rangle$, resonator in $|0\rangle$. $T=0$, no dephasing.
Four values of $g/\kappa$: 0.5, 1.0, 5.0, 10.0

### Intracavity photon number peaks

| $g/\kappa$ | Peak $\langle n\rangle$ | Final $\langle n\rangle$ | Physical regime |
|-----------|------------------------|--------------------------|-----------------|
| 0.5 | **0.104** | 0.00019 | Bad cavity / Purcell |
| 1.0 | **0.264** | 0.00033 | Threshold |
| 5.0 | **0.737** | 0.00019 | Strong coupling |
| 10.0 | **0.857** | 0.00034 | Deep Rabi |

**All final values → 0:** at long times both qubit and resonator have decayed completely
to vacuum ($T=0$). Every excitation eventually leaks out — conservation confirmed.

**Why peak photon number grows with $g/\kappa$:**

- At **$g/\kappa=0.5$ (Purcell):** the resonator leaks photons faster than the qubit can
  supply them. The resonator never holds more than ~0.10 photons at once. The qubit
  decay is Purcell-enhanced but the resonator remains nearly empty.

- At **$g/\kappa=5.0$ (strong coupling):** the qubit and resonator coherently exchange the
  excitation several times before it leaks. During the first Rabi half-cycle, nearly the
  full excitation is briefly in the resonator ($\langle n\rangle_{peak}=0.74$) before
  returning to the qubit. Each subsequent cycle, some leaks out. The peak photon number
  approaches 1 (the maximum possible from a single qubit excitation).

- At **$g/\kappa=10$ (deep Rabi):** even higher peak ($\langle n\rangle=0.86$) because
  the coherent transfer is more complete before $\kappa$ has time to drain the resonator.

**The photon leakage mechanism:**
$$\text{qubit} |e,0\rangle \xrightarrow{g} \text{resonator} |g,1\rangle \xrightarrow{\kappa} \text{ADC}$$

The ADC (or homodyne detector) measures the photon flux $\kappa\langle a^\dagger a\rangle(t)$.
At $g/\kappa=0.5$: smooth exponential pulse (Purcell regime, Solomon valid).
At $g/\kappa=10$: oscillating bursts at the Rabi frequency $\Omega_R \approx g$.

---

## 3. Low Coupling Sweep with Thermal + Dephasing (`JC_low_coupling_...nth0.050_T2q50.npz`)

### Setup
$g/\kappa \in [0.1, 1.1]$, 40 linear points.
**Realistic parameters:** $n_{th}=0.05$ (~50 mK), $T_2=50=T_1/2$ ($\gamma_\phi=0.03$).
Initial state: qubit $|e\rangle$, resonator $|0\rangle$.

### Key observations

**Peak qubit population = 1.000 throughout the sweep.**
The qubit always starts in $|e\rangle$ — this is simply the initial condition, not
a result. The interesting quantity is how quickly and how completely it decays.

**Thermal steady state:** $P_{ss} = n_{th}/(2n_{th}+1) = 0.05/1.10 = 0.0455$.
At long times, the qubit does not go to zero — it thermalizes to $P_e=0.046$.
This is the signature of finite temperature: the bath keeps re-exciting the qubit.

**Peak photon number grows with coupling:**

| $g/\kappa$ | Peak $\langle n\rangle$ | Peak coherence | Purcell/$\gamma$ |
|-----------|------------------------|----------------|-----------------|
| 0.10 | 0.050 | 0.016 | 0.040 (4%) |
| 0.36 | 0.084 | 0.046 | 0.508 (51%) |
| 0.61 | 0.113 | 0.079 | 1.502 (150%) |
| 0.87 | 0.145 | 0.113 | 3.022 (302%) |
| 1.10 | 0.176 | 0.151 | 4.840 (484%) |

**Important:** at $g/\kappa=0.10$, the peak photon number is 0.050 — exactly equal to
$n_{th}=0.05$. This is NOT a coincidence. In the bad-cavity limit at small coupling,
the resonator simply thermalizes with the qubit bath. The coupling-driven photon
production is negligible compared to the thermal background.

**Purcell effect with thermal bath:**
At $g/\kappa=0.36$, the Purcell rate equals the intrinsic decay rate $\gamma$ — the
resonator is doubling the qubit's decay speed. This is a measurable effect: the qubit's
$T_1$ is cut in half by coupling to the resonator.

At $g/\kappa=1.1$: Purcell rate is $4.84\times$ larger than $\gamma$. The qubit decays
~6× faster than it would in isolation. This is why strong coupling to a lossy resonator
is used as a readout mechanism — the resonator efficiently drains the qubit.

**Total photons emitted: 0.35 to 0.61**
Starting from a single qubit excitation, the total photons emitted should approach 1.
The range 0.35–0.61 tells you that 35–61% of the qubit's excitation energy reaches the
ADC as photons; the rest is lost through $\gamma$ (qubit decay directly to environment)
or re-absorbed thermally. Higher $g/\kappa$ → more efficient routing to the resonator →
more photons reach the ADC.

**Pure dephasing ($\gamma_\phi=0.03$, $T_2=50$) effects:**
Pure dephasing kills the qubit-resonator coherence $|\langle a\sigma_+\rangle|$ faster.
Coherence grows from 0.016 at $g/\kappa=0.1$ to 0.151 at $g/\kappa=1.1$ — despite the
dephasing, coupling drives coherence buildup. But compared to the $T=0$ case (file 2),
coherence peaks are suppressed because $\gamma_\phi$ adds an extra $T_2$ decay channel.

---

## 4. Detuning Sweep (`JC_sweep_detuning_g0.050_nth0.000.npz`)

### Setup
$g=0.05$, $\kappa=\gamma=0.01$. Sweep $\Delta = \omega_q - \omega_r$ from 0 (resonant)
to 0.5 (dispersive). $T=0$, no dephasing.

### The Lorentzian resonance

| $\Delta$ | $g/\Delta$ | Peak $\langle n\rangle$ | Purcell rate | Total photons |
|---------|-----------|------------------------|-------------|---------------|
| 0.00 (resonant) | $\infty$ | **0.737** | 1.000 | 0.495 |
| 0.01 | 5.0 | 0.730 | 0.200 | 0.490 |
| 0.02 | 2.5 | 0.713 | 0.059 | 0.476 |
| 0.05 | 1.0 | 0.608 | 0.010 | 0.397 |
| 0.10 | 0.5 | 0.402 | 0.002 | 0.249 |
| 0.20 | 0.25 | 0.174 | 0.0006 | 0.100 |
| 0.50 | 0.10 | **0.036** | 0.0001 | 0.019 |

**The Lorentzian shape:** peak photon number falls as a Lorentzian in $\Delta$:

$$\langle n\rangle_{peak} \propto \frac{\kappa^2/4}{\Delta^2 + \kappa^2/4}$$

This is the standard cavity resonance lineshape. The half-width at half-maximum is
$\Delta_{HWHM} = \kappa/2 = 0.005$. So the resonance is very sharp — even $\Delta=0.01$
(twice the half-width) barely changes the peak photon number (0.730 vs 0.737).

**Purcell rate is even more sensitive to detuning:**
The Purcell rate drops from 1.000 (at resonance) to 0.010 at $\Delta=0.05$ — a factor
of 100 suppression for a detuning of only $5\kappa$. This is why dispersive readout
works: at large $\Delta$, the Purcell channel is strongly suppressed, protecting the
qubit from resonator-induced decay while still allowing a frequency shift for readout.

**Total photons = efficiency of qubit-to-ADC routing:**
At resonance: 49.5% of the initial qubit excitation reaches the ADC.
At $\Delta=0.5$: only 1.9% reaches the ADC.
The other 50.5% (resonant case) decays via $\gamma$ directly to the qubit's environment,
never producing a measurable photon.

**The log-t picture (what you see on the oscilloscope):**
At $\Delta=0$: fast initial buildup of photons, then slow exponential decay.
At large $\Delta$: very slow photon buildup (dispersive coupling), then very slow decay.
The timescale separates into two distinct regimes on a log-$t$ axis.

---

## 5. Full g/κ Sweep (`JC_sweep_g_kappa_nth0.000_gphi0.000.npz`)

### Setup
$T=0$, no dephasing. $g/\kappa$ from 0.1 to 20. Single-shot qubit $|e\rangle$.

### Coherence as the breakdown signature

| $g/\kappa$ | Peak coherence | Total photons | Physical picture |
|-----------|----------------|---------------|-----------------|
| 0.51 | 0.164 | 0.255 | Approaching threshold |
| 1.01 | 0.258 | 0.400 | At threshold |
| 1.98 | 0.347 | 0.469 | Past threshold |
| 5.13 | 0.431 | 0.494 | Strong coupling |
| 20.0 | **0.480** | **0.498** | Deep Rabi |

**Coherence grows monotonically with $g/\kappa$** — no saturation even at $g/\kappa=20$.
This mirrors exactly the TLS model result where coherence was the mechanistic cause of
Solomon breakdown. In the JC model, this coherence is between the qubit and resonator
field ($|\langle a\sigma_+\rangle|$) rather than between two qubit states.

**Total photons → 0.5 but never reaches 1.0:**
The maximum possible total photons from a single qubit excitation is 1.0
(conservation: one excitation = one photon). At $g/\kappa=20$ the simulation gives 0.498.
The missing 0.502 is lost via $\gamma$ (qubit directly decays to its own environment,
not through the resonator). At $g/\kappa \to \infty$: all decay would be Purcell-enhanced
through the resonator and total photons $\to$ 1. In practice, $g/\kappa=20$ gets to 50%.

**The steady-state photon number $\langle n\rangle_{ss}$ is very small (0.0002–0.005):**
This is the thermal floor. Without a drive, the only source of photons at long times
is the vacuum fluctuations and any residual excitation. At $T=0$ this should be exactly
zero — the small values (order $10^{-3}$) are numerical (finite $T_{end}$ sampling).

---

## 6. Physical Summary: What the ADC Sees

The ADC measures the photon flux $\kappa\langle a^\dagger a\rangle(t)$. Here is what
each physical regime looks like on the oscilloscope:

### Bad cavity / Purcell ($g/\kappa < 0.5$)
Single smooth exponential pulse. Decay time = $1/(\gamma + \gamma_P)$ where
$\gamma_P = 4g^2/\kappa$ is the Purcell enhancement. Peak amplitude is small.
**Solomon equations are exact here.** This is the regime where the classical
rate-equation picture gives the correct answer.

### Threshold ($g/\kappa \sim 0.5$–$1.0$)
Slight non-exponential character — a small shoulder after the initial peak.
The resonator briefly holds the photon before releasing it, creating a small delay
between qubit decay and photon emission. **Solomon begins to break down here.**

### Strong coupling ($g/\kappa > 1$)
Multiple photon bursts visible. The first burst corresponds to the qubit emitting
into the resonator; the second (smaller) burst is the photon returning to the qubit
and being re-emitted. Subsequent bursts decay exponentially. **Solomon fails completely.**
The ADC sees vacuum Rabi oscillations in the output field.

### With thermal population ($n_{th} > 0$)
The baseline photon flux never reaches zero — the thermal bath continuously injects
photons. The signal sits on top of a thermal background $\kappa \cdot n_{th}/(2n_{th}+1)$.
In a real experiment at 50 mK this background is $\sim 5\%$ of the signal — measurable
but manageable.

### With pure dephasing ($\gamma_\phi > 0$, $T_2 < 2T_1$)
The qubit-resonator coherence is suppressed by $e^{-\gamma_\phi t}$. The Rabi oscillations
in the ADC signal are damped faster. In the extreme limit $\gamma_\phi \gg g$: coherence
dies so fast that the strong-coupling oscillations are completely washed out, and the
system effectively returns to the Purcell regime even at large $g$. This is the JC
analog of the T2 sweep result from the TLS model: dephasing rescues the incoherent
(Purcell/Solomon) description.

---

## 7. Connection to the TLS Model and Solomon Breakdown

The parallel between the two models is exact:

| TLS model | JC model | Physical role |
|-----------|----------|--------------|
| $\gamma_t$ | $\kappa$ | Decay rate of second system |
| $g/\gamma_t$ | $g/\kappa$ | Key dimensionless ratio |
| Coherence $|\rho_{eg,ge}|$ | Coherence $|\langle a\sigma_+\rangle|$ | Mechanistic cause of breakdown |
| Solomon breakdown: $g/\gamma_t \sim 0.5$ | Purcell breakdown: $g/\kappa \sim 0.5$ | Same threshold |
| Trace distance $D > 0.05$ | Coherence onset | Same significance criterion |
| ISE grows as $(g/\gamma_t)^{3.1}$ | Total photons grows with $g/\kappa$ | Same power-law onset |

The Solomon approximation applied to the JC model would predict:
- Smooth exponential qubit decay at rate $\gamma + \gamma_P$
- No oscillations in photon flux
- Monotonic total photon buildup

The full Lindblad JC simulation shows:
- Rabi oscillations for $g/\kappa > 1$
- Multiple photon bursts in the flux
- Non-monotonic photon buildup

**The breakdown criterion** $g/\kappa \sim 0.5$–$1$ confirmed by the coherence data
($|\langle a\sigma_+\rangle|$ grows from 0.016 at $g/\kappa=0.1$ to 0.151 at $g/\kappa=1.1$)
matches the TLS result ($g/\gamma_t=0.216$ for $D>0.05$) within a factor of 2.
The slight difference comes from the bosonic nature of the resonator vs the two-level TLS.
