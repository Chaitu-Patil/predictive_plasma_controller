import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit

np.random.seed(41)

E_CHARGE   = 1.602e-19
EPS0       = 8.854e-12
M_ELECTRON = 9.109e-31

N0                = 1e19
B0                = 5.0
FLUCTUATION_SIGMA = 3e18
RAMP_RATE         = 2e18

FIXED_FREQUENCY = 143e9

DT    = 0.01
T_MAX = 200.0
TIME  = np.arange(0, T_MAX, DT)

ANALYSIS_MASK = TIME >= 10.0
DISPLAY_START = 10.0
DISPLAY_END   = 20.0
DISPLAY_MASK  = (TIME >= DISPLAY_START) & (TIME <= DISPLAY_END)

CONTROL_NOISE_STD = 0.0001
LATENCY_STEPS     = 3
HISTORY_WINDOW    = 10

RESONANCE_THRESHOLD = 0.005

plt.rcParams["lines.linewidth"] = 1.2


def density_ramp(t):
    tau = 20.0
    return N0 + RAMP_RATE * tau * (1 - np.exp(-t / tau))

def density_oscillatory(t):
    damping = np.exp(-t / 200.0)
    return N0 + damping * (3e18 * np.sin(4 * t) + 1e18 * np.sin(9 * t))

def density_noise(t, sigma=3e18):
    return N0 + np.random.normal(0, sigma)

def density_spike(t, spike_amount=8e18, elm_interval=0.5, elm_duration=0.02):
    phase = (t + np.random.normal(0, 0.002)) % elm_interval
    if phase < elm_duration:
        return N0 + spike_amount
    return N0

density_models = {
    "Ramp":        density_ramp,
    "Oscillatory": density_oscillatory,
    "Noise":       density_noise,
    "Spike":       density_spike,
}


def plasma_frequency(n):
    return (1 / (2 * np.pi)) * np.sqrt(n * E_CHARGE**2 / (EPS0 * M_ELECTRON))

def cyclotron_frequency(B):
    return (E_CHARGE * B) / (2 * np.pi * M_ELECTRON)

def resonant_frequency(fp, fc):
    return np.sqrt(fp**2 + fc**2)

def absorption_efficiency(fmw, fres):
    width = 0.05 * fres
    return np.exp(-((fmw - fres)**2) / (2 * width**2))


def compute_metrics(f_mw, f_res):
    frac_error        = np.abs(f_mw - f_res) / f_res
    time_in_resonance = np.mean(frac_error <= RESONANCE_THRESHOLD) * 100
    mean_abs_error    = np.mean(frac_error) * 100
    return time_in_resonance, mean_abs_error


def predict_linear(history, latency):
    steps = np.arange(len(history))
    slope, intercept = np.polyfit(steps, history, 1)
    t_pred = len(history) - 1 + latency
    return max(slope * t_pred + intercept, 0)

def predict_sine_fft(history, latency):
    h        = np.array(history)
    mean_h   = np.mean(h)
    centered = h - mean_h
    n        = len(h)

    fft_vals   = np.fft.rfft(centered)
    freqs      = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_vals[1:])
    dom_idx    = np.argmax(magnitudes) + 1

    omega  = 2 * np.pi * freqs[dom_idx]
    A      = 2 * magnitudes[dom_idx - 1] / n
    phi    = np.angle(fft_vals[dom_idx])
    t_pred = n - 1 + latency
    return max(A * np.sin(omega * t_pred + phi) + mean_h, 0)

def predict_noise_linear(history, latency):
    return predict_linear(history, latency)

_spike_cache = {"params": None, "step": -1}

def predict_spike_skewed(history, latency, step_idx, update_interval=10):
    global _spike_cache

    if (_spike_cache["params"] is None or
            step_idx - _spike_cache["step"] >= update_interval):
        h = np.array(history)
        t = np.arange(len(h))

        def skewed_sine(t, A, omega, phi, alpha, C):
            raw = np.sin(omega * t + phi)
            return A * np.power(np.maximum(raw, 0), alpha) + C

        try:
            popt, _ = curve_fit(
                skewed_sine, t, h,
                p0=[4e18, 5.0, 0.0, 2.0, N0],
                bounds=([0, 0, -np.pi, 0.5, 0],
                        [3e20, 50.0, np.pi, 5.0, 3e20]),
                maxfev=3000
            )
            _spike_cache["params"] = popt
            _spike_cache["step"]   = step_idx
        except Exception:
            return predict_linear(history, latency)

    A, omega, phi, alpha, C = _spike_cache["params"]
    t_pred = len(history) - 1 + latency
    raw    = np.sin(omega * t_pred + phi)
    return max(A * np.power(max(raw, 0), alpha) + C, 0)

PREDICTORS = {
    "Ramp":        lambda h, lat, idx: predict_linear(h, lat),
    "Oscillatory": lambda h, lat, idx: predict_sine_fft(h, lat),
    "Noise":       lambda h, lat, idx: predict_noise_linear(h, lat),
    "Spike":       lambda h, lat, idx: predict_spike_skewed(h, lat, idx),
}


def run_simulation(density_func, predictor_func, add_noise=True):
    eff_fixed   = []
    eff_predict = []
    res_freqs   = []
    mw_predict  = []

    fc              = cyclotron_frequency(B0)
    density_history = deque(maxlen=HISTORY_WINDOW)

    for idx, t in enumerate(TIME):
        n_measured = density_func(t)
        if add_noise:
            n_measured += np.random.normal(0, FLUCTUATION_SIGMA)
        n_measured = max(n_measured, 0)

        density_history.append(n_measured)

        fp_true   = plasma_frequency(n_measured)
        fres_true = resonant_frequency(fp_true, fc)

        f_fixed = FIXED_FREQUENCY

        if len(density_history) >= 3:
            n_predicted = predictor_func(list(density_history), LATENCY_STEPS, idx)
        else:
            n_predicted = n_measured

        fp_pred        = plasma_frequency(n_predicted)
        fres_pred      = resonant_frequency(fp_pred, fc)
        actuator_noise = np.random.normal(0, CONTROL_NOISE_STD)
        f_predict      = (1 + actuator_noise) * fres_pred

        eff_fixed.append(absorption_efficiency(f_fixed,    fres_true))
        eff_predict.append(absorption_efficiency(f_predict, fres_true))
        res_freqs.append(fres_true)
        mw_predict.append(f_predict)

    return (
        np.array(eff_fixed),
        np.array(eff_predict),
        np.array(res_freqs),
        np.array(mw_predict),
    )


fixed_avgs   = []
predict_avgs = []
labels       = []
combined_results = {}

print(f"{'Model':<14} {'Fixed η':>10} {'Predict η':>12} "
      f"{'Fixed TiR%':>12} {'Pred TiR%':>12} "
      f"{'Fixed MAE%':>12} {'Pred MAE%':>12}")
print("─" * 84)

for name, model in density_models.items():
    _spike_cache["params"] = None
    add_noise = (name != "Noise")

    ef, ep, res, mwp = run_simulation(model, PREDICTORS[name], add_noise)

    ef_a  = ef[ANALYSIS_MASK]
    ep_a  = ep[ANALYSIS_MASK]
    res_a = res[ANALYSIS_MASK]
    mwp_a = mwp[ANALYSIS_MASK]

    avgf = np.mean(ef_a)
    avgp = np.mean(ep_a)

    tir_f, mae_f = compute_metrics(np.full_like(res_a, FIXED_FREQUENCY), res_a)
    tir_p, mae_p = compute_metrics(mwp_a, res_a)

    fixed_avgs.append(avgf)
    predict_avgs.append(avgp)
    labels.append(name)
    combined_results[name] = (ef, ep, res, mwp)

    print(f"{name:<14} {avgf:>10.4f} {avgp:>12.4f} "
          f"{tir_f:>12.2f} {tir_p:>12.2f} "
          f"{mae_f:>12.4f} {mae_p:>12.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(TIME[DISPLAY_MASK], ef[DISPLAY_MASK], color="red",
             label=f"Fixed      (η={avgf:.3f}, TiR={tir_f:.1f}%)")
    plt.plot(TIME[DISPLAY_MASK], ep[DISPLAY_MASK], color="blue",
             label=f"Predictive (η={avgp:.3f}, TiR={tir_p:.1f}%)")
    plt.xlabel("Time (s)")
    plt.ylabel("Absorption Efficiency η")
    plt.title(f"{name} — Absorption Efficiency")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(TIME[DISPLAY_MASK], mwp[DISPLAY_MASK], color="blue",
             label=f"Predictive (MAE={mae_p:.3f}%)", zorder=1)
    plt.plot(TIME[DISPLAY_MASK],
             np.full(np.sum(DISPLAY_MASK), FIXED_FREQUENCY),
             color="red", label=f"Fixed (MAE={mae_f:.3f}%)", zorder=2)
    plt.plot(TIME[DISPLAY_MASK], res[DISPLAY_MASK], color="orange",
             linestyle="--", label="Resonant Frequency", zorder=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{name} — Frequency Tracking")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


fixed_matrix   = np.array([combined_results[m][0] for m in density_models])
predict_matrix = np.array([combined_results[m][1] for m in density_models])
res_matrix     = np.array([combined_results[m][2] for m in density_models])
pred_freq_mat  = np.array([combined_results[m][3] for m in density_models])

avg_fixed   = np.mean(fixed_matrix[:,   ANALYSIS_MASK], axis=0)
avg_predict = np.mean(predict_matrix[:, ANALYSIS_MASK], axis=0)
avg_res     = np.mean(res_matrix[:,     ANALYSIS_MASK], axis=0)
avg_freq    = np.mean(pred_freq_mat[:,  ANALYSIS_MASK], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(TIME[ANALYSIS_MASK], avg_fixed,   color="red",  label="Fixed Frequency (143 GHz)")
plt.plot(TIME[ANALYSIS_MASK], avg_predict, color="blue", label="Predictive Adaptive Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Average Absorption Efficiency η")
plt.title("Average Absorption Efficiency: Fixed vs Predictive (All Models)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(TIME[ANALYSIS_MASK], avg_freq, color="blue",
         label="Average Predictive Frequency", zorder=1)
plt.plot(TIME[ANALYSIS_MASK],
         np.full(np.sum(ANALYSIS_MASK), FIXED_FREQUENCY),
         color="red", label="Fixed Frequency (143 GHz)", zorder=2)
plt.plot(TIME[ANALYSIS_MASK], avg_res, color="orange",
         linestyle="--", label="Average Resonant Frequency", zorder=3)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Average Frequency Tracking: All Models")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

x     = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, fixed_avgs,   width, label="Fixed",      color="red")
plt.bar(x + width/2, predict_avgs, width, label="Predictive", color="blue")

for i, (f, p) in enumerate(zip(fixed_avgs, predict_avgs)):
    pct  = (p - f) / f * 100
    sign = "+" if pct >= 0 else ""
    plt.text(x[i] + width/2, p + 0.002,
             f"{sign}{pct:.1f}%", ha="center", fontsize=9)

plt.xticks(x, labels)
plt.ylabel("Average Absorption Efficiency η")
plt.title("Average Absorption Comparison: Fixed vs Predictive")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
