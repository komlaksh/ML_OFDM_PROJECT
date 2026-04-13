import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

MODEL_FILES = {
    "SVM": "SVM_model.pkl",
    "RF": "RF_model.pkl",
    "DT": "DT_model.pkl",
}


def load_dataset(path, n=200000):
    df = pd.read_csv(path)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=42)
    return df


def configure_model_for_prediction(model):
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    return model


def load_models(results_dir):
    models = {}
    for name, filename in MODEL_FILES.items():
        model_path = os.path.join(results_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[name] = configure_model_for_prediction(joblib.load(model_path))
    return models


def qpsk_mod(bits):
    bits = bits.reshape(-1, 2)
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    return symbols / np.sqrt(2)


def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    bits[:, 0] = np.real(symbols) > 0
    bits[:, 1] = np.imag(symbols) > 0
    return bits.reshape(-1)


def label_to_qpsk(labels):
    mapping = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)
    return mapping[labels] / np.sqrt(2)


def build_features(received_symbols, snr_db):
    snr_col = np.full(len(received_symbols), snr_db, dtype=np.float64)
    return np.column_stack((received_symbols.real, received_symbols.imag, snr_col))


def simulate_ofdm_frame(num_subcarriers, cp_len, snr_db, rng):
    bits = rng.integers(0, 2, size=num_subcarriers * 2)

    # QPSK modulation
    tx_symbols = qpsk_mod(bits)

    # IFFT (OFDM)
    tx_time = np.fft.ifft(tx_symbols)

    # Add CP
    tx_with_cp = np.concatenate([tx_time[-cp_len:], tx_time])

    # AWGN channel (NO multipath)
    rx_signal = tx_with_cp.copy()

    # Noise
    power = np.mean(np.abs(rx_signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = power / snr_linear

    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(rx_signal.shape) + 1j * rng.standard_normal(rx_signal.shape)
    )

    rx_signal = rx_signal + noise

    # Remove CP
    rx_no_cp = rx_signal[cp_len:]

    # FFT
    rx_freq = np.fft.fft(rx_no_cp)

    return bits, rx_freq


def compute_ber(models, snr_values, num_subcarriers, cp_len, iterations, random_state=42):
    rng = np.random.default_rng(random_state)
    ber_results = {"Traditional": []}
    for name in models:
        ber_results[name] = []

    for snr in snr_values:
        error_counts = {"Traditional": 0}
        for name in models:
            error_counts[name] = 0
        total_bits = 0

        for _ in range(iterations):
            bits, rx_freq = simulate_ofdm_frame(num_subcarriers, cp_len, snr, rng)
            traditional_bits = qpsk_demod(rx_freq)
            error_counts["Traditional"] += np.sum(bits != traditional_bits)

            features = build_features(rx_freq, snr)
            for name, model in models.items():
                try:
                    with joblib.parallel_backend("threading"):
                        labels = model.predict(features)
                except Exception:
                    labels = model.predict(features)

                estimated_symbols = label_to_qpsk(labels)
                estimated_bits = qpsk_demod(estimated_symbols)
                error_counts[name] += np.sum(bits != estimated_bits)

            total_bits += len(bits)

        for key in ber_results:
            ber_results[key].append(error_counts[key] / total_bits)

    return ber_results


def save_ber_table(snr_values, ber_results, output_path):
    table = {"SNR": snr_values}
    for name, values in ber_results.items():
        table[name] = values
    df = pd.DataFrame(table)
    df.to_csv(output_path, index=False)
    return df

def plot_ber(snr_values, ber_results, output_path):
    plt.figure(figsize=(7,5))

    # 🔷 Line styles (IEEE style)
    styles = {
        "Traditional": ("k--", "o"),   # black dashed
        "SVM": ("b-", "s"),            # blue solid
        "RF": ("g-", "^"),             # green solid
        "DT": ("r-", "d"),             # red solid
    }

    for name, values in ber_results.items():
        line, marker = styles.get(name, ("-", "o"))
        plt.semilogy(
            snr_values,
            values,
            line,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=name
        )

    # 🔷 Labels
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (BER)", fontsize=12)

    # 🔷 Title (professional)
    plt.title("BER Performance Comparison of ML Models in OFDM System", fontsize=12)

    # 🔷 Grid (major + minor)
    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.3)

    # 🔷 Legend
    plt.legend(fontsize=10)

    # 🔷 Axis limits (important for shape)
    plt.ylim(1e-5, 1)
    plt.xlim(min(snr_values), max(snr_values))

    # 🔷 Layout fix
    plt.tight_layout()

    # 🔷 Save high quality
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BER for conventional and ML-based OFDM detection.")
    parser.add_argument("--dataset", default="../data/dataset.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--sample-size", type=int, default=200000, help="Sample size for dataset validation.")
    parser.add_argument("--model-dir", default="../results", help="Directory containing trained model files.")
    parser.add_argument("--output-dir", default="../results", help="Directory to write BER plot and CSV output.")
    parser.add_argument("--snr-start", type=int, default=0, help="Starting SNR value in dB.")
    parser.add_argument("--snr-stop", type=int, default=20, help="Ending SNR value in dB.")
    parser.add_argument("--snr-step", type=int, default=2, help="SNR step size in dB.")
    parser.add_argument("--cp", type=int, default=8, help="Cyclic prefix length.")
    parser.add_argument("--iters", type=int, default=3000, help="Number of frames per SNR point.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for BER simulation.")
    return parser.parse_args()


def main():
    parser = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, parser.dataset))
    model_dir = os.path.abspath(os.path.join(script_dir, parser.model_dir))
    results_dir = os.path.abspath(os.path.join(script_dir, parser.output_dir))
    os.makedirs(results_dir, exist_ok=True)

    load_dataset(data_path, parser.sample_size)
    models = load_models(model_dir)

    snr_values = np.arange(parser.snr_start, parser.snr_stop + 1, parser.snr_step)
    ber_results = compute_ber(
        models,
        snr_values,
        num_subcarriers=64,
        cp_len=parser.cp,
        iterations=parser.iters,
        random_state=parser.seed,
    )

    plot_path = os.path.join(results_dir, "ber_final.png")
    csv_path = os.path.join(results_dir, "ber_final.csv")
    plot_ber(snr_values, ber_results, plot_path)
    save_ber_table(snr_values, ber_results, csv_path)

    print(f"Saved BER comparison plot to: {plot_path}")
    print(f"Saved BER results table to: {csv_path}")


if __name__ == "__main__":
    main()

