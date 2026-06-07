import functools

from forest import PredictorForest
from run_experiments import run, discretize, normalize, LOOKAHEAD_STEPS
from similarity import gaussian, hamming
from datasets import (
    load_airline_passengers,
    load_gutenberg_text,
    load_dna_sequence,
    load_weather_events,
    random_integers,
)

N_TREES = 5
DROPOUT = 0.2

forest_cls = functools.partial(PredictorForest, n_trees=N_TREES, dropout=DROPOUT)


def main() -> None:
    print(f"Universal Sequence Predictor — Forest Experiment Suite")
    print(f"Forest: {N_TREES} trees  |  dropout per tree: {DROPOUT}\n")

    print("[1/5] Airline passengers (numeric time series)...")
    raw = load_airline_passengers()
    seq = discretize(normalize(raw), n_bins=8)
    run("Airline Passengers", seq, gaussian(sigma=2.0), context_length=4,
        predictor_cls=forest_cls)

    print("\n[2/5] Alice in Wonderland (character-level text)...")
    try:
        seq = load_gutenberg_text(n_chars=1500)
        run("Alice in Wonderland", seq, hamming, context_length=3,
            predictor_cls=forest_cls)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[3/5] Bacteriophage lambda genome (DNA)...")
    try:
        seq = load_dna_sequence(n_bases=1500)
        run("Bacteriophage Lambda DNA", seq, hamming, context_length=4,
            predictor_cls=forest_cls)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[4/5] NYC daily weather codes (categorical events)...")
    try:
        seq = load_weather_events(n_days=500)
        run("NYC Weather Events", seq, hamming, context_length=3,
            predictor_cls=forest_cls)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[5/5] Python PRNG (unpredictable baseline)...")
    seq = random_integers(n=500, low=0, high=9)
    run("Python PRNG (random)", seq, gaussian(sigma=1.0), context_length=3,
        predictor_cls=forest_cls)

    print(f"\n{'═'*62}")
    print(f"Each row: {N_TREES} trees aggregated via confidence-weighted vote.")
    print("Node counts are summed across all trees.")
    print("Topology diverges via independent {:.0%} feedback dropout per tree.\n".format(DROPOUT))


if __name__ == "__main__":
    main()
