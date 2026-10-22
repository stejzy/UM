"""Algorytm genetyczny wybierający zestaw cech (maskę 0/1).

Chromosom: wektor 0/1 długości = liczba cech (1 = cecha używana).
GA nie wie nic o kNN: dostaje funkcję fitness(maska) -> liczba i szuka maski
o jak najwyższym fitness.
"""
import csv

import numpy as np

POPULATION = 30
GENERATIONS = 40
PATIENCE = 10          # stop, gdy przez tyle pokoleń nie ma poprawy
TOURNAMENT = 3         # selekcja turniejowa: losuj 3, wybierz najlepszego
CROSSOVER_P = 0.9      # prawdopodobieństwo krzyżowania
ELITE = 2              # najlepsi przechodzą do kolejnego pokolenia bez zmian


def run_ga(fitness, n_features, rng, initial=(), population=POPULATION,
           generations=GENERATIONS, patience=PATIENCE, log=None, names=None,
           log_file=None):
    """Zwraca (najlepsza maska, jej fitness, liczba ocenionych różnych masek).

    fitness:  funkcja maska (np.array 0/1) -> float, im więcej tym lepiej
    rng:      generator losowy (np.random.default_rng(seed)) - wynik powtarzalny
    initial:  maski, od których warto zacząć (np. zestaw globalny)
    log:      jeśli podany (np. "k = 30"), po każdym pokoleniu wypisywany jest
              najlepszy dotychczasowy wynik z tym napisem na początku linii
    names:    nazwy cech, żeby log pokazywał też wybrane cechy, a nie tylko maskę
    log_file: jeśli podana ścieżka, po każdym pokoleniu dopisywany jest wiersz CSV
              (do późniejszej analizy, czy GA się poprawia)
    """
    LOG_COLUMNS = ["generation", "best_fit", "generation_best", "generation_mean",
                   "generation_worst", "n_features", "best_mask", "features", "evaluated_masks"]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_COLUMNS)
    cache = {}   # te same maski pojawiają się wielokrotnie, liczymy je tylko raz

    def score(mask):
        key = mask.tobytes()
        if key not in cache:
            cache[key] = fitness(mask)
        return cache[key]

    def repair(mask):
        """Pusty zestaw cech nie ma sensu: włącz jedną losową cechę."""
        if not mask.any():
            mask[rng.integers(n_features)] = 1
        return mask

    # Populacja startowa: podane maski, każda cecha osobno, reszta losowo
    pop = [np.array(m, dtype=np.int8) for m in initial]
    pop += [np.eye(n_features, dtype=np.int8)[i] for i in rng.permutation(n_features)]
    pop = pop[:population]
    while len(pop) < population:
        pop.append(repair(rng.integers(0, 2, n_features).astype(np.int8)))

    best_mask, best_fit, stale = None, -np.inf, 0
    for generation in range(1, generations + 1):
        fits = np.array([score(m) for m in pop])
        order = np.argsort(-fits, kind="stable")
        if fits[order[0]] > best_fit:
            best_mask, best_fit, stale = pop[order[0]].copy(), fits[order[0]], 0
        else:
            stale += 1

        if log is not None:
            line = (f"{log} | pokolenie {generation:>2} | best fit {best_fit:.5f} | "
                    f"best mask {''.join(map(str, best_mask))}")
            if names is not None:
                line += f" | cechy: {', '.join(n for n, m in zip(names, best_mask) if m)}"
            print(line + f" | ocenionych masek: {len(cache)}", flush=True)

        if log_file is not None:
            features = [n for n, m in zip(names, best_mask) if m] if names is not None else []
            with open(log_file, "a", newline="", encoding="utf-8") as f:   # dopisanie wiersza
                csv.writer(f).writerow([
                    generation,
                    round(float(best_fit), 6),        # najlepszy dotychczas (nie maleje)
                    round(float(fits.max()), 6),      # najlepszy w tym pokoleniu
                    round(float(fits.mean()), 6),     # średnia populacji
                    round(float(fits.min()), 6),      # najgorszy w tym pokoleniu
                    int(best_mask.sum()),
                    " ".join(map(str, best_mask)),    # spacje: Excel nie zgubi zer z przodu
                    ", ".join(features),
                    len(cache),
                ])

        if stale >= patience:
            break

        def tournament():
            contenders = rng.choice(len(pop), TOURNAMENT, replace=False)
            return pop[contenders[np.argmax(fits[contenders])]]

        new_pop = [pop[i].copy() for i in order[:ELITE]]          # elitaryzm
        while len(new_pop) < population:
            a, b = tournament(), tournament()
            if rng.random() < CROSSOVER_P:                          # krzyżowanie jednorodne
                child = np.where(rng.random(n_features) < 0.5, a, b).astype(np.int8)
            else:
                child = a.copy()
            flip = rng.random(n_features) < 1.0 / n_features         # mutacja
            child[flip] = 1 - child[flip]
            new_pop.append(repair(child))
        pop = new_pop

    return best_mask, float(best_fit), len(cache)


if __name__ == "__main__":
    # Test na prostym problemie: fitness = liczba trafionych bitów ukrytego wzorca
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 18)
    mask, fit, evaluated = run_ga(lambda m: (m == target).sum(), 18, rng, generations=100,
                                  patience=30)
    print(f"Wzorzec:   {target}")
    print(f"Znaleziony:{mask}  fitness {fit:.0f}/18, ocenionych masek: {evaluated} "
          f"(z {2 ** 18} możliwych)")
    assert fit == 18
