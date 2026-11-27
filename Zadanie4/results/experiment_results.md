# Wyniki eksperymentów — Zadanie 4 (Collaborative Filtering)

5-krotna walidacja krzyżowa ze stratyfikacją po osobach (`seed=0`, 18 zakrytych / 72 treningowe oceny na osobę).
Algorytm: metoda najszybszego spadku z wykładu (`X`, `P`, `p0`).
Zgodnie z wykładem **nie łączono** jednoczesnego doboru `n` i regularyzacji `lambda` — przetestowano dwie osobne ścieżki:
- **Ścieżka A**: dobór wymiaru przestrzeni cech `n` bez regularyzacji (`lambda = 0`).
- **Ścieżka B**: ustalone wystarczająco duże `n` i dobór samej regularyzacji `lambda` (dla `X` i `P`).

**Baseline'y:** średnia filmu = `0.2568`, średnia osoby = `0.3244`, moda osoby = `0.4108`.

**Najlepsza konfiguracja:** `mode=onehot`, `n=150`, `lam=4.5`, `norm=none`, `update=joint` → Accuracy = **`0.5562`** (`± 0.0035`), Accuracy ±1 = `0.7891`, MAE = `0.7706`.

## 1. Ścieżka A — dobór wymiaru cech `n` bez regularyzacji (`lam = 0`)

| mode | n | lam | norm | update | accuracy | accuracy_1 | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onehot | 1 | 0.0000 | none | joint | 0.4565 | 0.7371 | 0.9591 | 1.4874 |
| onehot | 2 | 0.0000 | none | joint | 0.4766 | 0.7479 | 0.9238 | 1.4612 |
| onehot | 4 | 0.0000 | none | joint | 0.5027 | 0.7594 | 0.8767 | 1.4202 |
| onehot | 6 | 0.0000 | none | joint | 0.5127 | 0.7651 | 0.8523 | 1.3907 |
| onehot | 8 | 0.0000 | none | joint | 0.5148 | 0.7645 | 0.8483 | 1.3849 |
| onehot | 10 | 0.0000 | none | joint | 0.5146 | 0.7655 | 0.8461 | 1.3805 |
| onehot | 15 | 0.0000 | none | joint | 0.5110 | 0.7664 | 0.8453 | 1.3726 |
| onehot | 20 | 0.0000 | none | joint | 0.5043 | 0.7574 | 0.8688 | 1.3998 |
| onehot | 30 | 0.0000 | none | joint | 0.4969 | 0.7519 | 0.8878 | 1.4201 |
| scalar | 1 | 0.0000 | movie | joint | 0.3408 | 0.7735 | 0.9339 | 1.2289 |
| scalar | 2 | 0.0000 | movie | joint | 0.3563 | 0.7849 | 0.9070 | 1.2089 |
| scalar | 4 | 0.0000 | movie | joint | 0.3732 | 0.7964 | 0.8811 | 1.1952 |
| scalar | 6 | 0.0000 | movie | joint | 0.3877 | 0.7972 | 0.8724 | 1.2066 |
| scalar | 8 | 0.0000 | movie | joint | 0.3879 | 0.7880 | 0.8920 | 1.2437 |
| scalar | 10 | 0.0000 | movie | joint | 0.3872 | 0.7818 | 0.9117 | 1.2788 |
| scalar | 15 | 0.0000 | movie | joint | 0.3776 | 0.7599 | 0.9726 | 1.3727 |
| scalar | 20 | 0.0000 | movie | joint | 0.3650 | 0.7350 | 1.0327 | 1.4481 |

## 2. Ścieżka B — ustalone duże `n` + dobór regularyzacji `lam`

| mode | n | lam | norm | update | accuracy | accuracy_1 | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onehot | 150 | 1.0000 | none | joint | 0.5340 | 0.7694 | 0.8268 | 1.3780 |
| onehot | 150 | 2.0000 | none | joint | 0.5480 | 0.7801 | 0.7916 | 1.3397 |
| onehot | 150 | 3.0000 | none | joint | 0.5539 | 0.7863 | 0.7756 | 1.3223 |
| onehot | 150 | 4.0000 | none | joint | 0.5560 | 0.7889 | 0.7709 | 1.3186 |
| onehot | 150 | 4.5000 | none | joint | 0.5562 | 0.7891 | 0.7706 | 1.3188 |
| onehot | 150 | 5.0000 | none | joint | 0.5555 | 0.7890 | 0.7705 | 1.3171 |
| onehot | 150 | 6.0000 | none | joint | 0.5522 | 0.7880 | 0.7769 | 1.3251 |
| onehot | 150 | 8.0000 | none | joint | 0.5383 | 0.7819 | 0.8034 | 1.3528 |
| onehot | 150 | 12.0000 | none | joint | 0.4909 | 0.7563 | 0.8935 | 1.4326 |
| scalar | 50 | 1.0000 | movie | joint | 0.3476 | 0.7396 | 1.0184 | 1.3862 |
| scalar | 50 | 4.0000 | movie | joint | 0.3707 | 0.7881 | 0.9009 | 1.2201 |
| scalar | 50 | 8.0000 | movie | joint | 0.3673 | 0.8071 | 0.8656 | 1.1594 |
| scalar | 50 | 12.0000 | movie | joint | 0.3563 | 0.8069 | 0.8717 | 1.1491 |
| scalar | 50 | 16.0000 | movie | joint | 0.3461 | 0.8017 | 0.8863 | 1.1583 |
| scalar | 50 | 20.0000 | movie | joint | 0.3350 | 0.7912 | 0.9087 | 1.1787 |

## 3. Porównanie aktualizacji (`joint` vs `alternating`) i normalizacji (`none`, `movie`, `user`)

| mode | n | lam | norm | update | accuracy | accuracy_1 | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onehot | 150 | 4.5000 | none | joint | 0.5562 | 0.7891 | 0.7706 | 1.3188 |
| onehot | 150 | 4.5000 | none | alternating | 0.5561 | 0.7890 | 0.7707 | 1.3186 |
| onehot | 150 | 4.5000 | movie | joint | 0.5516 | 0.7863 | 0.7795 | 1.3264 |
| onehot | 150 | 4.5000 | movie | alternating | 0.5519 | 0.7863 | 0.7795 | 1.3271 |
| onehot | 150 | 4.5000 | user | joint | 0.5561 | 0.7888 | 0.7709 | 1.3191 |
| onehot | 150 | 4.5000 | user | alternating | 0.5562 | 0.7887 | 0.7707 | 1.3188 |
| scalar | 50 | 12.0000 | none | joint | 0.3597 | 0.8083 | 0.8663 | 1.1474 |
| scalar | 50 | 12.0000 | none | alternating | 0.3597 | 0.8084 | 0.8662 | 1.1474 |
| scalar | 50 | 12.0000 | movie | joint | 0.3563 | 0.8069 | 0.8717 | 1.1491 |
| scalar | 50 | 12.0000 | movie | alternating | 0.3565 | 0.8070 | 0.8714 | 1.1491 |
| scalar | 50 | 12.0000 | user | joint | 0.3599 | 0.8079 | 0.8665 | 1.1473 |
| scalar | 50 | 12.0000 | user | alternating | 0.3601 | 0.8080 | 0.8664 | 1.1473 |

## 4. Stabilność na innym podziale foldów (`seed = 1`)

| rank | mode | n | lam | norm | update | accuracy_seed0 | accuracy_seed1 | mae_seed1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | onehot | 150 | 4.5000 | none | joint | 0.5562 | 0.5571 | 0.7653 |
| 2 | onehot | 150 | 4.5000 | user | alternating | 0.5562 | 0.5565 | 0.7663 |
| 3 | onehot | 150 | 4.5000 | user | joint | 0.5561 | 0.5564 | 0.7666 |
| 4 | onehot | 150 | 4.5000 | none | alternating | 0.5561 | 0.5572 | 0.7651 |
| 5 | onehot | 150 | 4.0000 | none | joint | 0.5560 | 0.5562 | 0.7673 |
