import csv
import random
import sys

# użycie: python fill_random.py [plik_wejściowy] [plik_wyjściowy]
IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "task.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else IN_PATH

with open(IN_PATH, newline="") as f:
    rows = list(csv.reader(f, delimiter=";"))

for row in rows:
    if row[3] == "NULL":
        row[3] = str(random.randint(0, 5))  # losowa ocena z zakresu <0,5>

with open(OUT_PATH, "w", newline="") as f:
    csv.writer(f, delimiter=";", lineterminator="\n").writerows(rows)
