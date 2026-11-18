#!/usr/bin/env python3
"""
Reproduce los números de las Conclusiones del informe.

- Cuenta los nodos por árbol para b = 3 y b = 50 con d = 3 y d = 4.
- Calcula el ancho del intervalo de b = 3 con n = 100.
- Promedia 400 corridas de b = 3 con n = 100 (semillas 0 a 399).
- Compara el ancho del intervalo con el mismo total de nodos:
  b = 3 con n = 318877 contra b = 50 con n = 100.

Uso:
    python conclusiones.py

Tarda unos minutos: la corrida de b = 3 con 318877 árboles es la más pesada.
"""

from broadie_glasserman import precio, tabla

CASO = dict(s0=100.0, K=100.0, tipo="compra", r=0.05, delta=0.10,
            sigma=0.20, T=1.0, d=3)


def nodos(b, d):
    """Cantidad de nodos de un árbol b-nomial con d pasos."""
    return sum(b ** k for k in range(d + 1))


def main():
    n3 = nodos(3, 3)
    n50 = nodos(50, 3)
    n50d4 = nodos(50, 4)
    total_nodos = 100 * n50
    arboles = total_nodos // n3

    print("Nodos")
    tabla(
        ["caso", "nodos"],
        [
            ["b=3, d=3", str(n3)],
            ["b=50, d=3", str(n50)],
            ["b=50, d=4", str(n50d4)],
            ["100 árboles b=50, d=3", str(total_nodos)],
            ["árboles b=3 con ese total", str(arboles)],
        ],
    )
    print()

    r100 = precio(**CASO, b=3, n=100, semilla=2026)
    ancho100 = r100["ic"][1] - r100["ic"][0]
    print("b=3, n=100")
    print(f"  ancho del intervalo = {ancho100:.4f}")
    print(f"  porcentaje de 5.731 = {ancho100 / 5.731:.1%}")
    print()

    bajos, altos = [], []
    for s in range(400):
        r = precio(**CASO, b=3, n=100, semilla=s)
        bajos.append(r["bajo"])
        altos.append(r["alto"])
    print("b=3, n=100, 400 semillas (0 a 399)")
    print(f"  media del estimador bajo = {sum(bajos) / len(bajos):.4f}")
    print(f"  media del estimador alto = {sum(altos) / len(altos):.4f}")
    print()

    r3 = precio(**CASO, b=3, n=arboles, semilla=2026)
    r50 = precio(**CASO, b=50, n=100, semilla=2026)
    print("Mismo total de nodos")
    tabla(
        ["caso", "n", "ancho"],
        [
            ["b=3", str(arboles), f"{r3['ic'][1] - r3['ic'][0]:.4f}"],
            ["b=50", "100", f"{r50['ic'][1] - r50['ic'][0]:.4f}"],
        ],
    )


if __name__ == "__main__":
    main()
