#!/usr/bin/env python3
"""
Ejemplo numérico del informe: un solo árbol con b = 3 ramas y d = 2 pasos,
con las normales fijas Z1 = (2, 0, -2) y Z2 = (1.5, 0, -1.5).

Imprime el árbol y las cuentas de la sección "Ejemplo" del informe. Al final
comprueba que el resultado coincide con estimadores() de broadie_glasserman.py.

Uso:
    python ejemplo.py
"""

from math import exp, sqrt

import numpy as np

from broadie_glasserman import estimadores

CASO = dict(s0=105.0, K=100.0, r=0.04, delta=0.12, sigma=0.25, T=0.5)
D, B = 2, 3
Z1 = np.array([2.0, 0.0, -2.0])
Z2 = np.array([1.5, 0.0, -1.5])


def armar_arbol():
    """Arma el árbol con las Z fijas. Los valores se imprimen redondeados."""
    dt = CASO["T"] / D
    deriva = (CASO["r"] - CASO["delta"] - 0.5 * CASO["sigma"] ** 2) * dt
    vol = CASO["sigma"] * sqrt(dt)
    niveles = [np.array([CASO["s0"]])]
    for z in (Z1, Z2):
        padres = niveles[-1]
        hijos = padres[:, None] * np.exp(deriva + vol * z[None, :])
        niveles.append(hijos.ravel())
    return niveles


def main():
    K, r, T = CASO["K"], CASO["r"], CASO["T"]
    descuento = exp(-r * T / D)
    niveles = armar_arbol()
    pagos = [np.maximum(nivel - K, 0.0) for nivel in niveles]

    print("Ejemplo: b = 3, d = 2")
    print(f"  S0 = {CASO['s0']:.2f}, K = {K:.2f}, r = {CASO['r']}, "
          f"delta = {CASO['delta']}, sigma = {CASO['sigma']}, T = {CASO['T']}")
    print(f"  descuento por paso = {descuento:.5f}")
    print()

    print("Árbol")
    for i in range(D):
        for k, s in enumerate(niveles[i]):
            print(f"  t{i} nodo {k + 1}: S = {s:.2f}, h = {pagos[i][k]:.2f}")
            for j in range(B):
                hijo = niveles[i + 1][k * B + j]
                print(f"    hijo {j + 1}: S = {hijo:.2f}, h = {pagos[i + 1][k * B + j]:.2f}")
    print()

    alto = pagos[D]
    bajo = pagos[D]
    for i in range(D - 1, -1, -1):
        print(f"t{i}")
        hijos_alto = descuento * alto.reshape(-1, B)
        hijos_bajo = descuento * bajo.reshape(-1, B)
        alto = np.empty(len(niveles[i]))
        bajo = np.empty(len(niveles[i]))
        for k, h in enumerate(pagos[i]):
            print(f"  nodo {k + 1}: S = {niveles[i][k]:.2f}, h = {h:.2f}")
            continuacion = hijos_alto[k].mean()
            alto[k] = max(h, continuacion)
            print(f"    alto: continuación = {continuacion:.2f}, VH = {alto[k]:.2f}")
            eta = np.empty(B)
            for j in range(B):
                otros = (hijos_bajo[k].sum() - hijos_bajo[k, j]) / (B - 1)
                eta[j] = h if h >= otros else hijos_bajo[k, j]
                print(f"      j = {j + 1}: C = {otros:.2f}, Y = {eta[j]:.2f}")
            bajo[k] = eta.mean()
            print(f"    bajo: VL = {bajo[k]:.2f}")
    print(f"Theta = {alto[0]:.2f}, theta = {bajo[0]:.2f}")

    bajo_impl, alto_impl = estimadores(niveles, K, "compra", r, T, B)
    assert np.isclose(bajo[0], bajo_impl) and np.isclose(alto[0], alto_impl)
    print("Coincide con estimadores() de broadie_glasserman.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
