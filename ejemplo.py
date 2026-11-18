#!/usr/bin/env python3
"""
Ejemplo numérico del informe: un solo árbol con b = 3 ramas y d = 2 pasos,
con las normales fijas Z1 = (2, 0, -2) y Z2 = (1.5, 0, -1.5).

Imprime el árbol con los precios y los estimadores de cada nodo. Al final
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
    """Arma el árbol con las Z fijas."""
    dt = CASO["T"] / D
    deriva = (CASO["r"] - CASO["delta"] - 0.5 * CASO["sigma"] ** 2) * dt
    vol = CASO["sigma"] * sqrt(dt)
    niveles = [np.array([CASO["s0"]])]
    for z in (Z1, Z2):
        padres = niveles[-1]
        hijos = padres[:, None] * np.exp(deriva + vol * z[None, :])
        niveles.append(hijos.ravel())
    return niveles


def estimadores_por_nivel(niveles):
    """Aplica la recursión y devuelve h, alto y bajo para cada nivel."""
    K, r, T = CASO["K"], CASO["r"], CASO["T"]
    descuento = exp(-r * T / D)
    h = [np.maximum(nivel - K, 0.0) for nivel in niveles]
    alto = [None] * (D + 1)
    bajo = [None] * (D + 1)
    alto[D] = h[D].copy()
    bajo[D] = h[D].copy()
    for t in range(D - 1, -1, -1):
        hijos_alto = descuento * alto[t + 1].reshape(-1, B)
        hijos_bajo = descuento * bajo[t + 1].reshape(-1, B)
        alto[t] = np.empty(len(niveles[t]))
        bajo[t] = np.empty(len(niveles[t]))
        for k in range(len(niveles[t])):
            hk = h[t][k]
            alto[t][k] = max(hk, hijos_alto[k].mean())
            eta = np.empty(B)
            for j in range(B):
                otros = (hijos_bajo[k].sum() - hijos_bajo[k, j]) / (B - 1)
                eta[j] = hk if hk >= otros else hijos_bajo[k, j]
            bajo[t][k] = eta.mean()
    return h, alto, bajo


def imprimir_arbol(niveles, h, alto, bajo):
    """Dibuja el árbol con los precios y los estimadores en cada nodo."""
    ancho = 92
    filas = [[" "] * ancho for _ in range(12)]

    def poner(fila, x, texto):
        for i, ch in enumerate(texto):
            X = x + i
            if 0 <= X < ancho:
                filas[fila][X] = ch

    def centrar(fila, x, texto):
        poner(fila, x - len(texto) // 2, texto)

    hojas_x = [4 + 10 * i for i in range(9)]
    t1_x = [14, 44, 74]
    raiz_x = 44

    centrar(0, raiz_x, f"{niveles[0][0]:.2f}")
    centrar(1, raiz_x, f"VH {alto[0][0]:.2f}  VL {bajo[0][0]:.2f}")
    poner(2, raiz_x, "│")
    for x in range(t1_x[0], t1_x[-1] + 1):
        poner(3, x, "─")
    for x in t1_x:
        poner(3, x, "┼")
    poner(3, t1_x[0], "┌")
    poner(3, t1_x[-1], "┐")
    for x in t1_x:
        poner(4, x, "│")

    for x, k in zip(t1_x, range(3)):
        centrar(5, x, f"{niveles[1][k]:.2f}")
        centrar(6, x, f"VH {alto[1][k]:.2f}  VL {bajo[1][k]:.2f}")

    for xp in t1_x:
        poner(7, xp, "│")
        for x in range(xp - 10, xp + 11):
            poner(8, x, "─")
        for x in (xp - 10, xp, xp + 10):
            poner(8, x, "┼")
        poner(8, xp - 10, "┌")
        poner(8, xp + 10, "┐")
        for x in (xp - 10, xp, xp + 10):
            poner(9, x, "│")

    for x, k in zip(hojas_x, range(9)):
        centrar(10, x, f"{niveles[2][k]:.2f}")
        centrar(11, x, f"h {h[2][k]:.2f}")

    for fila in filas:
        print("".join(fila).rstrip())


def main():
    niveles = armar_arbol()
    h, alto, bajo = estimadores_por_nivel(niveles)

    print(f"Ejemplo: b = {B}, d = {D}")
    print(f"  S0 = {CASO['s0']:.2f}, K = {CASO['K']:.2f}, r = {CASO['r']}, "
          f"delta = {CASO['delta']}, sigma = {CASO['sigma']}, T = {CASO['T']}")
    print("  En t0 y t1 se muestran VH y VL; en t2, el payoff h.")
    print()
    imprimir_arbol(niveles, h, alto, bajo)
    print()
    print(f"Theta = {alto[0][0]:.2f}, theta = {bajo[0][0]:.2f}")

    bajo_impl, alto_impl = estimadores(niveles, CASO["K"], "compra", CASO["r"], CASO["T"], B)
    assert np.isclose(bajo[0][0], bajo_impl) and np.isclose(alto[0][0], alto_impl)
    print("Coincide con estimadores() de broadie_glasserman.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
