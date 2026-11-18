#!/usr/bin/env python3
"""
Valoración de opciones americanas mediante simulación de Monte Carlo.
Método de Broadie y Glasserman (1997).

Curso: Modelos Matemáticos en Finanzas Cuantitativas
Universidad Nacional de Córdoba, 2026
Autores: Agustín Osiecki

La primera parte del archivo es el método: la simulación del árbol,
los estimadores alto y bajo y el intervalo de confianza, con la
ecuación de cada uno en su docstring. Después viene el caso de
estudio del informe: el caso b = 3, la tabla de convergencia y la
validación.

Uso:
    python broadie_glasserman.py
"""

from math import exp, sqrt
from statistics import NormalDist

import numpy as np


# ---------------------------------------------------------------------------
# El método.
# ---------------------------------------------------------------------------


def pago(s, K, tipo):
    """Pago de ejercicio h(t, s). "compra" es un call, "venta" un put."""
    if tipo == "compra":
        return np.maximum(s - K, 0.0)
    return np.maximum(K - s, 0.0)


def simular_arbol(s0, r, delta, sigma, T, d, b, rng):
    """Simula un árbol no recombinante.

    Devuelve los d+1 niveles del árbol. El nivel t es un arreglo con los
    b**t precios de la fecha t. Cada nodo genera b hijos independientes:

        S_hijo = S_padre · exp[(r - delta - sigma²/2)·dt + sigma·√dt·Z]

    con dt = T/d y Z ~ N(0,1). El árbol no recombina: dos nodos con el
    mismo precio son nodos distintos.
    """
    # shortcut: guarda el árbol entero, O(b**d) memoria; profundidad primero si d crece
    dt = T / d
    deriva = (r - delta - sigma**2 / 2) * dt
    volatilidad = sigma * sqrt(dt)

    niveles = [np.array([s0])]
    for _ in range(d):
        padres = niveles[-1]
        z = rng.standard_normal((padres.size, b))
        hijos = padres[:, None] * np.exp(deriva + volatilidad * z)
        niveles.append(hijos.ravel())
    return niveles


def estimadores(niveles, K, tipo, r, T, b):
    """Recursión hacia atrás sobre un árbol simulado.

    Devuelve (bajo, alto), el valor de cada estimador en la raíz. Sea
    h_t el pago de ejercicio en la fecha t, dt = T/d y
    descuento = e^(-r·dt). En la madurez ambos estimadores valen h_d.
    En cada nodo anterior, con j recorriendo los b hijos:

        alto_t = max(h_t, descuento · (1/b)·Σ_j alto_hijo_j)

        bajo_t = (1/b)·Σ_j Y_j
        Y_j = h_t                      si h_t ≥ otros_j
            = descuento · bajo_hijo_j  si no
        otros_j = (1/(b-1))·Σ(i≠j) descuento · bajo_hijo_i

    donde alto_hijo_j y bajo_hijo_i son los estimadores de los hijos
    del nodo.
    """
    d = len(niveles) - 1
    descuento = exp(-r * T / d)
    h = [pago(nivel, K, tipo) for nivel in niveles]

    # Estimador alto: inducción hacia atrás desde la madurez.
    alto = h[d]
    for t in range(d - 1, -1, -1):
        continuacion = descuento * alto.reshape(-1, b).mean(axis=1)
        alto = np.maximum(h[t], continuacion)

    # Estimador bajo: la rama j decide con las otras b-1 ramas.
    bajo = h[d]
    for t in range(d - 1, -1, -1):
        hijos = descuento * bajo.reshape(-1, b)
        otros = (hijos.sum(axis=1, keepdims=True) - hijos) / (b - 1)
        intrinseco = h[t][:, None]
        bajo = np.where(intrinseco >= otros, intrinseco, hijos).mean(axis=1)

    return float(bajo[0]), float(alto[0])


def precio(s0, K, tipo, r, delta, sigma, T, d, b, n, confianza=0.90, semilla=None):
    """Promedia n árboles independientes y arma el intervalo de confianza.

    Devuelve un diccionario con las medias, los errores estándar, el
    intervalo conservador y el precio puntual. Sea h_0 el pago de
    ejercicio inmediato, bajo y alto las medias sobre los n árboles,
    error_bajo y error_alto sus errores estándar y z el cuantil normal
    de la confianza:

        ic = (max(h_0, bajo - z·error_bajo), alto + z·error_alto)
        puntual = 0.5·max(h_0, bajo) + 0.5·alto
    """
    rng = np.random.default_rng(semilla)
    bajos = np.empty(n)
    altos = np.empty(n)
    for i in range(n):
        arbol = simular_arbol(s0, r, delta, sigma, T, d, b, rng)
        bajos[i], altos[i] = estimadores(arbol, K, tipo, r, T, b)

    z = NormalDist().inv_cdf(1 - (1 - confianza) / 2)
    h0 = float(pago(s0, K, tipo))
    bajo, alto = float(bajos.mean()), float(altos.mean())
    error_bajo = float(bajos.std(ddof=1) / sqrt(n))
    error_alto = float(altos.std(ddof=1) / sqrt(n))

    return {
        "bajo": bajo,
        "alto": alto,
        "error_bajo": error_bajo,
        "error_alto": error_alto,
        "ic": (max(h0, bajo - z * error_bajo), alto + z * error_alto),
        "puntual": 0.5 * max(h0, bajo) + 0.5 * alto,
    }


# ---------------------------------------------------------------------------
# Caso de estudio del informe: tablas y validación.
# ---------------------------------------------------------------------------


def fila(res):
    """Columnas comunes a las dos tablas."""
    return [
        f"{res['bajo']:.4f}",
        f"{res['alto']:.4f}",
        f"[{res['ic'][0]:.4f}, {res['ic'][1]:.4f}]",
        f"{res['puntual']:.4f}",
    ]


def tabla(encabezados, filas):
    """Imprime columnas alineadas a la derecha."""
    anchos = [max(len(e), *(len(f[i]) for f in filas)) for i, e in enumerate(encabezados)]
    print("  ".join(e.rjust(a) for e, a in zip(encabezados, anchos)))
    for f in filas:
        print("  ".join(c.rjust(a) for c, a in zip(f, anchos)))


def imprimir_caso(res, verdad):
    """Bloque del caso b = 3."""
    inf, sup = res["ic"]
    print("Caso b = 3")
    print(f"  {'Estimador bajo':<18}{res['bajo']:9.4f}   (error estándar {res['error_bajo']:.4f})")
    print(f"  {'Estimador alto':<18}{res['alto']:9.4f}   (error estándar {res['error_alto']:.4f})")
    print(f"  {'Intervalo 90%':<18}   [{inf:.4f}, {sup:.4f}]")
    print(f"  {'Precio puntual':<18}{res['puntual']:9.4f}")
    print(f"  {'Valor verdadero':<18}{verdad:9.4f}")
    print(f"  {'Error relativo':<18}{abs(res['puntual'] - verdad) / verdad:9.2%}")
    print()


def imprimir_convergencia(ramas, resultados, verdad):
    """Tabla 1 del informe: convergencia con la cantidad de ramas."""
    print("Convergencia con la cantidad de ramas")
    filas = [
        [str(b), *fila(res), f"{abs(res['puntual'] - verdad) / verdad:.2%}"]
        for b, res in zip(ramas, resultados)
    ]
    tabla(["b", "bajo", "alto", "IC 90%", "puntual", "error rel."], filas)
    print()


def imprimir_validacion(caso, d, n, semilla):
    """Tabla 2 del informe: comparación con los valores de referencia."""
    print("Validación contra Broadie y Glasserman (1997)")
    verdades = {90.0: 2.303, 100.0: 5.731, 110.0: 11.341}
    filas = []
    for s0, valor in verdades.items():
        res = precio(s0=s0, **caso, d=d, b=50, n=n, semilla=semilla)
        error = abs(res["puntual"] - valor) / valor
        filas.append([f"{s0:.0f}", *fila(res), f"{valor:.3f}", f"{error:.2%}"])
    tabla(["S(0)", "bajo", "alto", "IC 90%", "puntual", "verdadero", "error rel."], filas)


def main():
    """Corre el caso de estudio del informe."""
    caso = dict(K=100.0, tipo="compra", r=0.05, delta=0.10, sigma=0.20, T=1.0)
    d, n, semilla = 3, 100, 2026
    verdad = 5.731

    print("Método de Broadie-Glasserman")
    print("Opción de compra americana: S0 = 100, K = 100, r = 0.05, δ = 0.10, σ = 0.20, T = 1")
    print(f"d = {d}, n = {n}, confianza = 90%")
    print()

    r3 = precio(s0=100.0, **caso, d=d, b=3, n=n, semilla=semilla)
    imprimir_caso(r3, verdad)

    ramas = [2, 3, 5, 10, 20, 50]
    convergencia = [precio(s0=100.0, **caso, d=d, b=b, n=n, semilla=semilla) for b in ramas]
    imprimir_convergencia(ramas, convergencia, verdad)
    imprimir_validacion(caso, d, n, semilla)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
