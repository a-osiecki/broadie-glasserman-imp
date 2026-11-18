# Broadie-Glasserman

Implementación del método de Broadie y Glasserman (1997) para valorar opciones americanas mediante simulación de Monte Carlo. El método arma árboles simulados no recombinantes y calcula dos estimadores, uno sesgado hacia arriba y otro hacia abajo, que acotan el precio y permiten construir un intervalo de confianza.

La corrida principal valora una opción de compra americana con `S0 = 100`, `K = 100`, `r = 0.05`, `delta = 0.10`, `sigma = 0.20` y `T = 1`, con `d = 3` fechas de ejercicio.

## Requisitos

- Python 3
- NumPy
- Matplotlib, solo para generar la figura

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Los dos primeros scripts usan solo NumPy. `figura_convergencia.py` además necesita Matplotlib.

## Archivos

- `broadie_glasserman.py`: implementación del método y del caso de estudio. Imprime el caso `b = 3`, la tabla de convergencia y la validación.
- `ejemplo.py`: ejemplo de un solo árbol con `b = 3` y `d = 2`. Imprime el árbol con los estimadores de cada nodo y verifica el resultado contra `estimadores()`.
- `conclusiones.py`: reproduce los números de las conclusiones: nodos por árbol, 400 semillas y el ancho del intervalo con el mismo presupuesto de nodos.
- `figura_convergencia.py`: genera `figura_convergencia.pdf` con la convergencia de los estimadores.

## Uso

```bash
python broadie_glasserman.py   # caso b = 3, tabla de convergencia y validación
python ejemplo.py              # árbol del ejemplo
python conclusiones.py         # números de las conclusiones (tarda unos minutos)
python figura_convergencia.py  # figura de convergencia
```

La corrida principal usa semilla 2026, `n = 100` árboles y `d = 3` fechas. Los resultados son reproducibles.
