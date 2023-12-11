import pstats
p = pstats.Stats('output.prof')
p.sort_stats('cumulative').print_stats(10)  # Esto imprimirá las 10 funciones principales en términos de tiempo acumulado.
