# Eliuth Balderas Neri
# A01703315

# FILES TXTs
#   ./figuras/triangulo.txt
#   ./figuras/cuadrado.txt
#   ./figuras/rombo.txt
#   ./figuras/circulo.txt  -> Este es el archivo de test (no visto)

NOISE_PCTS = [0, 10, 20, 30]  # porcentajes de bits que se voltean
STRIDE     = 7                # salto para distribuir los flips


# Convierte de binario a bipolar
def to_bipolar(v01):
    out = []
    i = 0
    L = len(v01)
    while i < L:
        out.append(1 if v01[i] == 1 else -1)
        i += 1
    return out


# de bipolar a binario
def to_binary(vpm):
    out = []
    i = 0
    L = len(vpm)
    while i < L:
        out.append(1 if vpm[i] >= 0 else 0)
        i += 1
    return out

# se apalana la matriz
def flatten_matrix(mat):
    out = []
    i = 0
    R = len(mat)
    while i < R:
        row = mat[i]
        j = 0
        C = len(row)
        while j < C:
            out.append(row[j])
            j += 1
        i += 1
    return out

#de vector a grid
def reshape_vector_to_grid(vec, rows, cols):
    g = []
    k = 0
    r = 0
    while r < rows:
        fila = []
        c = 0
        while c < cols:
            fila.append(vec[k])
            k += 1
            c += 1
        g.append(fila)
        r += 1
    return g

# distancia de hamming entre los vectores
def hamming_distance(a, b):
    i = 0
    L = len(a)
    d = 0
    while i < L:
        if a[i] != b[i]:
            d += 1
        i += 1
    return d

# signo de hopfield 
def sign(x):
    if x >= 0:
        return 1
    return -1


def print_grid_ascii(vec, rows, cols, one='█', zero='·'):
    g = reshape_vector_to_grid(vec, rows, cols)
    r = 0
    while r < rows:
        line = ""
        c = 0
        while c < cols:
            line = line + (one if g[r][c] == 1 else zero)
            c += 1
        print(line)
        r += 1

#clase hopfield
class Hopfield:
    #red de neuronas
    def __init__(self, N):
        self.N = N
        # Matriz de pesos NxN inicializada en 0.0
        self.W = []
        i = 0
        while i < N:
            self.W.append([0.0] * N)
            i += 1
        # se guardan copias de los patrones en bipolar
        self.memory = []
    # se entrena con los patrones binarios
    def train(self, patterns_bin):
        self.memory = []
        N = self.N
        # Reiniciamos W a 0
        self.W = []
        i = 0
        while i < N:
            self.W.append([0.0] * N)
            i += 1

        # Para cada patrón binario p, convertimos a bipolar x y acumulamos x x^T
        i_pat = 0
        Lp = len(patterns_bin)
        while i_pat < Lp:
            p = patterns_bin[i_pat]
            x = to_bipolar(p[:])
            self.memory.append(x)

            i = 0
            while i < N:
                xi = x[i]
                j = 0
                while j < N:
                    if i != j:
                        self.W[i][j] = self.W[i][j] + (xi * x[j])
                    j += 1
                i += 1
            i_pat += 1

    # E(x) = -1/2 x^T W x 
    def energy(self, x_bip):
        N = self.N
        s = 0.0
        i = 0
        while i < N:
            j = 0
            Wi = self.W[i]
            xi = x_bip[i]
            while j < N:
                s = s + Wi[j] * xi * x_bip[j]
                j += 1
            i += 1
        return -0.5 * s

    # convergencia
    def recall(self, probe_bin, max_iters=200, mode="async"):
        # Estado bipolar
        x = to_bipolar(probe_bin[:])
        last = None
        it = 0

        while it < max_iters:
            it += 1
            if mode == "sync":
                # h = W x (producto matriz-vector)
                h = []
                i = 0
                while i < self.N:
                    s = 0.0
                    j = 0
                    Wi = self.W[i]
                    while j < self.N:
                        s = s + Wi[j] * x[j]
                        j += 1
                    h.append(s)
                    i += 1
                # aplica signo a toda la capa
                i = 0
                while i < self.N:
                    x[i] = sign(h[i])
                    i += 1
            else:
                # se actualiza secuencialmente cada neurona
                i = 0
                while i < self.N:
                    s = 0.0
                    j = 0
                    Wi = self.W[i]
                    while j < self.N:
                        s = s + Wi[j] * x[j]
                        j += 1
                    x[i] = sign(s)
                    i += 1

            # si no cambió en la iteración previa, entnces  cumple con criterio de convergencia: 
            if last is not None:
                same = True
                k = 0
                while k < self.N:
                    if x[k] != last[k]:
                        same = False
                        break
                    k += 1
                if same:
                    return to_binary(x), True, it, self.energy(x)
            last = x[:]

        # No convergió dentro del límite
        return to_binary(x), False, max_iters, self.energy(x)

    # se devuelve el patrón más cercano
    def nearest_stored_pattern(self, x_bin):
        best_i = -1
        best_d = 999999
        i = 0
        L = len(self.memory)
        while i < L:
            mb = to_binary(self.memory[i])
            d = hamming_distance(mb, x_bin)
            if d < best_d:
                best_d = d
                best_i = i
            i += 1
        return best_i, best_d



# se lee txt
def read_pattern_txt(path):
    rows = []
    try:
        f = open(path, "r", encoding="utf-8")
    except:
        print("ERROR: no se pudo abrir:", path)
        return None, 0, 0

    for line in f:
        s = line.strip()
        if s != "":
            rows.append(s)
    f.close()

    if len(rows) == 0:
        print("ERROR: archivo vacío:", path)
        return None, 0, 0

    r = len(rows)
    c = len(rows[0])

    # Valida ancho constante para evitar errores 
    k = 0
    while k < r:
        if len(rows[k]) != c:
            print("ERROR: filas con longitudes distintas en:", path)
            return None, 0, 0
        k += 1

    # de matriz 0/1 y luego a vector
    mat = []
    i = 0
    while i < r:
        fila = []
        j = 0
        s = rows[i]
        while j < c:
            if s[j] == '1':
                fila.append(1)
            else:
                fila.append(0)
            j += 1
        mat.append(fila)
        i += 1

    return flatten_matrix(mat), r, c

# nos da el porcentaje de ruido determinista
def flip_bits_by_percent_stride(vec, pct, stride, start_offset):
    out = vec[:]
    L = len(out)
    if pct <= 0:
        return out

    # cantidad de bits a voltear, al menos 1
    k = (L * pct) // 100
    if k < 1:
        k = 1  

    i = start_offset % L
    flips = 0
    while flips < k and flips < L:
        out[i] = 1 - out[i]
        i = (i + stride) % L
        flips += 1
    return out


def int_log2_floor(n):
    if n <= 1:
        return 0
    k = 0
    v = n
    while v > 1:
        v = v // 2
        k += 1
    return k


#estimaciones de capacidad 
def capacity_estimates_classic(N):
    p1 = (15 * N) // 100
    l2 = int_log2_floor(N)
    if l2 == 0:
        p2 = N // 2
    else:
        p2 = N // (2 * l2)
    return p1, p2

# files de entrenamiento y test
TRAIN_FILES = [
    "figuras/triangulo.txt",
    "figuras/cuadrado.txt",
    "figuras/rombo.txt",
]

# no visto ->test
TEST_FILES = [
    "figuras/circulo.txt",  
]


def load_patterns(file_list):
    vecs = []
    names = []
    rc = None
    idx = 0
    L = len(file_list)
    while idx < L:
        path = file_list[idx]
        v, r, c = read_pattern_txt(path)
        if v is None:
            print("Aviso: se omite", path)
            idx += 1
            continue
        if rc is None:
            rc = (r, c)
        else:
            if rc != (r, c):
                print("ERROR: tamaños distintos entre patrones. Esperado", rc, "y llegó", (r, c))
                return None, None, None
        vecs.append(v)
        nm = path.replace("\\", "/").split("/")[-1]
        if len(nm) >= 4 and nm[-4:] == ".txt":
            nm = nm[:-4]
        names.append(nm)
        idx += 1
    return vecs, names, rc


# maximo procentaje de ruido
def max_pct_recovery(net, vec, stride, start_offset):
    best = 0
    pct = 0
    while pct <= 100:
        probe = flip_bits_by_percent_stride(vec, pct, stride, start_offset)
        final, conv, it, E = net.recall(probe, max_iters=100)
        if final == vec:
            best = pct
            pct += 1
        else:
            break
    return best

# muestra el patrón con el máximo porcentaje de ruido y uno porciento más para contrastar el fallo
def show_threshold_demo(net, name, vec, rows, cols, stride, start_offset, train_names):
    tol = max_pct_recovery(net, vec, stride, start_offset)  
    print(f"\n[{name}] Max ruido = {tol}%")

    # Caso 1: máximo ruido
    probe_ok = flip_bits_by_percent_stride(vec, tol, stride, start_offset)
    final_ok, conv_ok, it_ok, E_ok = net.recall(probe_ok, max_iters=100)
    ok_ok = (final_ok == vec)
    print(f"  Con {tol}% -> conv:{conv_ok} iters:{it_ok} ok:{ok_ok} | Energía:{E_ok}")
    print_grid_ascii(final_ok, rows, cols)

    # Caso 2: un paso más allá del umbral (debería fallar o desviarse)
    if tol < 100:
        over = tol + 1
    else:
        over = 99 if tol == 100 else max(tol - 1, 1)

    probe_bad = flip_bits_by_percent_stride(vec, over, stride, start_offset)
    final_bad, conv_bad, it_bad, E_bad = net.recall(probe_bad, max_iters=100)
    idx_m, dist_m = net.nearest_stored_pattern(final_bad)
    nearest = train_names[idx_m] if idx_m >= 0 else "?"
    ok_bad = (final_bad == vec)
    print(f"  Con {over}% -> conv:{conv_bad} iters:{it_bad} ok:{ok_bad} | nearest:{nearest} dist:{dist_m} | Energía:{E_bad}")
    print_grid_ascii(final_bad, rows, cols)



def main():
    train_vecs, train_names, rc = load_patterns(TRAIN_FILES)
    if train_vecs is None:
        print("No fue posible cargar los patrones de entrenamiento.")
        return

    r, c = rc
    N = r * c
    p1, p2 = capacity_estimates_classic(N)
    print("N =", N, "| Capacidad aprox -> 0.15N =", p1, ", N/(2 log2 N) ~", p2)

    net = Hopfield(N)
    net.train(train_vecs)

    print("\nMáximo porcentaje de ruido por patrón)")
    i = 0
    while i < len(train_names):
        name = train_names[i]
        vec  = train_vecs[i]
        tol = max_pct_recovery(net, vec, STRIDE, start_offset=i)
        print("-", name + ":", str(tol) + "%")
        i += 1

    print("\nDistorción")
    i = 0
    while i < len(train_names):
        name = train_names[i]
        vec  = train_vecs[i]
        show_threshold_demo(net, name, vec, r, c, STRIDE, start_offset=i, train_names=train_names)
        i += 1


    # patron no visto
    print("\nPatrón NO visto con ruido por porcentaje")
    test_vecs, test_names, _ = load_patterns(TEST_FILES)
    if test_vecs is None:
        print("No fue posible cargar el patrón de prueba.")
        return

    j = 0
    while j < len(test_names):
        name = test_names[j]
        vec  = test_vecs[j]
        for pct in NOISE_PCTS:
            probe = vec[:] if pct == 0 else flip_bits_by_percent_stride(vec, pct, STRIDE, start_offset=j)
            final, conv, it, E = net.recall(probe, max_iters=100)

            # Más parecido al INPUT vs. al ESTADO FINAL
            idx_in,  dist_in  = net.nearest_stored_pattern(vec)
            idx_out, dist_out = net.nearest_stored_pattern(final)
            nearest_in  = train_names[idx_in]  if idx_in  >= 0 else "?"
            nearest_out = train_names[idx_out] if idx_out >= 0 else "?"

            print("\n[", name, "] ruido =", str(pct)+"%  -> convergió:", conv, "iters:", it, "| Energía:", E)
            print("Más parecido (input):", nearest_in,  "dist:", dist_in)
            print("Más parecido (final):", nearest_out, "dist:", dist_out)
            print_grid_ascii(final, r, c)
        j += 1

main()
