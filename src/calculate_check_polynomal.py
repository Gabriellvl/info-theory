# GF(2^8) arithmetic for CD standard (primitive polynomial 0x11D)

PRIM = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1

def gf_add(a, b):
    return a ^ b

def gf_mul(a, b):
    """
    Multiplies two elements in GF(2^8) using the Russian peasant multiplication algorithm.

    Args:
        a (int): The first operand, an integer representing an element in GF(2^8).
        b (int): The second operand, an integer representing an element in GF(2^8).

    Returns:
        int: The product of a and b in GF(2^8).

    Note:
        This function assumes the existence of a global variable `PRIM` representing the primitive
        polynomial for GF(2^8) reduction. The multiplication is performed bitwise, with reduction
        by the primitive polynomial when necessary.
    """
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= PRIM
    return res

def gf_pow(a, n):
    res = 1
    while n > 0:
        if n & 1:
            res = gf_mul(res, a)
        a = gf_mul(a, a)
        n >>= 1
    return res

def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i+j] ^= gf_mul(p[i], q[j])
    return r

# Build log/antilog tables for α = 0x02
def build_tables():
    alpha = 0x02
    exp = [1] * 255
    log = [0] * 256
    for i in range(1, 255):
        exp[i] = gf_mul(exp[i-1], alpha)
    for i in range(255):
        log[exp[i]] = i
    return log, exp

log_table, exp_table = build_tables()

def byte_to_alpha_sum(b):
    """Return GF(2^8) element as sum of a^k terms."""
    if b == 0:
        return "0"
    terms = []
    for i in range(8):
        if b & (1 << i):
            # Convert basis element x^i into α^k
            # x = α^1 in this field
            # So x^i = α^i
            terms.append(f"a^{i}")
    return " + ".join(terms)

# ---------------------------------------------------------
#   MAIN FUNCTION: build the C1 check polynomial h(x)
# ---------------------------------------------------------

def make_check_polynomial():
    alpha = 0x02  # primitive element in CD GF(2^8)
    h = [1]       # start with polynomial "1"

    # h(x) = Π_{i=4}^{31} (x - α^i)
    for i in range(4, 32):
        root = gf_pow(alpha, i)
        factor = [root, 1]   # (x - α^i) = root + 1*x
        h = poly_mul(h, factor)

    return h

def make_check_polynomial_alpha():
    alpha = 0x02
    h = [1]  # start with polynomial "1"

    # h(x) = Π_{i=4}^{31} (x - α^i)
    for i in range(4, 32):
        root = exp_table[i % 255]
        factor = [root, 1]
        h = poly_mul(h, factor)

    # Convert coefficients to α-exponent sums
    poly_terms = []
    deg = len(h) - 1

    for i in range(deg, -1, -1):
        coef = h[i]
        coef_str = byte_to_alpha_sum(coef)

        if i == 0:
            poly_terms.append(f"{coef_str}\n")
        elif i == 1:
            poly_terms.append(f"({coef_str}) x\n")
        else:
            poly_terms.append(f"({coef_str}) x^{i}\n")

    poly_str = " + ".join(poly_terms)
    return h, poly_str


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    coeffs, poly = make_check_polynomial_alpha()
    print("Check polynomial h(x) in a-exponent form:")
    print(poly)