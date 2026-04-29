import galois
import numpy as np

class RSCode:
    def __init__(self, m,t,l,m0):
        self.m = m #GF(2^m) field
        self.t = t #Error correction capability
        self.n = 2**m-1 #Code length
        self.k = self.n-2*t #Information length
        self.l = l #Shortened information length (-> shortened code length = l+n-k)
        self.m0 = m0 #m0 of the Reed-Solomon code, determines first root of generator
        
        self.g = self.makeGenerator(m,t,m0) # generator polynomial represented by a galois.Poly variable

    def encode(self,msg):
        # Systematically encodes information words using the Reed-Solomon code
        # Input:
        #  -msg: a 2D array of galois.GF elements, every row corresponds with a GF(2^m) information word of length self.l
        # Output:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword corresponding to systematic Reed-Solomon coding of the corresponding information word
        assert np.shape(msg)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(msg) is galois.GF(2**self.m) , 'each element of msg  must be a galois.GF element'

        GF = galois.GF(2**self.m)
        msg = GF(msg) # Ensure msg is a GF array

        n_parity = self.n - self.k  # number of parity symbols
        g = self.g      # generator polynomial (galois.Poly over GF)

        n_rows = msg.shape[0]
        code = GF.Zeros((n_rows, self.l + n_parity))

        for r in range(n_rows):
            # Interpret the row as coefficients of m(x) in descending degree order:
            # msg[r,0] is coeff of x^(l-1), ..., msg[r,l-1] is constant term.

            # Multiply by x^n_parity -> append n_parity zeros to the coefficient list (descending order)
            m_shift = galois.Poly(np.concatenate([msg[r, :], GF.Zeros(n_parity)]), field=GF)

            # Remainder r(x) = m(x)*x^n_parity mod g(x)
            remainder = m_shift % g

            # Pad remainder to exactly n_parity symbols (descending degrees n_parity-1..0)
            rem_coeffs = remainder.coeffs
            if rem_coeffs.size < n_parity:
                rem_coeffs = np.concatenate([GF.Zeros(n_parity - rem_coeffs.size), rem_coeffs])

            # Systematic codeword: [message | parity]
            code[r, :] = np.concatenate([msg[r, :], rem_coeffs])

        assert np.shape(code)[1] == self.l + self.n - self.k, \
            "the number of columns must be equal to self.l+self.n-self.k"
        assert type(code) is GF, "each element of code must be a galois.GF element"

        return code

    def decode(self,code): #Gemini Version; only works for m=1
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        #insert your code here

        # ---------------------------------------------------------
        # Reed-Solomon Decoding: Euclid's Method (Algorithm 2.4.4)
        # ---------------------------------------------------------
        
        GF = type(code)               # Extract the precise Galois Field class from the input
        num_rows, N = code.shape
        two_t = self.n - self.k       # Number of parity symbols (2t)
        t = two_t // 2                # Error correction capability
        alpha = GF.primitive_element  # Primitive element of GF(2^m)

        # Initialize output arrays
        decoded = GF.Zeros((num_rows, self.l))
        nERR = np.full(num_rows, -1, dtype=int)  # Default to -1 (Decoder Failure)

        for i in range(num_rows):
            r = code[i]
            r_poly = galois.Poly(r)  # R(x) where r[0] is x^{N-1}

            # 1. Syndrome Computation: Evaluate R(x) at alpha^1, ..., alpha^{2t}
            S = GF.Zeros(two_t)
            has_error = False
            for j in range(1, two_t + 1):
                S[j-1] = r_poly(alpha**j)
                if S[j-1] != 0:
                    has_error = True

            if not has_error:
                # No errors detected, extract the information word directly
                decoded[i] = r[:self.l]  
                nERR[i] = 0
                continue

            # 2. Euclid's Algorithm (Algorithm 2.4.4)
            # galois.Poly expects highest degree coefficients first, so reverse the array
            S_poly = galois.Poly(S[::-1], field=GF)

            # Step 1: Initialization
            O_prev2 = galois.Poly([1] + [0]*two_t, field=GF) # Omega_{-1}(z) = z^{2t}
            L_prev2 = galois.Poly([0], field=GF)             # Lambda_{-1}(z) = 0
            O_prev1 = S_poly                                 # Omega_0(z) = S(z)
            L_prev1 = galois.Poly([1], field=GF)             # Lambda_0(z) = 1

            # Step 2-6: Iteration
            while O_prev1.degree >= t:
                # q_i(z) and remainder from polynomial division
                q, rem = divmod(O_prev2, O_prev1) 
                
                # Step 5: Update
                O_curr = rem  # rem is O_{i-2}(z) - q_i(z)O_{i-1}(z)
                L_curr = L_prev2 - q * L_prev1

                # Shift variables for next iteration
                O_prev2, O_prev1 = O_prev1, O_curr
                L_prev2, L_prev1 = L_prev1, L_curr

            Omega = O_prev1
            Lambda = L_prev1

            # Check for decoder failure: Lambda(0) must not be 0
            l0 = Lambda.coeffs[-1] if Lambda.degree >= 0 else GF(0)
            if l0 == 0:
                continue  # Decoder failure, nERR[i] remains -1

            # Normalize Lambda and Omega so that Lambda(0) = 1
            l0_inv = GF(1) / l0
            Lambda = Lambda * l0_inv
            Omega = Omega * l0_inv

            # 3. Chien Search: Find roots of Lambda(z)
            error_locs = []
            error_powers = []

            for m in range(N):
                p_m = N - 1 - m             # Power of x corresponding to index m
                z = alpha**(-int(p_m))      # Root of Lambda(z) is the inverse of the error location
                if Lambda(z) == 0:
                    error_locs.append(m)
                    error_powers.append(p_m)

            # Check decoder failure: number of roots != degree of Lambda
            if len(error_locs) != Lambda.degree:
                continue

            # 4. Forney Algorithm: Calculate error values
            Lambda_prime = Lambda.derivative()
            e = GF.Zeros(N)
            decoder_failure = False

            for m, p_m in zip(error_locs, error_powers):
                z = alpha**(-int(p_m))
                num = Omega(z)
                den = Lambda_prime(z)
                
                if den == 0:
                    decoder_failure = True
                    break
                    
                # e_k = - (Omega(z_k) / Lambda'(z_k)) * X_k^{1-m_0}
                # Since m_0 = 1, X_k^{1-m_0} = 1.
                # In GF(2^m), addition and subtraction are identical, so we drop the minus sign.
                e[m] = num / den

            if decoder_failure:
                continue

            # 5. Correction and Extraction
            r_corrected = r + e
            decoded[i] = r_corrected[:self.l]  # Assumes systematic encoding, info at the front
            nERR[i] = len(error_locs)

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)
    

    def decode_1st_draft(self,code):
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        #insert your code here
        # ---------------------------------------------------------
        # Reed-Solomon Decoding: Euclid's Method (Algorithm 2.4.4)
        # ---------------------------------------------------------
        
        GF = type(code)               # Extract the precise Galois Field class from the input
        num_rows, N = code.shape
        two_t = self.n - self.k       # Number of parity symbols (2t)
        t = two_t // 2                # Error correction capability
        alpha = GF.primitive_element  # Primitive element of GF(2^m)

        # Initialize output arrays
        decoded = GF.Zeros((num_rows, self.l))
        nERR = np.full(num_rows, -1, dtype=int)  # Default to -1 (Decoder Failure)

        for i in range(num_rows):
            r = code[i]
            r_poly = galois.Poly(r)  # R(x) where r[0] is x^{N-1}

            # 1. Syndrome Computation: Evaluate R(x) at alpha^1, ..., alpha^{2t}
            S = GF.Zeros(two_t)
            has_error = False
            for j in range(1, two_t + 1):
                S[j-1] = r_poly(alpha**j)
                if S[j-1] != 0:
                    has_error = True

            if not has_error:
                # No errors detected, extract the information word directly
                decoded[i] = r[:self.l]  
                nERR[i] = 0
                continue

            # 2. Euclid's Algorithm (Algorithm 2.4.4)
            # S(z) = S_1 + S_2 z + ... + S_{2t} z^{2t-1}
            # galois.Poly expects highest degree coefficients first, so reverse the array
            S_poly = galois.Poly(S[::-1], field=GF)

            # Step 1: Initialization
            O_prev2 = galois.Poly([1] + [0]*two_t, field=GF) # Omega_{-1}(z) = z^{2t}
            L_prev2 = galois.Poly([0], field=GF)             # Lambda_{-1}(z) = 0
            O_prev1 = S_poly                                 # Omega_0(z) = S(z)
            L_prev1 = galois.Poly([1], field=GF)             # Lambda_0(z) = 1

            # Step 2-6: Iteration
            while O_prev1.degree >= t:
                # q_i(z) and remainder from polynomial division
                q, rem = divmod(O_prev2, O_prev1) 
                
                # Step 5: Update
                O_curr = rem  # rem is O_{i-2}(z) - q_i(z)O_{i-1}(z)
                L_curr = L_prev2 - q * L_prev1

                # Shift variables for next iteration
                O_prev2, O_prev1 = O_prev1, O_curr
                L_prev2, L_prev1 = L_prev1, L_curr

            Omega = O_prev1
            Lambda = L_prev1

            # Check for decoder failure: Lambda(0) must not be 0
            l0 = Lambda.coeffs[-1] if Lambda.degree >= 0 else GF(0)
            if l0 == 0:
                continue  # Decoder failure, nERR[i] remains -1

            # Normalize Lambda and Omega so that Lambda(0) = 1
            l0_inv = GF(1) / l0
            Lambda = Lambda * l0_inv
            Omega = Omega * l0_inv

            # 3. Chien Search: Find roots of Lambda(z)
            error_locs = []
            error_powers = []

            for m in range(N):
                p_m = N - 1 - m             # Power of x corresponding to index m
                z = alpha**(-int(p_m))      # Root of Lambda(z) is the inverse of the error location
                if Lambda(z) == 0:
                    error_locs.append(m)
                    error_powers.append(p_m)

            # Check decoder failure: number of roots != degree of Lambda
            if len(error_locs) != Lambda.degree:
                continue

            # 4. Forney Algorithm: Calculate error values
            Lambda_prime = Lambda.derivative()
            e = GF.Zeros(N)
            decoder_failure = False

            for m, p_m in zip(error_locs, error_powers):
                z = alpha**(-int(p_m))
                num = Omega(z)
                den = Lambda_prime(z)
                
                if den == 0:
                    decoder_failure = True
                    break
                    
                # e_k = - (Omega(z) / Lambda'(z)) * alpha^{p_m}
                # Note: In GF(2^m), addition is subtraction, so minus sign is ignored
                e[m] = (num / den) * (alpha**int(p_m))

            if decoder_failure:
                continue

            # 5. Correction and Extraction
            r_corrected = r + e
            decoded[i] = r_corrected[:self.l]  # Assumes systematic encoding, info at the front
            nERR[i] = len(error_locs)

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)


    

    def test_decode(self, code): #enkel om te testen _ NIET INDIENEN
        # Decode Reed-Solomon codes
        # Input:
        #  - code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword
        #          of length self.l + self.n - self.k  (shortened codeword length)
        # Output:
        #  - decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information
        #             word of length self.l
        #  - nERR: 1D numpy array containing the number of corrected symbols for every codeword,
        #          -1 if error correction failed

        assert np.shape(code)[1] == self.l + self.n - self.k, \
            'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m), \
            'each element of code  must be a galois.GF element'

        GF = type(code)  # Preserve the exact field instance used by the input array

        # Reuse an existing RS object if your class already created one (common in assignments),
        # otherwise construct a compatible one.
        rs = None
        for attr in ("rs", "RS", "reed_solomon", "ReedSolomon"):
            if hasattr(self, attr):
                rs = getattr(self, attr)
                break

        if rs is None:
            # If your course uses a non-default starting root 'c', store it as self.c; otherwise default is fine.
            c = getattr(self, "c", 1)
            rs = galois.ReedSolomon(self.n, self.k, field=GF, c=c)

        # Decode; errors=True returns (message, n_errors) where n_errors is -1 on failure. [1](https://mhostetter.github.io/galois/latest/api/galois.ReedSolomon.decode/)
        decoded, nERR = rs.decode(code, errors=True)

        # Ensure nERR is a 1D NumPy array even if a scalar is returned
        nERR = np.atleast_1d(np.array(nERR, dtype=int))

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m), 'each element of decoded must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR)) == 1, 'nERR must be a 1D numpy array'

        return (decoded, nERR)


    def decode_temp(self, code):#werkt niet
        """
        Decode Reed-Solomon codes (syndrome-based, Berlekamp–Massey + Chien + Forney)
        WITHOUT using galois.ReedSolomon (or other RS builtin).

        Input:
        code: 2D array of GF elements, each row is a shortened systematic codeword
                length = self.l + self.n - self.k  = (k-s) + (n-k) = n-s
        Output:
        decoded: 2D array of GF elements, each row is information word length self.l
        nERR: 1D numpy array, number of corrected symbols per row, -1 if failed
        """

        assert np.shape(code)[1] == self.l + self.n - self.k, \
            'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m), \
            'each element of code  must be a galois.GF element'

        GF = type(code)

        # --- Parameters ---
        n = self.n
        k = self.k
        parity = n - k                 # = 2t for standard RS
        t = parity // 2
        s_short = k - self.l           # shortening amount (message shortened by s_short)
        c = getattr(self, "c", 1)      # first consecutive root (fcr); common default is 1

        alpha = GF.primitive_element
        two_t = parity

        # --- Helpers ---
        def poly_eval_ascending(coeffs, x):
            """
            Evaluate polynomial with coefficients in ascending order:
            p(x) = coeffs[0] + coeffs[1] x + ... + coeffs[d] x^d
            """
            y = GF(0)
            for a in reversed(coeffs):
                y = y * x + a
            return y

        def codeword_eval_descending(vec, x):
            """
            Evaluate codeword polynomial where vec is in descending-power coefficient order:
            r(x) = vec[0] x^(n-1) + ... + vec[n-1]
            Horner on descending coefficients:
            """
            y = GF(0)
            for a in vec:
                y = y * x + a
            return y

        def syndromes(r_full):
            """
            Compute S_i = r( alpha^(c+i) ) for i=0..(2t-1)
            """
            S = GF.Zeros(two_t)
            for i in range(two_t):
                x = alpha ** (c + i)
                S[i] = codeword_eval_descending(r_full, x)
            return S

        def berlekamp_massey(S):
            """
            Berlekamp–Massey over GF for syndrome sequence S[0..2t-1].
            Returns locator polynomial Lambda in ascending order and its degree L.
            """
            # Lambda(x) and B(x) stored as length (2t+1) buffers
            Lambda = GF.Zeros(two_t + 1)
            B = GF.Zeros(two_t + 1)
            Lambda[0] = 1
            B[0] = 1

            L = 0
            m = 1
            b = GF(1)

            for n_idx in range(two_t):
                # discrepancy d = S[n] + sum_{i=1..L} Lambda[i] * S[n-i]
                d = S[n_idx]
                for i in range(1, L + 1):
                    d += Lambda[i] * S[n_idx - i]

                if d == 0:
                    m += 1
                else:
                    T = Lambda.copy()
                    coef = d / b
                    # Lambda(x) = Lambda(x) - coef * x^m * B(x)
                    # subtraction is field subtraction (same as addition in GF(2^m))
                    for i in range(0, two_t + 1 - m):
                        Lambda[i + m] -= coef * B[i]

                    if 2 * L <= n_idx:
                        L = n_idx + 1 - L
                        B = T
                        b = d
                        m = 1
                    else:
                        m += 1

            # Trim to degree L
            return Lambda[:L + 1], L

        def locator_derivative(Lambda):
            """
            Formal derivative Lambda'(x) = sum_{i=1..L} i * Lambda[i] x^(i-1)
            (valid generally; in characteristic 2, even i terms vanish automatically)
            """
            L = len(Lambda) - 1
            if L <= 0:
                return GF.Zeros(1)
            d = GF.Zeros(L)
            for i in range(1, L + 1):
                d[i - 1] = GF(i) * Lambda[i]
            return d

        def omega_poly(S, Lambda):
            """
            Omega(x) = (S(x) * Lambda(x)) mod x^(2t)
            as used for Forney-style evaluation. [3](https://en.wikipedia.org/wiki/Forney_algorithm)[1](https://courses.cs.duke.edu/spring11/cps296.3/decoding_rs.pdf)
            """
            # convolution in GF
            prod = GF.Zeros(min(two_t, len(S) + len(Lambda) - 1))
            for i in range(len(S)):
                for j in range(len(Lambda)):
                    ij = i + j
                    if ij >= two_t:
                        break
                    prod[ij] += S[i] * Lambda[j]
            return prod

        # --- Main loop over codewords ---
        N = code.shape[0]
        decoded = GF.Zeros((N, self.l))
        nERR = np.full((N,), -1, dtype=int)

        for row in range(N):
            r_short = code[row]

            # Reconstruct full-length systematic codeword by prefixing the shortened-away symbols with zeros.
            # Shortened word length is n - s_short.
            r_full = np.concatenate([GF.Zeros(s_short), r_short])

            # 1) Syndromes
            S = syndromes(r_full)

            # If no errors, extract message directly
            if np.all(S == 0):
                msg_full = r_full[:k]
                decoded[row] = msg_full[s_short:]   # drop the padded leading zeros
                nERR[row] = 0
                continue

            # 2) Berlekamp–Massey: find error locator polynomial Lambda
            Lambda, L = berlekamp_massey(S)

            # If degree too large, fail
            if L > t:
                decoded[row] = GF.Zeros(self.l)
                nERR[row] = -1
                continue

            # 3) Chien search: find error positions.
            # We test each symbol position pos and check Lambda(X_pos^{-1}) == 0
            # with X_pos = alpha^(n-1-pos) (consistent with descending polynomial form). [2](https://en.wikipedia.org/wiki/Chien_search)[1](https://courses.cs.duke.edu/spring11/cps296.3/decoding_rs.pdf)
            err_pos = []
            for pos in range(n):
                X = alpha ** (n - 1 - pos)
                if poly_eval_ascending(Lambda, 1 / X) == 0:
                    err_pos.append(pos)

            # If number of roots doesn't match degree, fail
            if len(err_pos) != L:
                decoded[row] = GF.Zeros(self.l)
                nERR[row] = -1
                continue

            # 4) Compute Omega and Forney magnitudes
            Omega = omega_poly(S, Lambda)
            Lambda_p = locator_derivative(Lambda)

            # Forney error value formula (general fcr c):
            # e_j = - X_j^(1-c) * Omega(X_j^{-1}) / Lambda'(X_j^{-1}) [3](https://en.wikipedia.org/wiki/Forney_algorithm)
            r_corr = r_full.copy()
            for pos in err_pos:
                X = alpha ** (n - 1 - pos)
                Xinv = 1 / X

                denom = poly_eval_ascending(Lambda_p, Xinv)
                if denom == 0:
                    # derivative zero at root -> cannot evaluate reliably
                    err_pos = None
                    break

                numer = poly_eval_ascending(Omega, Xinv)
                e = - (X ** (1 - c)) * numer / denom
                r_corr[pos] -= e

            if err_pos is None:
                decoded[row] = GF.Zeros(self.l)
                nERR[row] = -1
                continue

            # 5) Verify corrected word (all syndromes should be zero)
            S2 = syndromes(r_corr)
            if not np.all(S2 == 0):
                decoded[row] = GF.Zeros(self.l)
                nERR[row] = -1
                continue

            # Count corrected symbols within the transmitted (shortened) part only
            corrected_in_short = sum(1 for p in err_pos if p >= s_short)
            nERR[row] = corrected_in_short

            # Extract message (systematic): first k symbols are message; drop padded shortening prefix
            msg_full = r_corr[:k]
            decoded[row] = msg_full[s_short:]

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m), 'each element of decoded must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR)) == 1, 'nERR must be a 1D numpy array'

        return (decoded, nERR)




    @staticmethod
    def makeGenerator(m, t, m0):
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(2^m)
        # Input:
        #  -m: order of the galois field is 2^m
        #  -t: error correction capability of the Reed-Solomon code
        #  -m0: determines the first root of the generator polynomial
        # Output:
        #  -generator: generator polynomial represented by a galois.Poly variable

        
        # Finite field GF(2^m)
        GF = galois.GF(2**m)

        print(f"Making the generator polynomial with t={t} and m0={m0} in the", GF.properties)

        # Primitive element of the field
        alpha = GF.primitive_element
        # print(f"Primitive element alpha: {alpha}")

        # Start with generator = 1
        generator = galois.Poly([1], field=GF)

        # g(x) = ∏_{i=0}^{2t-1} (x - α^{m0+i}) with 2t = n-k
        for i in range(2*t):
            root = alpha ** (m0 + i)
            factor = galois.Poly([1, root], field=GF)  # x + root (since char = 2)
            generator *= factor
                
        print("Generator polynomial g(x):", generator)

        assert type(generator) == type(galois.Poly([0],field=galois.GF(2**m))), 'generator must be a galois.Poly object'
        return generator

    @staticmethod
    def test():
        # function that illustrates how the other code of this class can be tested
        m0 = 1 # Also test with other values of m0!
        m=8
        t=5
        l=10
        rs = RSCode(m,t,l,m0) # Construct the RSCode object
        p=2
        prim_poly=galois.primitive_poly(p,m)
        galois_field=galois.GF(p**m,irreducible_poly=prim_poly)


        msg = galois_field(np.random.randint(0,2**m-1,(t,l))) # Generate a random message of 5 information words

        print("initial message ",msg)

        code = rs.encode(msg) # Encode this message

        # Introduce errors
        code[1,[2, 17]] = code[1,[4, 17]]+galois_field(1)
        code[2,7] = 0;
        code[3,[3, 1, 18, 19, 5]] = np.random.randint(0,2**m-1,(1,5))
        code[4,[3, 1, 18, 19, 5, 12]] = np.random.randint(0,2**m-1,(1,6))


        [decoded,nERR] = rs.decode(code) # Decode
        print("Decoded: ", decoded)


        print(nERR)
        assert((decoded[0:4,:] == msg[0:4,:]).all())
        pass

if __name__ == "__main__":
    # g = RSCode.makeGenerator(m=8, t=2, m0=0)
    # print(g)
    # print(g.degree)  # moet = 2t
    RSCode.test()