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

        #insert your code here

        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'
        return code

    def decode(self,code):
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        #insert your code here

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)




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
        galois_field=galois.GF(p**m,prim_poly)


        msg = galois_field(np.random.randint(0,2**8-1,(5,10))) # Generate a random message of 5 information words

        code = rs.encode(msg) # Encode this message

        # Introduce errors
        code[1,[2, 17]] = code[1,[4, 17]]+galois_field(1)
        code[2,7] = 0;
        code[3,[3, 1, 18, 19, 5]] = np.random.randint(0,2**8-1,(1,5))
        code[4,[3, 1, 18, 19, 5, 12]] = np.random.randint(0,2**8-1,(1,6))


        [decoded,nERR] = rs.decode(code) # Decode


        print(nERR)
        assert((decoded[0:4,:] == msg[0:4,:]).all())
        pass

if __name__ == "__main__":
    g = RSCode.makeGenerator(m=8, t=2, m0=0)
    print(g)
    print(g.degree)  # moet = 2t