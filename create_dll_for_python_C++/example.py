import ctypes
import numpy


import ctypes

# Load DLL into memory.
n=10

hllDll = ctypes.WinDLL ("MathLibrary.dll")

hllDll.fprod.argtypes = [ctypes.c_double]
hllDll.fprod.restype  = ctypes.c_double

def fprod(n):
    return hllDll.fprod(float(n))

def main():
    fprod(n)

if __name__ == "__main__":
    main()

'''

%%time
fprod(41)
102334155.0
Wall time: 437 ms

def fibonacci(n):
    if n in (1, 2):
        return n-1
    return fibonacci(n - 1) + fibonacci(n - 2)

%%time
print(fibonacci(41))

102334155
Wall time: 37.1 s
 
'''