import ctypes
import os

cwd = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(cwd, "cephes.so")
cepheslib = ctypes.CDLL(lib_path)

def cephes_igamc(a: float, x: float) -> float:
	a_double = ctypes.c_double(a)
	x_double = ctypes.c_double(x)
	cepheslib.cephes_igamc.restype = ctypes.c_double
	r_value = cepheslib.cephes_igamc(a_double, x_double)
	return r_value


def main():
	a = 2**(2-1)
	x = 5.550792/2
	igamc_val = cephes_igamc(a, x)
	print(f"Value is {igamc_val:0.6f}")

if __name__ == "__main__":
	main()
