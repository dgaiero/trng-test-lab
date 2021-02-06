import inspect
import json
import math
import os
import traceback
from typing import List
import itertools

from timeit import default_timer as timer
from scipy.special import gamma, gammainc
# from utilities.cephes.cepheswrapper import cephes_igamc as igamc
from utilities.TestResult import TestResult
from utilities.bma import berlekamp_Massey_algorithm
from utilities.Note import Note
from collections import OrderedDict

# https://stackoverflow.com/questions/38713199/incomplete-gamma-function-in-scipy


def igamc(a, x): return gamma(a)*(1 - gammainc(a, x))
def partition(n, N): return [(n[i:i+N]) for i in range(0, len(n), N)]


def notes_print(notes: List[Note]):
    printed = [n for n in notes if n.condition is True]
    if len(printed) != 0:
        print("** NOTE **")
        for note in printed:
            print(f"\t- {note.note}")


class TestSet:
    """Class to hold all test functions."""

    def __init__(self, n: int, bit_string: str, config_path: str = None):
        self.config_path = config_path
        self._test_data = None
        self.load_config_data()

        self.n = n
        self.bit_string = bit_string
        self._bit_string_parsed = self.validate_bitstring(
            self.n, self.bit_string)
        self._zeros = [n for n in self._bit_string_parsed if n == 0]
        self._ones = [n for n in self._bit_string_parsed if n == 1]
        self._num_zeros = len(self._zeros)
        self._num_ones = len(self._ones)
        notes = [Note('It is recommended to use a minimum of 100 bits.', self.n < 100),
                 Note('Tests are not guaranteed to run in any specific order.', 1 == 1), Note('test', 0 == 1)]
        notes_print(notes)

    def load_config_data(self):
        if self.config_path:
            if os.path.isfile(self.config_path) == True:
                with open(self.config_path) as jsonfile:
                    self._test_data = json.load(jsonfile)

    def validate_bitstring(_, length: int, bitstring: str):
        bit_string_parsed = [None]*length
        for idx, char in enumerate(bitstring):
            try:
                char_int = int(char)
            except ValueError:
                raise ValueError("Unexpected character in input."
                                 f"(expected [0, 1], actual {char} at position {idx})")
            bit_string_parsed[idx] = char_int
        return bit_string_parsed

    def build_testset(self):
        all_methods = dir(self)
        self._testset = \
            [test for test in all_methods if test.startswith("test")]

    def run_testset(self):
        self._n_pass = 0
        self._pass_messages = []
        self._n_fail = 0
        self._fail_messages = []
        self._total_test_exec = 0
        start = 0
        end = 0
        for test in self._testset:
            test_data = None
            if self._test_data:
                if test in self._test_data:
                    test_data = self._test_data[test]
            handle = getattr(self, test)
            print(f"\nRunning {test}()")
            print("-" * (10+len(test)))
            docstring = inspect.getdoc(handle)
            if (docstring):
                print(docstring)
            else:
                print("No description provided.")
            try:
                start = timer()
                results = handle(test_data)
                end = timer()
            except BaseException as err:
                self._fail_messages.append(f"::{test}() FAILED ({err})::")
                print(self._fail_messages[-1])
                traceback.print_exc()
                self._n_fail += 1
            else:
                for result in results:
                    print_msg = f"::{test}() {result}::"
                    if result.status is True:
                        self._pass_messages.append(print_msg)
                        self._n_pass += 1
                    else:
                        self._fail_messages.append(print_msg)
                        self._n_fail += 1
                    print(print_msg)
            finally:
                test_time = end - start
                self._total_test_exec += test_time
                test_complete_msg = f"Completed {test} in {test_time:0.3f} s"
                print(test_complete_msg)
                print("-" * (10+len(test)))

    def print_summary(self):
        print("\nTest Summary:")
        print("-------------")
        print(f"Total Pass: {self._n_pass}")
        for msg in self._pass_messages:
            print(f"\t{msg}")
        print(f"Total Fail: {self._n_fail}")
        for msg in self._fail_messages:
            print(f"\t{msg}")
        print(f"Total Time: {self._total_test_exec:0.2f} s")

    def test_001_frequency(self, _):
        """This test focuses on verifying the proportion of ones and zeros in a
        given sequence. Verification is done to see how close this proportion
        equals 0.5 (i.e. the number of ones and zeros should be the same)
        """
        s_n = self._num_ones + (self._num_zeros*-1)
        s_obs = abs(s_n)/math.sqrt(self.n)
        p_value = math.erfc(s_obs/math.sqrt(2))
        return [TestResult(p_value)]

    def test_002_runs(self, _):
        """This test looks at the total number of uninterrupted bits (a run).
        This test determines if the oscillations between runs of zeros and ones
        is too fast or slow.
        """
        tau = 2/math.sqrt(self.n)
        pi = self._num_ones/self.n
        prereq_freq = abs(pi-0.5) >= tau
        if prereq_freq:
            raise AssertionError(
                f"abs(pi-0.5) >= tau::{abs(pi-0.5)}>={tau}::monobit test failure")
        v_n_obs = 1
        for i in range(0, self.n - 1):
            if self._bit_string_parsed[i] != self._bit_string_parsed[i + 1]:
                v_n_obs += 1
        p_value = math.erfc((abs(v_n_obs-2*self.n*pi*(1-pi))) /
                            (2*math.sqrt(2*self.n)*pi*(1-pi)))
        return [TestResult(p_value)]

    def test_003_non_overlapping_template_matching_test(self, data):
        """This test looks to see if there are too many occurrences of a given
        non-periodic pattern. An m-bit window is used to search for a specific
        bit pattern. If the pattern is not found, the window slides by one bit
        position. If it is found, the window slides to one bit after the found
        pattern.
        """
        B = data["B"]
        m = len(B)
        M = data["M"]
        N = data["N"]
        N_expected = math.floor(self.n/M)
        notes = [
            Note(f"Unexpected N Value (expected {N_expected} actual {N})", N_expected != N)]
        # https://www.geeksforgeeks.org/python-split-string-in-groups-of-n-consecutive-characters/
        blocks = partition(self.bit_string, M)
        W = [0]*N
        for idx, block in enumerate(blocks):
            if len(block) != M:
                notes.append(Note(f"Discarded byte string. Length does"
                                  f"not match M (expected {M} actual {len(block)})",
                                  len(block) != M))
                continue
            start_idx = 0
            end_idx = m
            while end_idx < M:
                substring = block[start_idx:end_idx]
                if substring == B:
                    start_idx += m
                    end_idx += m
                    W[idx] += 1
                else:
                    start_idx += 1
                    end_idx += 1
        mu = (M-m+1)/(2**m)
        sig_2 = M*(1/(2**m)-(2*m-1)/(2**(2*m)))
        x_2_obs = sum([((x-mu)**2)/sig_2 for x in W])
        p_value = igamc(N/2, x_2_obs/2)
        notes_print(notes)
        return [TestResult(p_value)]

    def test_004_overlapping_template_matching_test(self, data):
        """This test focuses on the number of occurrences of pre-specified
        strings. The difference between this test and the non-overlaping test is
        that when a pattern is found, the window slides only one bit before
        resuming the search.
        """
        B = data["B"]
        K = data["K"]
        notes = [Note(f"unsupported K value (expected 5 actual {K})", K != 5)]
        notes_print(notes)
        if K != 5:
            K = 5
        m = len(B)
        M = data["M"]
        N = data["N"]
        pi = data["pi"]

        # https://www.geeksforgeeks.org/python-split-string-in-groups-of-n-consecutive-characters/
        blocks = partition(self.bit_string, M)
        v = [0]*(K+1)
        for block in blocks:
            start_idx = 0
            end_idx = m
            n_hit = 0
            while end_idx < M:
                substring = block[start_idx:end_idx]
                if substring == B:
                    n_hit += 1
                start_idx += 1
                end_idx += 1
            if n_hit < (K+1):
                v[n_hit] += 1
        x_2_obs = 0
        for i in range(0, K):
            numerator = (v[i] - N*pi[i])**2
            denominator = N*pi[i]
            x_2_obs += numerator/denominator

        p_value = igamc(K/2, x_2_obs/2)
        return [TestResult(p_value)]

    def test_005_linear_complexity(self, data):
        """This test looks at the length of a linear feedback shift register
        and determines whether the sequence is considered complex enough to be
        random.
        """
        M = data["M"]
        K = data["K"]
        pi = data["pi"]
        notes = [Note(f"unsupported K value (expected 6 actual {K})", K != 6)]
        notes_print(notes)
        if K != 6:
            K = 6
        N = math.floor(self.n/M)
        blocks = partition(self.bit_string[0:(M*N)-1], M)
        blocks_parsed = [self.validate_bitstring(len(block), block) for block in blocks]
        l = [None]*N
        for idx, block in enumerate(blocks_parsed):
            _, l[idx] = berlekamp_Massey_algorithm(block)
        mu = (M/2)+(9+(-1)**(M+1))/36-(M/3+2/9)/(2**M)
        v = [0]*(K+1)
        for l_i in l:
            t_i = (-1)**M*(l_i-mu)+2/9
            if t_i < -2.5:
                v[0] += 1
            elif t_i <=-1.5:
                v[1] += 1
            elif t_i <= -0.5:
                v[2] += 1
            elif t_i <= 0.5:
                v[3] += 1
            elif t_i <=1.5:
                v[4] += 1
            elif t_i <= 2.5:
                v[5] += 1
            else:
                v[6] += 1
        x_2_obs = sum([((v[i] - N*pi[i])**2)/(N*pi[i]) for i in range(K+1)])

        p_value = igamc(K/2, x_2_obs/2)
        return [TestResult(p_value)]

    def test_006_approximate_entropy(self, data):
        """This test checks all possible overlapping m-bit patterns in the
        sequence. The purpose of this test is to compare the frequency of
        overlapping patterns in two adjacent lengths against what is expected
        of a random sequence.
        """
        m = data["m"]
        m_arr = [m, m+1]
        phi = []
        def log_special(x): return x if x == 0 else math.log(x)
        for _m in m_arr:
            bit_string_aug = self.bit_string[0:_m-1]
            bit_string_aug = ''.join([self.bit_string, bit_string_aug])
            C_i_val = list(itertools.product([0, 1], repeat=_m))
            for idx, val in enumerate(C_i_val):
                val = [str(n) for n in val]
                C_i_val[idx] = ''.join(val)
            C_i = {key: 0 for key in C_i_val}
            for i in range(len(bit_string_aug)-_m+1):
                block = bit_string_aug[i:i+_m]
                C_i[block] += 1
            C_i = {key: i/self.n for key, i in C_i.items()}
            phi.append(sum([pi*log_special(pi) for pi in C_i.values()]))
        x_2 = 2*self.n*(math.log(2)-(phi[0]-phi[1]))
        p_value = igamc(2**(m-1), x_2/2)
        return [TestResult(p_value)]

    def test_007_random_excursions_variant(self, _):
        """This test looks at the number of times a particular state occurs in a
        random walk. The purpose of this test is to check for deviations from
        the expected number of occurrences.
        """
        norm_seq = [(n*2)-1 for n in self._bit_string_parsed]
        s = list(itertools.accumulate(norm_seq))
        s_prime = [0] + s + [0]
        x = list(range(-18,19))
        xi = {key: 0 for key in x}
        del xi[0]
        xi = {key: s_prime.count(key) for key in xi.keys()}
        p_values = []
        J = sum(x > 0 for x in xi.values())
        for key, val in xi.items():
            result = TestResult(math.erfc((abs(val-J))/(math.sqrt(2*J*(4*abs(key)-2)))))
            result.msg = f"State = {key}"
            p_values.append(result)
        return p_values


def open_file(filename, mode):
    with open(filename, mode) as file:
        contents = file.read()
    return contents


def open_file2(filename, mode):
    with open(filename, mode) as file:
        contents = ''.join(line.strip() for line in file)
    return contents

def main():
    # from secrets import randbelow
    # bit_string = [str(randbelow(2)) for _ in range(2**20)]
    # bit_string = ''.join(bit_string)
    # bit_string = ''.join([bin(i)[2:].zfill(8) for i in bit_string])

    # cwd = os.path.dirname(os.path.realpath(__file__))
    # file_location = os.path.join(cwd, "data", "NIST", "data.e")
    # bit_string = open_file2(file_location, "r")
    file_list = ['output_lcg.txt', 'output_sq.txt', 'output_trng.txt']
    cwd = os.path.dirname(os.path.realpath(__file__))
    for file in file_list:
        file_location = os.path.join(cwd, "data", "ours", file)
        bit_string = open_file2(file_location, "r")
        print("-" * 80)
        print(f"Testing file {file}")
        print("-" * 80)
        TestSetRunner = TestSet(len(bit_string), bit_string, config_path="testdata.json")
        TestSetRunner.build_testset()
        TestSetRunner.run_testset()
        TestSetRunner.print_summary()
        print("-" * 80)
        print("-" * 80)


if __name__ == "__main__":
    main()
