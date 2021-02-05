import inspect
import math
import os
import json
from scipy.special import gammainc
from scipy.special import gamma

# https://stackoverflow.com/questions/38713199/incomplete-gamma-function-in-scipy
igamc = lambda a, x: gamma(a)*(1 - gammainc(a, x))

class Test:
    """Class to hold all test functions."""

    def __init__(self, n: int, bit_string: str, config_path: str=None):
        self.config_path = config_path
        self._test_data = None
        self.load_config_data()

        self.n = n
        self.bit_string = bit_string
        self._bit_string_parsed = self.validate_bitstring(self.n, self.bit_string)
        self._zeros = [n for n in self._bit_string_parsed if n == 0]
        self._ones = [n for n in self._bit_string_parsed if n == 1]
        self._num_zeros = len(self._zeros)
        self._num_ones = len(self._ones)
        notes = [['It is recommended to use a minimum of 100 bits.', self.n < 100],
            ['Tests are not guaranteed to run in any specific order.', 1==1]]
        print ("** NOTE **")
        for note in notes:
            if note[1]:
                print(f"\t- {note[0]}")
        print(f"\t- End of notes.")

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
                raise ValueError("Unexpected character in input." \
                   f"(expected [0, 1], actual {char} at position {idx})")
            bit_string_parsed[idx] = char_int
        return bit_string_parsed

    def build_testset(self):
        all_methods = dir(self)
        self._testset = \
            [test for test in all_methods if test.startswith("test")]

    def run_testset(self):
        n_pass = 0
        pass_messages = []
        n_fail = 0
        fail_messages = []
        for test in self._testset:
            test_data = None
            if self._test_data:
                if test in self._test_data:
                    test_data = self._test_data[test]
            handle = getattr(self, test)
            print(f"\nRunning {test}()")
            print("-" * (10+len(test)))
            print(inspect.getdoc(handle))
            try:
                success_msg = handle(test_data)
            except AssertionError as err:
                fail_messages.append(f"::{test}() FAILED ({err})::")
                print(fail_messages[-1])
                n_fail += 1
            else:
                pass_messages.append(f"::{test}() PASSED ({success_msg})::")
                print(pass_messages[-1])
                n_pass += 1

    def test_frequency(self, _):
        """This test focuses on verifying the proportion of ones and zeros in a
        given sequence. Verification is done to see how close this proportion
        equals 0.5 (i.e. the number of ones and zeros should be the same)
        """
        s_n = self._num_ones + (self._num_zeros*-1)
        s_obs = abs(s_n)/math.sqrt(self.n)
        p_value = math.erfc(s_obs/math.sqrt(2))
        if p_value < 0.01:
            raise AssertionError(f"{p_value:0.2f} < 0.01::sequence is non-random")
        return f"{p_value:0.2f} > 0.01::sequence is random"

    def test_runs(self, _):
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
        if p_value < 0.01:
            raise AssertionError(
                f"{p_value:0.2f} < 0.01::sequence is non-random")
        return f"{p_value:0.2f} > 0.01::sequence is random"

    def test_non_overlapping_template_matching_test(self, data):
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
        # https://www.geeksforgeeks.org/python-split-string-in-groups-of-n-consecutive-characters/
        blocks = [(self.bit_string[i:i+M]) for i in range(0, len(self.bit_string), M)]
        W = [0]*N
        for idx, block in enumerate(blocks):
            start_idx = 0
            end_idx = m
            while end_idx < M:
                substring = block[start_idx:end_idx]
                if substring == B:
                    start_idx += m
                    end_idx += 3
                    W[idx] += 1
                else:
                    start_idx += 1
                    end_idx += 1
        mu = (M-m+1)/(2**m)
        sig_2 = M*(1/(2**m)-(2*m-1)/(2**(2*m)))
        x_2_obs = sum([((x-mu)**2)/sig_2 for x in W])
        p_value = igamc(N/2, x_2_obs/2)
        if p_value < 0.01:
            raise AssertionError(
                f"{p_value:0.2f} < 0.01::sequence is non-random")
        return f"{p_value:0.2f} > 0.01::sequence is random"

    # def test_overlapping_template_matching_test(self, data):
    #     """This test focuses on the number of occurrences of pre-specified
    #     strings. The difference between this test and the non-overlaping test is
    #     that when a pattern is found, the window slides only one bit before
    #     resuming the search.
    #     """
    #     B = data["B"]
    #     K = data["K"]
    #     m = len(B)
    #     M = data["M"]
    #     N = data["N"]
    #     # https://www.geeksforgeeks.org/python-split-string-in-groups-of-n-consecutive-characters/
    #     blocks = [(self.bit_string[i:i+M]) for i in range(0, len(self.bit_string), M)]
    #     W = [0]*N
    #     for idx, block in enumerate(blocks):
    #         start_idx = 0
    #         end_idx = m
    #         while end_idx < M:
    #             substring = block[start_idx:end_idx]
    #             if substring == B:
    #                 W[idx] += 1
    #             start_idx += 1
    #             end_idx += 1
    #     lambda_val = (M-m+1)/(2**m)
    #     eta = lambda_val/2
    #     lambda_2_obs = sum([()/(N) for x in W])


    #     sig_2 = M*(1/(2**m)-(2*m-1)/(2**(2*m)))
    #     x_2_obs = sum([((x-mu)**2)/sig_2 for x in W])
    #     p_value = igamc(N/2, x_2_obs/2)
    #     if p_value < 0.01:
    #         raise AssertionError(
    #             f"{p_value:0.2f} < 0.01::sequence is non-random")
    #     return f"{p_value:0.2f} > 0.01::sequence is random"

def main():
    bit_string = '10111011110010110100011100101110111110000101101001'
    TestSet = Test(len(bit_string), bit_string, config_path="testdata.json")
    TestSet.build_testset()
    TestSet.run_testset()


if __name__ == "__main__":
    main()
