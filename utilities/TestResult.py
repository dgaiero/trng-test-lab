class TestResult:

	def __init__(self, p_value, msg=''):
		self.p_value = p_value
		self.msg = msg
		self.status = p_value >= 0.01

	def __str__(self):
		msg_print = ''
		if self.msg:
			msg_print = f"::{self.msg}"
		if self.p_value >= 0.01:
			return f"PASSED ({self.p_value:0.3f} > 0.01{msg_print}::sequence is random)"
		return f"FAILED ({self.p_value:0.3f} < 0.01{msg_print}::sequence is non-random)"
