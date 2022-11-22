# This module contains functions for validating the rest of the package and ensuring that they return
# correct outputs. The idea is to exhaustively save everything, and backtracking from the output,
# repeatedly run my own forward pass and make sure it matches the real output, and do this for
# multiple inputs per model.

# Validation steps: first, make sure that all possible key entries in the pretty final output
# corresponding to that tensor are in fact the same tensor data. Then plug that data in, and run
# the feedforward pass from there, and check that it matches the output.

# As a "debug mode", keep ALL functions applied and their arguments without discarding (this might
# require tweaking the logic of expand_multiple_functions).
