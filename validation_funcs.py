# This module contains functions for validating the rest of the package and ensuring that they return
# correct outputs. The idea is to exhaustively save everything, and backtracking from the output,
# repeatedly run my own forward pass and make sure it matches the real output, and do this for
# multiple inputs per model.
