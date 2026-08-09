#!/usr/bin/env python3.11
import sys
import os
import numpy as np

# Resolve nn_core package from src/ (works whether or not pip install -e . was run)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nn_core.logger import ObjLogger, title_message

def main():
    # Auto logger name from this file -> "Test1"
    logger = ObjLogger("NN_Lecture")
    # logger("Script started", color="green")
    # logger("Processing data...", logLevel="DEBUG", color="cyan")
    # logger("Warning occurred", logLevel="WARNING", color="yellow")
    # logger("An error occurred!", logLevel="ERROR", color="red")
    # logger("Script completed successfully", logLevel="INFO")

    # Custom logger name
    custom_logger = ObjLogger("Testing_Script")
    # custom_logger("Using a custom logger name", logLevel="debug", color="magenta")

    x = 1
    logger(f"{x}, {type(x)}", color="cyan")
    logger(sys.float_info, color="cyan")

    title_message(f"Trying NumPy")
    x = np.float64(5.23654789)
    logger(f"{x}, {type(x)}", color="blue")

    title_message(f"Trying Basket logic")
    basket = {'apple', 'orange', 'aa', 'bb',
              'apple', 'bb'}
    logger(basket, color="blue")
    basket = ['apple', 'orange', 'aa', 'bb',
              'apple', 'bb']
    logger(basket, color="cyan")
    basket = [{'apple', 'orange', 'aa', 'bb',
              'apple', 'bb'}]
    logger(basket, color="yellow")


if __name__ == "__main__":
    main()
