#!/usr/bin/env python3.11
import sys, os, argparse
import numpy as np
# setting ENV as $pwd so current dir
ENV = os.getcwd()
sys.path.append(ENV)
sys.path.append(ENV + '/modules/')
sys.path.append(ENV + '/neural_networks/')
sys.path.append(ENV + '/computer_vision/')

from GeneralUtils import title_message
from modules.logger import ObjLogger

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
