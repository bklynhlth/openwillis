# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def make_dir(dir_name):
    """
    ------------------------------------------------------------------------------------------------------

    Creates a directory if it doesn't already exist.

    Parameters:
    ...........
    dir_name : str
        The path to the directory

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name):
    """
    ------------------------------------------------------------------------------------------------------
    Deletes a directory if it exists.

    Parameters:
    ...........
    dir_name : str
        The path to the directory

    Returns:
    ...........
    None
    ------------------------------------------------------------------------------------------------------
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def clean_dir(dir_name):
    """
    ------------------------------------------------------------------------------------------------------

    Creates a directory if it doesn't exist, or deletes the directory if it does.

    Parameters:
    ...........
    dir_name : str
        The path to the directory

    Returns:
    ...........
    None

    ------------------------------------------------------------------------------------------------------
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
