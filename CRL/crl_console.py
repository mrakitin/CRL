from CRL import CRL, DEFAULTS_FILE
from console_utils import console

if __name__ == '__main__':
    """
    Example of execution:

    $ python crl_console.py --cart_ids 2 4 6 7 8 --energy=21500 --p0=6.52

    "d","d_ideal","f","p0","p1","p1_ideal"
    0.000372455276869,-0.0669574652539,1.04864377922,6.52,1.24962754472,1.31695746525
    """
    console(CRL, DEFAULTS_FILE)
