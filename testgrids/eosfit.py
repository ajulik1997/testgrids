# eosfit.py fits E(V) data to a Birch-Murnaghan equation of state. 
# Current version: 3.1
#
# Copyright (C) 2012 Kurt Lejaeghere <Kurt.Lejaeghere@UGent.be>, Center for
# Molecular Modeling (CMM), Ghent University, Ghent, Belgium
#
# eosfit.py is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# eosfit.py is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for 
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with eosfit.py; if not, see <http://www.gnu.org/licenses/>.

# The following code is based on the source code of eos.py from the Atomic 
# Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>.

# Python and numpy are required to use this script.

# Edit for use in script by Alexander Liptak <Alexander.Liptak.2015@live.rhul.ac.uk>

import numpy as np

def BM(volumes, energies):

    fitdata = np.polyfit(np.power(volumes, np.divide(-2, 3)), energies, 3, full=True)
    residuals0 = np.divide(fitdata[1], np.sum(np.power(np.subtract(energies, np.average(energies)), 2)))
    deriv0 = np.poly1d(fitdata[0])
    deriv1 = np.polyder(deriv0, 1)
    deriv2 = np.polyder(deriv1, 1)
    deriv3 = np.polyder(deriv2, 1)

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        if x > 0 and deriv2(x) > 0:
            volume0 = x**(-3./2.)
            break

    if volume0 == 0:
        print('Error: No minimum could be found')
        exit()
    
    derivV2 = 4./9. * x**5. * deriv2(x)
    derivV3 = (-20./9. * x**(13./2.) * deriv2(x) -
        8./27. * x**(15./2.) * deriv3(x))
    bulk_modulus0 = derivV2 / x**(3./2.)
    bulk_deriv0 = -1 - x**(-3./2.) * derivV3 / derivV2

    return volume0, bulk_modulus0, bulk_deriv0, residuals0

