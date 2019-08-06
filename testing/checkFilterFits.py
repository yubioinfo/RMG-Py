#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
This script checks that the filter fits stored at RMG-database/input/FilterArrheniusFits are up to date with the current
database
"""
import math
import operator
import os.path

import nose.tools
import numpy as np

from rmgpy import settings
from rmgpy.data.kinetics.database import FilterLimitFits
from rmgpy.data.rmg import RMGDatabase
from rmgpy.kinetics.arrhenius import Arrhenius
from rmgpy.quantity import ScalarQuantity
from rmgpy.thermo.thermoengine import submit


################################################################################
class TestArrheniusFilterFits():
    """
    This class contains unit tests for the database for rigorous error checking.
    This class cannot inherit from unittest.TestCase if we want to use nose test generators.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the database before running the tests.
        """
        database_directory = settings['database.directory']
        cls.database = RMGDatabase()
        cls.database.load(database_directory,
                          thermoLibraries=['primaryThermoLibrary', 'Klippenstein_Glarborg2016', 'BurkeH2O2',
                                           'thermo_DFT_CCSDTF12_BAC', 'CBS_QB3_1dHR',
                                           'DFT_QCI_thermo', 'Narayanaswamy', 'Lai_Hexylbenzene',
                                           'SABIC_aromatics', 'vinylCPD_H'],
                          kineticsFamilies='all')

    def test_filter_fits(self):
        """
        This method tests if the database file containing the Arrhenius fits for reaction filtering is up-to-date.
        """
        # Load file with Arrhenius fits.
        database_directory = settings['database.directory']
        filename = 'FilterArrheniusFits.yml'
        path = os.path.join(database_directory, filename)
        path_notebook = os.path.join(os.path.dirname(database_directory), 'scripts', 'generateFilterArrheniusFits.ipynb')

        filter_fits = FilterLimitFits()

        class_dictionary = {'ScalarQuantity': ScalarQuantity,
                            'Arrhenius': Arrhenius,
                            'FilterLimitFits': FilterLimitFits,
        }

        FilterLimitFits.load_yaml(filter_fits, path, class_dictionary)

        file_families_unimol, file_arr_unimol = zip(*filter_fits.unimol.items())
        file_families_bimol, file_arr_bimol = zip(*filter_fits.bimol.items())

        # First check if the number of reaction families (without surface reactions) hasn't changed.
        if (len(self.database.kinetics.families.keys()) != len(file_families_unimol)) or (len(
                self.database.kinetics.families.keys()) != len(file_families_bimol)):
            raise ValueError("Arrhenius fits for reaction filtering need to be updated. "
                             "Please run the `generateFilterArrheniusFits` ipython notebook located in"
                             "`RMG-database/scripts/` and commit the newly generated file.")

        # Generate fits for each family.
        # Discrete temperatures for Arrhenius fit evaluation.
        Tmin = 298.0
        Tmax = 2500.0
        Tcount = 50
        temperatures = 1 / np.linspace(1 / Tmax, 1 / Tmin, Tcount)

        # Unimolecular reactions
        for ind, file_family_name in enumerate(file_families_unimol):
            # Has to be updated once training reactions with surface chemistry works.
            if 'Surface' not in file_family_name:
                # Fit new Arrhenius.
                new_arr_unimol = self.analyze_reactions(temperatures, file_family_name, molecularity=1)
                file_fit = file_arr_unimol[ind]

                # Compare new fit with old one. assert_almost_equal has difficulties to compare large floats e.g. for A
                if new_arr_unimol is not None:
                    if not self.is_close(new_arr_unimol.A.value_si, file_fit.A.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_unimol.Ea.value_si, file_fit.Ea.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_unimol.n.value_si, file_fit.n.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_unimol.T0.value_si, file_fit.T0.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_unimol.Tmin.value_si, file_fit.Tmin.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_unimol.Tmax.value_si, file_fit.Tmax.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))

        # Bimolecular reactions
        for ind, file_family_name in enumerate(file_families_bimol):
            # Has to be updated once training reactions with surface chemistry works.
            if 'Surface' not in file_family_name:
                # Fit new Arrhenius.
                new_arr_bimol = self.analyze_reactions(temperatures, file_family_name, molecularity=2)
                file_fit = file_arr_bimol[ind]

                # Compare new fit with old one.
                if new_arr_bimol is not None:
                    if not self.is_close(new_arr_bimol.A.value_si, file_fit.A.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_bimol.Ea.value_si, file_fit.Ea.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_bimol.n.value_si, file_fit.n.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_bimol.T0.value_si, file_fit.T0.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_bimol.Tmin.value_si, file_fit.Tmin.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))
                    if not self.is_close(new_arr_bimol.Tmax.value_si, file_fit.Tmax.value_si):
                        raise ValueError("""Arrhenius fits for reaction filtering need to be updated. Please run the
                        ipython notebook {0} and commit the newly generated file.""".format(path_notebook))

    def analyze_reactions(self, temperatures, fam_name, molecularity=1):
        """
        Refitting Arrhenius for reaction filtering using the current RMG database.
        """
        fam = self.database.kinetics.families[fam_name]
        dep = fam.getTrainingDepository()
        rxns = []

        # Extract all training reactions for selected family
        for entry in dep.entries.values():
            r = entry.item
            r.kinetics = entry.data
            r.index = entry.index
            for spc in r.reactants + r.products:
                if spc.thermo is None:
                    submit(spc)
            rxns.append(r)

        # Only proceed if at least one training reaction is available
        if rxns:
            # Get kinetic rates for unimolecular reactions
            k_list = []
            index_list = []
            for rxn in rxns:
                if len(rxn.reactants) == molecularity:
                    k_list.append(rxn.kinetics)
                    index_list.append(rxn.index)
                if len(rxn.products) == molecularity:
                    k_list.append(rxn.generateReverseRateCoefficient())
                    index_list.append(rxn.index)

            # Get max. kinetic rates at each discrete temperature
            if k_list:
                k_max_list = []
                max_rxn_list = set()
                for T in temperatures:
                    kvals = [k.getRateCoefficient(T) for k in k_list]
                    mydict = dict(zip(index_list, kvals))

                    # Find key and value of max rate coefficient
                    key_max_rate = max(mydict.iteritems(), key=operator.itemgetter(1))[0]

                    max_entry = dep.entries.get(key_max_rate)
                    max_rxn = max_entry.item
                    max_rxn_list.add(max_rxn)

                    kval = mydict[key_max_rate]
                    k_max_list.append(kval)

                units = 's^-1' if molecularity == 1 else 'm^3/(mol*s)'

                arr = Arrhenius().fitToData(temperatures, np.array(k_max_list), units)
                return arr

        else:
            arr = None
            return arr

    @staticmethod
    def is_close(a, b, rel_tol=1e-2, abs_tol=0.0):
        """
        Python 2 implementation of Python 3.5 math.isclose()
        https://hg.python.org/cpython/file/tip/Modules/mathmodule.c#l1993
        """
        # sanity check on the inputs
        if rel_tol < 0 or abs_tol < 0:
            raise ValueError("tolerances must be non-negative")

        # short circuit exact equality -- needed to catch two infinities of
        # the same sign. And perhaps speeds things up a bit sometimes.
        if a == b:
            return True

        # This catches the case of two infinities of opposite sign, or
        # one infinity and one finite number. Two infinities of opposite
        # sign would otherwise have an infinite relative tolerance.
        # Two infinities of the same sign are caught by the equality check
        # above.
        if math.isinf(a) or math.isinf(b):
            return False

        # now do the regular computation
        # this is essentially the "weak" test from the Boost library
        diff = math.fabs(b - a)
        result = (((diff <= math.fabs(rel_tol * b)) or
                   (diff <= math.fabs(rel_tol * a))) or
                  (diff <= abs_tol))
        return result


################################################################################
if __name__ == '__main__':
    nose.run(argv=[__file__, '-v', '--nologcapture'], defaultTest=__name__)
