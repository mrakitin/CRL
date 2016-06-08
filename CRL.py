# -*- coding: utf-8 -*-
# Author: Maksim Rakitin (BNL)
# 2016

from __future__ import division

import copy
import json
import math
import os

try:
    import numpy as np

    NUMPY = True
except:
    NUMPY = False
NUMPY = False  # explicitly avoid usage of NumPy.

DAT_DIR = 'dat'
CONFIG_DIR = 'configs'


class CRL:
    def __init__(self,
                 cart_ids,
                 beamline='smi',
                 dl_lens=2e-3,  # distance between two lenses within a cartridge [m]
                 dl_cart=30e-3,  # distance between centers of two neighbouring cartridges [m]
                 r_array=(50, 200, 500),  # radii of available lenses in different cartridges [um]
                 lens_array=(1, 2, 4, 8, 16),  # possible number of lenses in cartridges
                 p0=6.2,  # dist from z=50.9 m to first lens in most upstream cart at most upstream pos of transfocator
                 d_ssa_focus=8.1,  # [m]
                 energy=24000.0,  # [eV]
                 teta0=60e-6,  # [rad]
                 data_file=os.path.join(DAT_DIR, 'Be_delta.dat'),
                 use_numpy=NUMPY):

        # Input variables:
        self.cart_ids = cart_ids
        self.beamline = beamline
        self.dl_lens = dl_lens
        self.dl_cart = dl_cart
        self.r_array = r_array
        self.lens_array = lens_array
        self.p0 = p0
        self.d_ssa_focus = d_ssa_focus
        self.energy = energy
        self.teta0 = teta0
        self.data_file = data_file
        self.use_numpy = use_numpy

        # Prepare other necessary variables:
        self.read_config_file()  # defines self.config_file and self.transfocator_config
        self._get_lens_config()  # defines self.lens_config
        self._calc_delta()  # defines self.delta
        self._calc_y0()  # defines self.y0
        self.radii = None
        self.n = None
        self.T = None
        self.y = None
        self.teta = None
        self.ideal_focus = None
        self.p1 = None
        self.p1_ideal = None

        # Perform calculations:
        self.calc_T_total()
        self.calc_y_teta()

    def calc_delta_focus(self, p):
        if p is not None:
            d = self.d_ssa_focus - (self.p0 + p + self.transfocator_config[self._find_element_by_id(self.cart_ids[-1])][
                'offset_cart'] * self.dl_cart)
        else:
            d = None
        return d

    def calc_ideal_focus(self, radius, n):
        self.ideal_focus = radius / (2 * n * self.delta)

    def calc_ideal_lens(self):
        self._get_radii_n()
        tolerance = 1e-8
        if abs(sum(self.radii) / len(self.radii) - self.radii[0]) < tolerance:
            self.calc_ideal_focus(self.radii[0], self.n)
            self.p1_ideal = 1 / (1 / self.ideal_focus - 1 / self.p0)
        else:
            print('Radii of the specified lenses ({}) are different! Cannot calculate ideal lens.'.format(self.radii))
        return self.p1_ideal

    def calc_lens_array(self, radius, n):
        """Calculate accumulated T_fs for one cartridge with fixed radius.

        :param radius: radius.
        :param n: number of lenses in one cartridge.
        :return T_fs_accum: accumulated T_fs.
        """
        T_dl = self._calc_T_dl(self.dl_lens)
        T_fs = self._calc_T_fs(radius)

        T_fs_accum = self._dot(self._matrix_power(self._dot(T_fs, T_dl), n - 1), T_fs)
        return T_fs_accum

    def calc_real_lens(self):
        self.p1 = self.y / math.tan(math.pi - self.teta)
        return self.p1

    def calc_T_total(self):
        dist_list = []
        for i in range(len(self.cart_ids) - 1):
            dist_list.append(self._calc_distance(self.cart_ids[i], self.cart_ids[i + 1]))

        R_list = []
        N_list = []
        for i in range(len(self.cart_ids)):
            for j in range(len(self.transfocator_config)):
                if self.cart_ids[i] == self.transfocator_config[j]['id']:
                    name = self.transfocator_config[j]['name']
                    R_list.append(self.lens_config[name]['radius'])
                    N_list.append(self.lens_config[name]['lens_number'])

        if len(self.cart_ids) == 1:
            self.T = self.calc_lens_array(R_list[0], N_list[0])
        elif len(self.cart_ids) > 1:
            A = self._calc_T_dl(dist_list[0])
            B = self.calc_lens_array(R_list[0], N_list[0])
            self.T = self._dot(A, B)
            for i in range(len(self.cart_ids) + len(self.cart_ids) - 3):
                if i % 2 == 0:
                    B = self.calc_lens_array(R_list[int((i + 2) / 2)], N_list[int((i + 2) / 2)])
                    self.T = self._dot(B, self.T)
                else:
                    A = self._calc_T_dl(dist_list[int((i + 1) / 2)])
                    self.T = self._dot(A, self.T)
        else:
            raise Exception('No lenses in the beam!')

    def calc_y_teta(self):
        (self.y, self.teta) = self._dot(self.T, [self.y0, self.teta0])

    def get_inserted_lenses(self):
        self._get_radii_n()
        return {
            'ids': self.cart_ids,
            'radii': self.radii,
            'total_lenses': self.n
        }

    def read_config_file(self):
        self.config_file = os.path.join(CONFIG_DIR, '{}_crl.json'.format(self.beamline))
        if not os.path.isfile(self.config_file):
            raise Exception('Config file <{}> not found. Check the name of the file/beamline.'.format(self.config_file))
        with open(self.config_file, 'r') as f:
            self.transfocator_config = json.load(f)['crl']

    def _calc_delta(self):
        self.delta = None
        skiprows = 2
        energy_column = 0
        delta_column = 1
        error_msg = 'Error! Use energy range from {} to {} eV.'
        if self.use_numpy:
            data = np.loadtxt(self.data_file, skiprows=skiprows)
            try:
                idx_previous = np.where(data[:, energy_column] <= self.energy)[0][-1]
                idx_next = np.where(data[:, energy_column] > self.energy)[0][0]
            except IndexError:
                raise Exception(error_msg.format(data[0, energy_column], data[-1, energy_column]))

            idx = idx_previous if abs(data[idx_previous, energy_column] - self.energy) <= abs(
                data[idx_next, energy_column] - self.energy) else idx_next
            self.delta = data[idx][delta_column]
        else:
            with open(self.data_file, 'r') as f:
                content = f.readlines()
                energies = []
                deltas = []
                for i in range(skiprows, len(content)):
                    energies.append(float(content[i].split()[energy_column]))
                    deltas.append(float(content[i].split()[delta_column]))
                indices_previous = []
                indices_next = []
                try:
                    for i in range(len(energies)):
                        if energies[i] <= self.energy:
                            indices_previous.append(i)
                        else:
                            indices_next.append(i)
                    idx_previous = indices_previous[-1]
                    idx_next = indices_next[0]
                except IndexError:
                    raise Exception(error_msg.format(energies[0], energies[-1]))

                idx = idx_previous if abs(energies[idx_previous] - self.energy) <= abs(
                    energies[idx_next] - self.energy) else idx_next
                self.delta = deltas[idx]

    def _calc_distance(self, id1, id2):
        """Calculate distance between two arbitrary cartridges specified by ids.

        :param id1: id of cartridge 1.
        :param id2: id of cartridge 2.
        :return dist: calculated distance.
        """

        el_num1 = self._find_element_by_id(id1)
        el_num2 = self._find_element_by_id(id2)
        if el_num1 is None or el_num2 is None:
            raise Exception('Provided id\'s are not valid!')

        lens_num1 = self.lens_config[self.transfocator_config[el_num1]['name']]['lens_number']
        coord1 = self.transfocator_config[el_num1]['offset_cart'] * self.dl_cart
        coord2 = self.transfocator_config[el_num2]['offset_cart'] * self.dl_cart
        dist = coord2 - coord1 - lens_num1 * self.dl_lens
        return dist

    def _calc_T_dl(self, dl):
        T_dl = [
            [1, dl],
            [0, 1],
        ]
        return T_dl

    def _calc_T_fs(self, radius):
        T_fs = [
            [1, 0],
            [-1 / (radius / (2 * self.delta)), 1],
        ]
        return T_fs

    def _calc_y0(self):
        self.y0 = self.p0 * math.tan(self.teta0)

    def _dot(self, A, B):
        """Multiplies matrix A by matrix B."""
        if self.use_numpy:
            C = np.dot(A, B)
        else:
            B0 = B[0]
            lenB = len(B)
            lenA = len(A)
            if len(A[0]) != lenB:  # Check matrix dimensions
                raise Exception('Matrices have wrong dimensions')
            if isinstance(B0, list) or isinstance(B0, tuple):  # B is matrix
                lenB0 = len(B0)
                C = [[0 for _ in range(lenB0)] for _ in range(lenA)]
                for i in range(lenA):
                    for j in range(lenB0):
                        for k in range(lenB):
                            C[i][j] += A[i][k] * B[k][j]
            else:  # B is vector
                C = [0 for _ in range(lenB)]
                for i in range(lenA):
                    for k in range(lenB):
                        C[i] += A[i][k] * B[k]
        return C

    def _find_element_by_id(self, id):
        element_number = None
        for i in range(len(self.transfocator_config)):
            if id == self.transfocator_config[i]['id']:
                element_number = i
                break
        return element_number

    def _find_lens_parameters_by_id(self, id):
        return self._find_lens_parameters_by_name(self._find_name_by_id(id))

    def _find_lens_parameters_by_name(self, name):
        return self.lens_config[name]

    def _find_name_by_id(self, id):
        real_id = self._find_element_by_id(id)
        name = self.transfocator_config[real_id]['name']
        return name

    def _get_lens_config(self):
        self.lens_config = {}
        for i in self.r_array:
            for j in self.lens_array:
                self.lens_config['T_{}_{}'.format(j, i)] = {
                    'radius': i * 1e-6,
                    'lens_number': j,
                }

    def _get_radii_n(self):
        self.radii = []
        self.n = 0
        for i in self.cart_ids:
            name = self._find_name_by_id(i)
            lens = self._find_lens_parameters_by_name(name)
            self.radii.append(lens['radius'])
            self.n += lens['lens_number']

    def _matrix_power(self, A, n):
        """Multiply matrix A n times.

        :param A: input square matrix.
        :param n: power.
        :return B: resulted matrix.
        """
        if len(A) != len(A[0]):
            raise Exception('Matrix is not square: {} x {}'.format(len(A), len(A[0])))

        if self.use_numpy:
            B = np.linalg.matrix_power(A, n)
        else:
            if n > 0:
                B = copy.deepcopy(A)
                for i in range(n - 1):
                    B = self._dot(A, B)
            elif n == 0:
                B = []
                for i in range(len(A)):
                    row = []
                    for j in range(len(A[0])):
                        if i == j:
                            row.append(1)
                        else:
                            row.append(0)
                    B.append(row)
            else:
                raise Exception('Negative power <{}> is not supported for matrix power operation.'.format(n))

        return B


if __name__ == '__main__':
    l = [2, 4, 6, 7, 8]
    e = 21500
    p0 = 6.52

    crl1 = CRL(cart_ids=l, energy=e, p0=p0, use_numpy=True)
    p1 = crl1.calc_real_lens()
    p1_ideal = crl1.calc_ideal_lens()

    crl2 = CRL(cart_ids=l, energy=e, p0=p0, use_numpy=False)
    p2 = crl2.calc_real_lens()
    p2_ideal = crl2.calc_ideal_lens()

    d = crl1.calc_delta_focus(p1)
    d_ideal = crl1.calc_delta_focus(p1_ideal)

    print('P0: {}, P1: {}, P1 ideal: {}'.format(crl1.p0, p1, p1_ideal))
