# -*- coding: utf-8 -*-

from __future__ import division

import copy
import math

try:
    import numpy as np

    NUMPY = True
except:
    NUMPY = False
NUMPY = False  # explicitly avoid usage of NumPy.

TRANSFOCATOR_CONFIG = [
    {
        'id': 1,
        'name': 'T_1_500',
        'offset_cart': 2,
    },
    {
        'id': 2,
        'name': 'T_2_50',
        'offset_cart': 3,
    },
    {
        'id': 3,
        'name': 'T_8_500',
        'offset_cart': 4,
    },
    {
        'id': 4,
        'name': 'T_4_50',
        'offset_cart': 5,
    },
    {
        'id': 5,
        'name': 'T_1_200',
        'offset_cart': 6,
    },
    {
        'id': 6,
        'name': 'T_16_50',
        'offset_cart': 7,
    },
    {
        'id': 7,
        'name': 'T_8_50',
        'offset_cart': 9,
    },
    {
        'id': 8,
        'name': 'T_1_50',
        'offset_cart': 11,
    },
]


class CRL:
    def __init__(self,
                 cart_ids,
                 dl_lens=2e-3,
                 dl_cart=30e-3,
                 r_array=(50, 200, 500),
                 lens_array=(1, 2, 4, 8, 16),
                 p0=6.2,
                 energy=24000.0,
                 teta0=60e-6,
                 transfocator_config=TRANSFOCATOR_CONFIG,
                 data_file='Be_delta.dat',
                 use_numpy=NUMPY):

        # Input variables:
        self.cart_ids = cart_ids
        self.dl_lens = dl_lens
        self.dl_cart = dl_cart
        self.r_array = r_array
        self.lens_array = lens_array
        self.p0 = p0
        self.energy = energy
        self.teta0 = teta0
        self.transfocator_config = transfocator_config
        self.data_file = data_file
        self.use_numpy = use_numpy

        # Prepare other necessary variables:
        self._process_lens_config()  # defines self.lens_config
        self._calc_delta()  # defines self.delta
        self._calc_y0()  # defines self.y0
        self.T = None
        self.y = None
        self.teta = None
        self.ideal_focus = None
        self.p1 = None
        self.p1_ideal = None

        # Perform calculations:
        self.calc_T_total()
        self.calc_y_teta()

    def calc_ideal_focus(self, radius, n):
        self.ideal_focus = radius / (2 * n * self.delta)

    def calc_ideal_lens(self):
        n = 0
        radii = []
        for i in self.cart_ids:
            name = self._find_name_by_id(i)
            lens = self._find_lens_parameters_by_name(name)
            n += lens['lens_number']
            radii.append(lens['radius'])
        tolerance = 1e-8
        if abs(sum(radii) / len(radii) - radii[0]) < tolerance:
            self.calc_ideal_focus(radii[0], n)
            self.calc_p1_ideal()
        else:
            print('Radii of the specified lenses ({}) are different! Cannot calculate the ideal lens.'.format(radii))
        return self.p1_ideal

    def calc_real_lens(self):
        self.calc_p1()
        return self.p1

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

    def calc_p1(self):
        self.p1 = self.y / math.tan(math.pi - self.teta)

    def calc_p1_ideal(self):
        self.p1_ideal = (1 / (1 / self.ideal_focus - 1 / self.p0))

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

    def _find_name_by_id(self, id):
        real_id = self._find_element_by_id(id)
        name = self.transfocator_config[real_id]['name']
        return name

    def _find_lens_parameters_by_id(self, id):
        return self._find_lens_parameters_by_name(self._find_name_by_id(id))

    def _find_lens_parameters_by_name(self, name):
        return self.lens_config[name]

    def _matrix_power(self, A, n):
        """Multiply matrix A n times.

        :param A: input square matrix.
        :param n: power.
        :return B: resulted matrix.
        """
        if self.use_numpy:
            B = np.linalg.matrix_power(A, n)
        else:
            B = copy.deepcopy(A)
            for i in range(n - 1):
                B = self._dot(A, B)
        return B

    def _process_lens_config(self):
        self.lens_config = {}
        for i in self.r_array:
            for j in self.lens_array:
                self.lens_config['T_{}_{}'.format(j, i)] = {
                    'radius': i * 1e-6,
                    'lens_number': j,
                }


if __name__ == '__main__':
    crl = CRL([2, 4, 6, 7, 8])
    p1 = crl.calc_real_lens()
    p1_ideal = crl.calc_ideal_lens()

    print()
