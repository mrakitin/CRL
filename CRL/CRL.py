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
except:
    pass

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# Fix for Jython:
try:
    SCRIPT_PATH = SCRIPT_PATH.replace(os.path.join(format(os.environ['HOME']), '.jython-cache/cachedir/classes'), '')
except:
    pass

DAT_DIR = os.path.join(SCRIPT_PATH, 'dat')
CONFIG_DIR = os.path.join(SCRIPT_PATH, 'configs')
DEFAULTS_FILE = os.path.join(CONFIG_DIR, 'defaults.json')


class CRL:
    def __init__(self, **kwargs):
        # Get input variables:
        self.defaults = _convert_types(_read_json(DEFAULTS_FILE)['defaults'])
        for key, default_val in self.defaults.items():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
            elif not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, default_val['default'])

        # Initialize non-input variables:
        self.radii = None
        self.n = None
        self.T = None
        self.y = None
        self.teta = None
        self.ideal_focus = None
        self.p1 = 0
        self.p1_ideal = 0
        self.d = 0
        self.d_ideal = 0
        self.f = 0

        # Read CRL config and check inputs:
        self.read_config_file()  # defines self.config_file and self.transfocator_config
        self._get_available_ids()  # defines self.available_ids
        if not self._check_ids():
            return

        # Initialize non-input variables using methods:
        self.data_file = os.path.join(DAT_DIR, self.data_file)
        self._get_lens_config()  # defines self.lens_config
        self._calc_delta()  # defines self.delta
        self._calc_y0()  # defines self.y0

        # Perform calculations:
        self.calc_T_total()
        self.calc_y_teta()
        self.calc_real_lens()
        self.calc_ideal_lens()
        self.d = self.calc_delta_focus(self.p1)
        self.d_ideal = self.calc_delta_focus(self.p1_ideal)

    def calc_delta_focus(self, p):
        if p is not None:
            d = self.d_ssa_focus - (self.p0 + p + self.transfocator_config[self._find_element_by_id(self.cart_ids[-1])][
                'offset_cart'] * self.dl_cart)
        else:
            d = None
        return d

    def calc_ideal_lens(self):
        self._get_radii_n()
        tolerance = 1e-8
        if abs(sum(self.radii) / len(self.radii) - self.radii[0]) < tolerance:
            self.ideal_focus = self.radii[0] / (2 * self.n * self.delta)
            self.p1_ideal = 1 / (1 / self.ideal_focus - 1 / self.p0)
        else:
            print('Radii of the specified lenses ({}) are different! Cannot calculate ideal lens.'.format(self.radii))

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
        self.f = 1 / (1 / self.p0 + 1 / self.p1)

    def calc_T_total(self):
        dist_list = []
        for i in range(len(self.cart_ids) - 1):
            dist_list.append(self._calc_distance(self.cart_ids[i], self.cart_ids[i + 1]))

        R_list = []
        N_list = []
        for i in range(len(self.cart_ids)):
            j = self._find_element_by_id(self.cart_ids[i])
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

    def print_result(self, output_format=None):
        python_data = {
            'p0': self.p0,
            'p1': self.p1,
            'p1_ideal': self.p1_ideal,
            'd': self.d,
            'd_ideal': self.d_ideal,
            'f': self.f,
        }
        if not output_format:
            output_format = self.output_format

        if output_format == 'csv':
            header = []
            data = []
            for key in sorted(python_data.keys()):
                header.append(key)
                data.append(python_data[key])
            output_text = '{}\n{}\n'.format(
                ','.join(['"{}"'.format(x) for x in header]),
                ','.join([str(x) for x in data]))
        elif output_format == 'json':
            output_text = json.dumps(
                python_data,
                sort_keys=True,
                indent=4,
                separators=(',', ': '),
            )
        else:  # plain text
            output_list = []
            for key in sorted(python_data.keys()):
                output_list.append('{}: {}'.format(key, python_data[key]))
            output_text = ', '.join(output_list)

        print(output_text)
        if self.outfile:
            with open(self.outfile, 'w') as f:
                f.write(output_text)

    def read_config_file(self):
        self.config_file = os.path.join(CONFIG_DIR, '{}_crl.json'.format(self.beamline))
        self.transfocator_config = _read_json(self.config_file)['crl']

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

    def _check_ids(self):
        """Check for incorrect input."""
        if not self.cart_ids:
            return False

        for input_id in self.cart_ids:
            if input_id not in self.available_ids:
                msg = 'Specified cart_id <{}> not in the list of available ids: <{}>.'
                raise Exception(msg.format(input_id, ', '.join(self.available_ids)))

        len_total = len(self.cart_ids)
        len_unique = len(set(self.cart_ids))
        if len_total != len_unique:
            msg = 'Number of non-unique cartridge ids: {}'
            raise Exception(msg.format(len_total - len_unique + 1))

        return True

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

    def _get_available_ids(self):
        self.available_ids = []
        for i in range(len(self.transfocator_config)):
            self.available_ids.append(self.defaults['cart_ids']['element_type'](self.transfocator_config[i]['id']))

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


def crl_console():
    import argparse

    data = _read_json(DEFAULTS_FILE)
    description = data['description']
    defaults = _convert_types(data['defaults'])

    # Processing arguments:
    required_args = []
    optional_args = []

    for key in sorted(defaults.keys()):
        if defaults[key]['default'] is None:
            required_args.append(key)
        else:
            optional_args.append(key)

    parser = argparse.ArgumentParser(description=description)

    for key in required_args + optional_args:
        args = []
        if 'short_argument' in defaults[key]:
            args.append('-{}'.format(defaults[key]['short_argument']))
        args.append('--{}'.format(key))

        kwargs = {
            'dest': key,
            'default': defaults[key]['default'],
            'required': False,
            'type': defaults[key]['type'],
            'help': '{}.'.format(defaults[key]['help']),
        }
        if defaults[key]['default'] is None:
            kwargs['required'] = True

        if defaults[key]['type'] == bool:
            kwargs['action'] = 'store_true'
            del (kwargs['type'])

        if defaults[key]['type'] in [list, tuple]:
            kwargs['type'] = defaults[key]['element_type']
            kwargs['nargs'] = '*'  # '*' - zero or more elements, '+' - one or more elements

        parser.add_argument(*args, **kwargs)

    args = parser.parse_args()

    crl = CRL(**args.__dict__)
    crl.print_result()
    # crl.print_result(output_format='csv')
    # crl.print_result(output_format='json')
    # crl.print_result(output_format='plain_text')


def _convert_types(input_dict):
    for key in input_dict.keys():
        for el_key in input_dict[key]:
            if el_key in ['type', 'element_type']:
                input_dict[key][el_key] = eval(input_dict[key][el_key])
    return input_dict


def _read_json(file_name):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except IOError:
        raise Exception('The specified file <{}> not found!'.format(file_name))
    except ValueError:
        raise Exception('Malformed JSON file <{}>!'.format(file_name))
    return data
