import json
import os


def console(class_name, parameters_file):
    import argparse

    data = read_json(parameters_file)
    description = data['description']
    parameters = convert_types(data['parameters'])

    # Processing arguments:
    required_args = []
    optional_args = []

    for key in sorted(parameters.keys()):
        if parameters[key]['default'] is None:
            required_args.append(key)
        else:
            optional_args.append(key)

    parser = argparse.ArgumentParser(description=description)

    for key in required_args + optional_args:
        args = []
        if 'short_argument' in parameters[key]:
            args.append('-{}'.format(parameters[key]['short_argument']))
        args.append('--{}'.format(key))

        kwargs = {
            'dest': key,
            'default': parameters[key]['default'],
            'required': False,
            'type': parameters[key]['type'],
            'help': '{}.'.format(parameters[key]['help']),
        }
        if parameters[key]['default'] is None:
            kwargs['required'] = True

        if parameters[key]['type'] == bool:
            kwargs['action'] = 'store_true'
            del (kwargs['type'])

        if parameters[key]['type'] == str:
            pass

        if parameters[key]['type'] in [list, tuple]:
            kwargs['type'] = parameters[key]['element_type']
            kwargs['nargs'] = '*'  # '*' - zero or more elements, '+' - one or more elements

        parser.add_argument(*args, **kwargs)

    args = parser.parse_args()
    for key in args.__dict__.keys():
        if type(args.__dict__[key]) == str and args.__dict__[key] == 'None':
            args.__dict__[key] = None

    class_name(**args.__dict__)


def convert_types(input_dict):
    for key in input_dict.keys():
        for el_key in input_dict[key]:
            if el_key in ['type', 'element_type']:
                input_dict[key][el_key] = eval(input_dict[key][el_key])
    return input_dict


def defaults_file(suffix=None, defaults_file_path=None):
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Fix for Jython:
    try:
        script_path = script_path.replace(os.path.join(format(os.environ['HOME']), '.jython-cache/cachedir/classes'),
                                          '')
    except:
        pass

    dat_dir = os.path.join(script_path, 'dat')
    config_dir = os.path.join(script_path, 'configs')
    if not defaults_file_path:
        file_name = 'defaults_{}.json'.format(suffix) if suffix else 'defaults.json'
        defaults_file = os.path.join(config_dir, file_name)
    else:
        defaults_file = defaults_file_path

    return {
        'dat_dir': dat_dir,
        'config_dir': config_dir,
        'defaults_file': defaults_file,
    }


def read_json(file_name):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except IOError:
        raise Exception('The specified file <{}> not found!'.format(file_name))
    except ValueError:
        raise Exception('Malformed JSON file <{}>!'.format(file_name))
    return data
