"""Display wording shared by the GUI and its headless presentation checks."""
from car_ai import FEATURES

LABEL_NAMES = {'unacc': 'Unacceptable', 'acc': 'Acceptable',
               'good': 'Good', 'vgood': 'Very good'}
FIELD_NAMES = dict(zip(FEATURES, ('Purchase cost', 'Maintenance cost', 'Doors',
                                'Passenger capacity', 'Boot size', 'Safety')))
TOKENS = (('vhigh', 'high', 'med', 'low'), ('vhigh', 'high', 'med', 'low'),
          ('2', '3', '4', '5more'), ('2', '4', 'more'),
          ('small', 'med', 'big'), ('low', 'med', 'high'))
FRIENDLY = {'vhigh': 'Very high', 'high': 'High', 'med': 'Medium', 'low': 'Low',
            '5more': '5 or more', 'more': 'More than 4', 'small': 'Small', 'big': 'Large'}


def display_value(token):
    return FRIENDLY.get(token, token)


def display_condition(index, token):
    return f'{FIELD_NAMES[FEATURES[index]]}: {display_value(token)}'


def display_conditions(rule):
    return [display_condition(i, value) for i, value in rule['conditions']] or ['All valid inputs']


def input_summary(values):
    return '  |  '.join(display_condition(i, value) for i, value in enumerate(values))
