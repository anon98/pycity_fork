"""
The pycity_scheduling framework


Copyright (C) 2023,
Institute for Automation of Complex Power Systems (ACS),
E.ON Energy Research Center (E.ON ERC),
RWTH Aachen University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


"""
Dictionary with electric vehicle data from German ADAC e.V.:
https://www.adac.de/rund-ums-fahrzeug/elektromobilitaet/kaufen/elektroautos-uebersicht/ (accessed on 2023/03/27)
"""

ev_data = dict()


ev_data['EV.01'] = {'name': 'Audi Q4',
                    'e_el_storage_max': 82.0,
                    }

ev_data['EV.02'] = {'name': 'BMW i3',
                    'e_el_storage_max': 37.9,
                    }

ev_data['EV.03'] = {'name': 'Citroen e-Berlingo',
                    'e_el_storage_max': 50.0,
                    }

ev_data['EV.04'] = {'name': 'Fiat 500e',
                    'e_el_storage_max': 23.8,
                    }

ev_data['EV.05'] = {'name': 'Renault ZOE',
                    'e_el_storage_max': 55.0,
                    }

ev_data['EV.06'] = {'name': 'Mercedes eVito',
                    'e_el_storage_max': 66.0,
                    }

ev_data['EV.07'] = {'name': 'Nissan Leaf',
                    'e_el_storage_max': 40.0,
                    }

ev_data['EV.08'] = {'name': 'Hyundai Kona',
                    'e_el_storage_max': 42.0,
                    }

ev_data['EV.09'] = {'name': 'Volvo XC40',
                    'e_el_storage_max': 82.0,
                    }

ev_data['EV.10'] = {'name': 'VW ID.3',
                    'e_el_storage_max': 82.0,
                    }

ev_data['EV.11'] = {'name': 'Renault Kangoo',
                    'e_el_storage_max': 45.0,
                    }

ev_data['EV.12'] = {'name': 'Porsche Taycan  Sport Turismo',
                    'e_el_storage_max': 93.4,
                    }

ev_data['EV.13'] = {'name': 'Opel Mokka-e',
                    'e_el_storage_max': 50.0,
                    }

ev_data['EV.14'] = {'name': 'Mazda MX-30 e-SKYACTIVE',
                    'e_el_storage_max': 35.5,
                    }

ev_data['EV.15'] = {'name': 'Tesla Model 3',
                    'e_el_storage_max': 80.5,
                    }

ev_data['EV.16'] = {'name': 'Mercedes Benz EQC 400 4MATIC',
                    'e_el_storage_max': 85.0,
                    }

ev_data['EV.17'] = {'name': 'Ford Mustang Mach-E Standard Range',
                    'e_el_storage_max': 75.7,
                    }
