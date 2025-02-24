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


from .stand_alone_optimization_algorithm import StandAlone
from .local_optimization_algorithm import LocalOptimization
from .exchange_admm_algorithm import ExchangeADMM
from .exchange_admm_algorithm_mpi import ExchangeADMMMPI
from .central_optimization_algorithm import CentralOptimization
from .dual_decomposition_algorithm import DualDecomposition
from .dual_decomposition_algorithm_mpi import DualDecompositionMPI
from .exchange_miqp_admm_algorithm import ExchangeMIQPADMM
from .exchange_miqp_admm_algorithm_mpi import ExchangeMIQPADMMMPI


__all__ = [
    'StandAlone',
    'LocalOptimization',
    'ExchangeADMM',
    'ExchangeADMMMPI',
    'CentralOptimization',
    'DualDecomposition',
    'DualDecompositionMPI',
    'ExchangeMIQPADMM',
    'ExchangeMIQPADMMMPI',
    'algorithm',
    'algorithms',

]


algorithms = {
    'stand-alone': StandAlone,
    'local': LocalOptimization,
    'exchange-admm': ExchangeADMM,
    'exchange-admm-mpi': ExchangeADMMMPI,
    'central': CentralOptimization,
    'dual-decomposition': DualDecomposition,
    'dual-decomposition-mpi': DualDecompositionMPI,
    'exchange-miqp-admm': ExchangeMIQPADMM,
    'exchange-miqp-admm-mpi': ExchangeMIQPADMMMPI,
}
