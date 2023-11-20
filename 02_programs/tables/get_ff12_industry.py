# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

nodur = [
    range(100, 999 + 1),
    range(2000, 2399 + 1),
    range(2700, 2749 + 1),
    range(2770, 2799 + 1),
    range(3100, 3199 + 1),
    range(3940, 3989 + 1),
]

durbl = [
    range(3716, 3716 + 1),
    range(3750, 3751 + 1),
    range(3792, 3792 + 1),
    range(3900, 3939 + 1),
    range(3990, 3999 + 1),
]

manuf = [
    range(2520, 2589 + 1),
    range(2600, 2699 + 1),
    range(2750, 2769 + 1),
    range(3000, 3099 + 1),
    range(3200, 3569 + 1),
    range(3580, 3629 + 1),
    range(3700, 3709 + 1),
    range(3712, 3713 + 1),
    range(3715, 3715 + 1),
    range(3717, 3749 + 1),
    range(3752, 3791 + 1),
    range(3793, 3799 + 1),
    range(3830, 3839 + 1),
    range(3860, 3899 + 1),
]

energy = [range(1200, 1399 + 1), range(2900, 2999 + 1)]

chems = [range(2800, 2829 + 1), range(2840, 2899 + 1)]

buseq = [
    range(3570, 3579 + 1),
    range(3660, 3692 + 1),
    range(3694, 3699 + 1),
    range(3810, 3829 + 1),
    range(7370, 7379 + 1),
]

telcom = [range(4800, 4899 + 1)]

utils = [range(4900, 4949 + 1)]

whole = [
    range(5000, 5999 + 1),
    range(7200, 7299 + 1),
    range(7600, 7699 + 1),
]

hlth = [range(2830, 2839 + 1), range(3693, 3693 + 1), range(3840, 3859 + 1), range(8000, 8099 + 1)]

fin = [range(6000, 6999 + 1)]


mapper = {
    "Consumer nondurables": nodur,
    "Consumer durables": durbl,
    "Manufacturing": manuf,
    "Energy": energy,
    "Chemicals": chems,
    "Business Equipment": buseq,
    "Telecom": telcom,
    "Utilities": utils,
    "Wholesale": whole,
    "Healthcare": hlth,
    "Finance": fin,
}

mapper_sics = {}
for key, value in mapper.items():
    tmp = []
    for r in value:
        tmp.extend([i for i in r])
    mapper_sics[key] = tmp


def get_ff12_industry(sic):
    for key, value in mapper_sics.items():
        if sic in value:
            return key
    return "Other"
