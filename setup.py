#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup

with open("bookit/include/VERSION", "r") as f:
    version = f.read().strip()

setup(
    name          = 'bookit',
    version       = version,
    license       = 'BSD 3-Clause License',
    description   = 'Description text',
    url           = 'https://github.com/tklijnsma/bookit.git',
    author        = 'Thomas Klijnsma',
    author_email  = 'tklijnsm@gmail.com',
    packages      = ['bookit'],
    package_data  = {'bookit': ['include/*']},
    include_package_data = True,
    zip_safe      = False,
    scripts       = [
        'bin/bookit-console',
        'bin/bookit-version',
        'bin/bookit-parse-transactions',
        'bin/bookit-add-transactions'
        ]
    )
