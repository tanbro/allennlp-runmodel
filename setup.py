#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
setuptools script file
"""

from setuptools import setup, find_packages

setup(
    name='allennlp-runmodel',
    namespace_packages=[],
    packages=find_packages('src'),
    package_dir={'': 'src'},

    description='Run a AllenNLP trained model, and serve it with WebAPI.',
    url='https://github.com/tanbro/allennlp-runmodel',
    author='liu xue yan',
    author_email='liu_xue_yan@foxmail.com',

    use_scm_version={
        # guess-next-dev:   automatically guesses the next development version (default)
        # post-release:     generates post release versions (adds postN)
        'version_scheme': 'guess-next-dev',
        'write_to': 'src/allennlp_runmodel/_version.py',
    },
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],

    install_requires=[
        'allennlp<0.8,>=0.7',
        'aiohttp<4.0,>=3.4',
        'click',
        'PyYAML',
    ],

    extras_require={
    },

    tests_require=[],

    package_data={
    },

    entry_points={
        'console_scripts': [
            'allennlp-runmodel = allennlp_runmodel.commands.run:cli',
        ],
    },

    python_requires='>=3.6'
)
