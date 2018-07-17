import os, io
from setuptools import setup, find_packages

setup(
    name='lightroot',
    version='1.0',
    author='Sirsh',
    author_email='amartey@gmail.com',
    license='MIT',
    url='git@https://github.com/GiovanniSena/LSFM_CYCB_analysis_v2',
    keywords='root microscopy tracking fuzzy registration',
    description='lighroot is an image processing pipeline for preprocessing lightsheet microscopy data and tracking transient fluorescent events in structured point clouds',
    long_description=('lighroot is an image processing pipeline for preprocessing lightsheet microscopy data and tracking transient fluorescent events in structured point clouds'),
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': [
            'lightroot = lightroot.__main__:main'
            ],
    },
    classifiers=[
        'Development Status :: Beta',
        'Intended Audience :: Developers',
        'License :: MIT',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Communications :: Chat',
        'Topic :: Internet',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)

