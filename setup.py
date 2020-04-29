import os

from setuptools import setup, find_packages

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "covid_model_deaths", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'covid-shared',
        'dill',
        'ipdb',
        'jupyter',
        'jupyterlab',
        'loguru',
        'matplotlib',
        'numpy',
        'openpyxl',
        'pandas',
        'pyarrow',
        'pyyaml',
        'scipy',
        'seaborn',
        'sklearn',
        'tqdm',
        'xlrd',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    doc_requirements = [
        'sphinx',
        'sphinx-click',
        'sphinx-autodoc-typehints',
        'sphinx-rtd-theme'
    ]

    internal_requirements = [
        'curvefit @ git+https://github.com/ihmeuw-msca/CurveFit@product#egg=curvefit'
        # Only available inside IHME infrastructure
        'db_queries',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        extras_require={
            'docs': doc_requirements,
            'internal': internal_requirements,
            'test': test_requirements,
            'dev': test_requirements + doc_requirements,
        },

        zip_safe=False,
    )
