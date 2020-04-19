import os

from setuptools import setup, find_packages

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "covid_model_deaths", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.md")) as f:
        long_description = f.read()

    install_requirements = [
        'jupyter',
        'jupyterlab',
        'ipdb',
        'dill',
        'pandas',
        'xlrd',
        'loguru',
        'pyyaml',
        'pyarrow',
        'openpyxl',
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'db_queries',
        'curvefit @ git+https://github.com/ihmeuw-msca/CurveFit@product#egg=curvefit'
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
            'test': test_requirements,
            'dev': test_requirements + doc_requirements,
            'docs': doc_requirements
        },

        zip_safe=False,
    )
