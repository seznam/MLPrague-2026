import runpy
from setuptools import setup
from setuptools import find_packages


__version__ = runpy.run_path("src/ml_prague_2026/version.py")["__version__"]


setup(
    name='szn-advertising-research-ml-prague-2026',
    package_dir={'': 'src'},
    include_package_data=True,
    packages=find_packages('src'),
    version=__version__,
    description='Research project',
    long_description='Research project',
    long_description_content_type="text/plain",
    author='Seznam.cz, a.s.',
    author_email='sklik.vyzkum@firma.seznam.cz',
    url="git@gitlab.seznam.net:advertising-research/common/ml-prague-2026.git",
    license='Proprietary',
    install_requires=[
        # here comes all package dependencies
    ]
)
