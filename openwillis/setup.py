from setuptools import setup, find_namespace_packages

#Long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='openwillis_test',
    version='3.0.0',
    description='digital health measurement',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bklynhlth/openwillis',
    author='bklynhlth',
    python_requires='>=3.9, <3.11',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    author_email='admin@bklynhlth.com',
    license='Apache',
    install_requires=[
        'openwillis_test-voice==1.0.0',
        'openwillis_test-transcribe==1.0.0',
        'openwillis_test-gps==1.0.0',
        'openwillis_test-speech==1.0.0',
        'openwillis_test-face==1.0.0',
    ],
)