from setuptools import setup, find_namespace_packages

with open("requirements.txt", "r") as fp:
    install_requires = fp.read()

setup(
    name='openwillis-voice',
    version='1.0.4',
    description='digital health measurement',
    long_description_content_type="text/markdown",
    url='https://github.com/bklynhlth/openwillis',
    author='bklynhlth',
    python_requires='>=3.9, <3.13',
    install_requires=install_requires,
    author_email='admin@bklynhlth.com',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    license='Apache'
)
