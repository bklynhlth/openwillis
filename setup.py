# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

import setuptools

#Long description
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    install_requires = fp.read()

setuptools.setup(name='openwillis',
                 version='0.2.3',
                 description='digital health measurement',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/bklynhlth/openwillis',
                 author='bklynhlth',
                 python_requires='>=3.6.*',
                 install_requires=install_requires,
                 author_email='admin@bklynhlth.com',
                 packages=setuptools.find_packages(),
                 zip_safe=False,
                 license='Apache'
                )
