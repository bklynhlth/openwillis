<img src="https://github.com/bklynhlth/openwillis/blob/main/resources/willis-openwillis.png" width="150">

OpenWillis is a python library for digital health measurement.

It was developed by [Brooklyn Health](https://brooklyn.health/openwillis) to establish standardized methods in digital phenotyping and make them open and accessible to the scientific community.

It is freely available for non-commercial use ([see license](https://github.com/bklynhlth/openwillis/blob/main/LICENSE.txt)).

The [OpenWillis Wiki](https://brooklynhealth.notion.site/OpenWillis-14983a8fe04781ddb2a2e999aeaaf05a) contains detailed documentation on the following: 
1. Function methods and documentation
2. Release notes
3. Instructions for getting started
4. Research guidelines
5. Contribution guidelines
6. User community events

Please use the following reference when reporting work that has used OpenWillis:
Worthington, M., Efstathiadis, G., Yadav, V., & Abbas, A. (2024). 172. OpenWillis: An Open-Source Python Library for Digital Health Measurement. *Biological Psychiatry*, 95(10), S169-S170.

Please report any issues using the [Issues](https://github.com/bklynhlth/openwillis/issues) tab.

If you’d like to contribute to OpenWillis or have general questions, please [get in touch](mailto:openwillis@brooklyn.health).

### Brief instructions for getting started 
Certain requirements are required prior to installing OpenWillis. For full details, please see installation instructions [here](https://brooklynhealth.notion.site/Installing-OpenWillis-14983a8fe047814b88ced7d3831791f2).

OpenWillis can be installed from PyPI using pip: 
```
pip install openwillis
```
#### Example use:
Below is an example use of the `facial_expressivity` function to calculate expressivity from a video.
```
import openwillis as ow

framewise_loc, framewise_disp, summary = ow.facial_expressivity('data/video.mp4', 'data/baseline.mp4')
```
All OpenWillis functions are listed in the wiki's [List of Functions](https://brooklynhealth.notion.site/15883a8fe04780739400c1d8ad94bb39?v=15883a8fe047806aa291000cb85dceae).

Each function has a document that details its use, methods utilized, input and output parameters, primary outcome measures, and any additional information relevant for the use of the function.


