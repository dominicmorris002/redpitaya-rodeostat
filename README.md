# Mobile AC Cyclic Voltammetry System Research - Rodeostat & Red Pitaya 

Hey! This project is for making an **AC Cyclic Voltammetry (ACCV) setup** using a Rodeostat potentiostat and two Red Pitaya devices.  

It’s all written in Python 3.8 and tested in PyCharm. PyRPL is used in this project to allow the Red Pitaya to act as a comerical Lock In Amplifier using continous data streaming and the FPGA.

---

Whats in here?

There are three main programs:

1. **Rodeostat Potentiostat**
   - Do regular Cyclic Voltammetry (CV)
   - Manual control of the Rodeostat

2. **Red Pitayas**
   - Read input/output with a basic oscilloscope
   - Lock-In Amplifier using PyRPL and AC Signal Generator
   - Measure DC steps from the potentiostat faster than Rodeostat

3. **Both Red Pitayas together**
   - Run both Red Pitayas at the same time to measure DC Ramp and Lock-On Amplifier Response and create the saved files and HTML Plots
   - Do AC with DC bias experiments (ACCV)



Features

- Saves data as **CSV files**
- Plots using python and HTML for interactive plots
- Uses a custom **PyRPL controller** with its own `scope.yaml`
- Easy to read console output

Results

![image](https://github.com/user-attachments/assets/6a56229c-7648-4caf-bdcf-074c723a72bd)




the Rodeostat was found to be highly accurate and produced the same data as a Autolab Potentiostat for our Experiments


<img width="1472" height="923" alt="image" src="https://github.com/user-attachments/assets/51a01dc8-aa69-4430-a9c1-bf164aa749e3" />


Make sure to calibrate the devices to your potentiostat, the Redpitaya was found to be highly accurate and produced the same data as a Signal Recovery 7280 DSP Lock In Amplifier


---------------------------------------------------------------------------------------------------

## Notes

I recommend downloading each of these before running.

# Python version
python==3.8.10

# Core packages
numpy==1.23.5
pandas==2.0.3
matplotlib==3.7.5
pyqtgraph==0.11.1
scipy==1.10.1
PyDAQmx==1.4.7

# PyRPL and Red Pitaya
pyrpl==0.9.7.0
rpds-py==0.20.1

# Potentiostat
potentiostat==0.0.4

# GUI / Qt dependencies
PyQt5==5.15.9
PyQt5-Qt5==5.15.2
PyQt5_sip==12.15.0
QtPy==1.9.0
qasync==0.28.0
Quamash==0.6.1

# Jupyter / Notebook stuff (optional)
ipython==8.12.3
jupyter_client==8.6.3
jupyter_core==5.8.1
nbclient==0.10.1
nbconvert==7.16.6
nbformat==5.10.4
matplotlib-inline==0.1.7

# Misc utilities
asttokens==3.0.0
attrs==25.3.0
backcall==0.2.0
bcrypt==5.0.0
beautifulsoup4==4.14.3
bleach==6.1.0
cffi==1.17.1
colorama==0.4.6
contourpy==1.1.1
cryptography==46.0.3
cycler==0.12.1
decorator==5.2.1
defusedxml==0.7.1
executing==2.2.1
fastjsonschema==2.21.2
fonttools==4.57.0
importlib_metadata==8.5.0
importlib_resources==6.4.5
jedi==0.19.2
Jinja2==3.1.6
jsonschema==4.23.0
jsonschema-specifications==2023.12.1
kiwisolver==1.4.7
MarkupSafe==2.1.5
mistune==3.2.0
nose==1.3.7
packaging==25.0
pandocfilters==1.5.1
paramiko==3.5.1
parso==0.8.5
pickleshare==0.7.5
pillow==10.4.0
pip==25.0.1
pkgutil_resolve_name==1.3.10
platformdirs==4.3.6
progressbar2==4.5.0
prompt_toolkit==3.0.52
pure_eval==0.2.3
pycparser==2.23
PyNaCl==1.6.0
pyparsing==3.1.4
pyserial==3.5
python-dateutil==2.9.0.post0
python-utils==3.8.2
pytz==2025.2
pywin32==311
PyYAML==6.0.3
pyzmq==27.1.0
referencing==0.35.1
scp==0.15.0
setuptools==75.3.2
six==1.17.0
soupsieve==2.7
stack-data==0.6.3
tinycss2==1.2.1
tornado==6.4.2
traitlets==5.14.3
typing_extensions==4.13.2
tzdata==2025.2
wcwidth==0.2.14
webencodings==0.5.1
wheel==0.45.1
zipp==3.20.2



Have a great day!   
— Dominic Morris :)
