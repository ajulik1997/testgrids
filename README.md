<h1 align="center">
    <br>
    testgrids
    <br>
</h1>

<h4 align="center">A Python3 script for work towards automatic fine-grid determination of CASTEP OTF pseudopotentials</h4>
<p align="center">
  <a href="https://www.python.org/downloads/release/python-375">
        <img src="https://img.shields.io/badge/python-3.7-brigtgreen.svg" alt="Python 3.7">
  </a>
  <a href="http://www.castep.org/">
        <img src="https://img.shields.io/badge/CASTEP-18.1.2-blue" alt="CASTEP 18.1.2">
  </a>
  <a href="">
        <img src="https://img.shields.io/github/languages/code-size/ajulik1997/testgrids" alt="Code size">
  </a>
  <a href="">
        <img src="https://img.shields.io/github/repo-size/ajulik1997/testgrids" alt="Repo size">
  </a>
   <a href="https://github.com/ajulik1997/testgrids/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/ajulik1997/testgrids" alt="MIT Licence">
  </a>
</p>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#readme">Readme</a> •
  <a href="#authors">Authors</a>
</p>

# Description

`testgrids.py` is a script written in Python 3 that automates the creation, submission and analysis of CASTEP jobs in order to determine the optimal `FINE_GRID_SIZE` value for that simulation.

This script is able to automatically generate custom data sets with a uniformly varying `FINE_GRID_SIZE` parameter, which are then submitted as CASTEP jobs to a SLURM Workload Manager queue. The data from completed (or failed) jobs is then aggregated, analysed, plotted and summarised into both machine- and human-readable files.

Ultimately, the goal of any future work on this script would be the determination of a set of `FINE_GMAX` values corresponding to `FINE_GRID_SIZE` precision levels from `COARSE` to `EXTREME` for all supported elements.

# Readme

Please consult the README [PDF document](https://github.com/ajulik1997/testgrids/blob/master/README.pdf) for information regarding the following:

- installation procedure
- required and optional dependencies
- usage examples for a number of different use-cases
- summary of available parameters, their default and accepted values
- tips for troubleshooting various error and warning messages
- summary of possible return codes and their meaning
- overview of the code for future development
- analysis of format used by all output files

The README document is typeset using `LaTeX`, with the source code available in [`testgrids/LaTeX`](https://github.com/ajulik1997/testgrids/tree/master/LaTeX).

# Authors

The code is written by [Alexander Liptak](https://www.fusion-cdt.ac.uk/student/alexander-liptak), under the supervision of [Prof. Keith Refson](https://pure.royalholloway.ac.uk/portal/en/persons/keith-refson(b347917c-bce2-4b5c-b3fe-2c2fd8c71616).html), as part of the 2019 UKCP Summer Project carried out at the Royal Holloway University of London.

<div>

<div>
    <img style="float: left; height: 125px;", src="https://intranet.royalholloway.ac.uk/staff/assets/img/brand-toolkit/colour-logo-new.png", alt="RHUL logo">
    <img style="float: right; height: 125px;" src="http://www.castep.org/files/CASTEP_Logo_mini-01.png", alt="CASTEP logo">
</div>



