#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:01:00 2021

@author: nulee
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import fftpack
import matplotlib as mpl
import os
# Global constants
C = 0.45  # Speed of light

# ---------------------
# Utility Functions
# ---------------------

def read_parameters(file_path):
    """
    Reads and prints the contents of a shock parameter file.

    Parameters:
    file_path (str): Path to the file directory.

    Returns:
    None
    """
    try:
        with open(file_path + "../input.shock", 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line)
    except FileNotFoundError:
        print(f"File not found: {file_path}../input.shock")

# ---------------------
# Analysis Functions
# ---------------------

def calculate_shock_parameters(delgam_e, c_omp, ppc0, mass_ratio, plasma_beta, gamma0):
    """
    Calculate and print various shock parameters.

    Parameters:
    delgam_e (float): Electron temperature in MeV
    c_omp (float): 
    ppc0 (int): Number of particles per cell
    mass_ratio (float): Mass ratio of ion to electron
    plasma_beta (float): Plasma beta value
    gamma0 (float): Initial Lorentz factor

    Returns:
    None
    """
    omp = C / c_omp
    relativistic_gamma = np.sqrt(1.0 / (1.0 - gamma0**2))
    qe = -(omp**2 * relativistic_gamma) / ((ppc0 * 0.5) * (1 + 1 / mass_ratio))
    qi = -qe
    delgam = delgam_e / mass_ratio
    me = abs(qi)
    mi = me * mass_ratio
    v_thi = np.sqrt(2 * delgam)
    v_the = v_thi * np.sqrt(mass_ratio)
    Binit = np.sqrt(4 * (ppc0 * 0.5) * delgam * mass_ratio * me * C**2 / plasma_beta)
    sigma1 = Binit**2 / relativistic_gamma / ppc0 / 0.5 / C**2 / me / (1 + 1 / mass_ratio)
    w_Li_ratio_to_wpe = 1 / 2 * np.sqrt(plasma_beta * 1 / delgam_e) * mass_ratio
    timestep = w_Li_ratio_to_wpe * 1 / omp
    gamma = 5 / 3
    cs = np.sqrt(2 * gamma * delgam)
    m0 = gamma0 * C / cs / C
    ms1 = (m0 * (gamma + 1) + np.sqrt((gamma + 1)**2 * m0**2 + 16)) / 4
    vsh = ms1 * cs
    simulated_vsh = vsh - gamma0 / C
    r_sh_for1gf = w_Li_ratio_to_wpe * (simulated_vsh)
    r_Li = mi * gamma0 * C / -qe / Binit / 10
    v_A = 2 * np.sqrt(delgam / plasma_beta)
    alfvenMs = vsh / v_A

    print("B0 = {:.6f}".format(Binit))
    print("vthi = {:.5f} c".format(v_thi))
    print("wLi/wpe = {:.1f}".format(w_Li_ratio_to_wpe))
    print("r_Li = {:.1f}".format(r_Li))
    print("timestep for 1wLi = {:.1f}".format(timestep))
    print("Mach number = {:.2f}".format(ms1))
    print("Cs= {:.5f} (c)".format(cs))
    print("vsh'(rest) = {:.5f} (c)".format(vsh))
    print("shock transition for 1 gf = {:.1f} c/wpe".format(r_sh_for1gf))
    print("vsh_simul = {:.5f} (c)".format(simulated_vsh))
    print("magnetization number : ", sigma1)
    print("alfven velocity : ", v_A)
    print("Alfven Ms = {:.2f}".format(alfvenMs))

# ---------------------
# Main Execution Block
# ---------------------

# Example of calling functions with appropriate parameters
file_path = "path/to/your/data/"
read_parameters(file_path)

# Define your parameters here


def file_open(path):
    ff=open(path+"../input.shock",'r')
    line=ff.readlines()
    for i in line:
        print(i)
    #print(line)
    ff.close()

def unified_spectrum(path, data_num, xmin, xmax, ymin, ymax, ion, mode):
    """
    Calculates the spectrum for either gamma or momentum based on the mode.

    Parameters:
    path (str): File path.
    data_num (int): Data number.
    xmin, xmax, ymin, ymax (float): Range for data selection.
    ion (bool): True for ion, False for electron.
    mode (str): 'gamma' for gamma spectrum, 'momentum' for momentum spectrum.

    Returns:
    Tuple (bins, histogram): Bins and normalized histogram data.
    """
    file_name = f"prtl.tot.{data_num:04d}" if mode == 'gamma' else f"prtl.tot.{data_num:03d}"
    f = h5py.File(os.path.join(path, file_name), 'r')
    
    particle_type = 'i' if ion else 'e'
    x = np.array(f[f'x{particle_type}'])
    y = np.array(f[f'y{particle_type}'])

    if mode == 'gamma':
        data = np.array(f[f'gamma{particle_type}'])
    else:
        u = np.array(f[f'u{particle_type}'])
        v = np.array(f[f'v{particle_type}'])
        w = np.array(f[f'w{particle_type}'])
        data = np.sqrt(u**2 + v**2 + w**2)

    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    filtered_data = data[mask]

    if mode == 'gamma':
        filtered_data = filtered_data[filtered_data > 1]
        data_to_hist = np.log10(filtered_data - 1)
    else:
        data_to_hist = np.log10(filtered_data)

    hist, bins = np.histogram(data_to_hist, bins=100 if mode == 'gamma' else 30)
    bins = 0.5 * (bins[1:] + bins[:-1])
    binsize = (bins[1] - bins[0]) * len(filtered_data)

    return 10**bins, hist / binsize * 0.4342

def adjust_field(data, name):
    """
    Adjusts the data based on the field name.

    Parameters:
    data (np.ndarray): Data array to be adjusted.
    name (str): Name of the field.

    Returns:
    np.ndarray: Adjusted data.
    """
    if name == 'bx':
        return np.cos(13 * np.pi / 180) * np.sin(90 * np.pi / 180) * 0.001163
    elif name == 'bz':
        return np.sin(13 * np.pi / 180) * np.sin(90 * np.pi / 180) * 0.001163
    else:
        return np.sin(13 * np.pi / 180) * np.cos(90 * np.pi / 180) * 0.001163

def compute_fft(path, datanum, horizontal, mx0, my0, mxmin, mxmax, mymin, mymax, name):
    """
    Computes the Fast Fourier Transform (FFT) of specified data.

    Parameters:
    path (str): File path for the data.
    datanum (int): Data file number.
    horizontal (bool): True for horizontal FFT, False for vertical.
    mx0, my0 (int): Size of the data array.
    mxmin, mxmax, mymin, mymax (int): Range for FFT computation.
    name (str): Field name to compute FFT on.

    Returns:
    Tuple of arrays (frequencies, fft_values)
    """
    field_file = f"flds.tot.{datanum:03d}"
    with h5py.File(path + field_file, 'r') as f:
        if name == 'eb':
            data = np.sqrt(np.array(f['bx'])**2 + np.array(f['by'])**2 + np.array(f['bz'])**2)
        elif name == 'ee':
            data = np.sqrt(np.array(f['ex'])**2 + np.array(f['ey'])**2 + np.array(f['ez'])**2)
        else:
            data = np.array(f[name])

    data = data[0, 2:my0-3, 2:mx0-3]  # Adjusted data slicing
    data -= adjust_field(data, name)  # Adjust field based on name

    k = []
    if horizontal:
        for i in range(mxmin, mxmax):
            k.append(np.mean(data[mymin:mymax, i]))
        xf = fftpack.fftfreq(mxmax - mxmin, d=0.1)
    else:
        for i in range(mymin, mymax):
            k.append(np.mean(data[i, mxmin:mxmax]))
        xf = fftpack.fftfreq(mymax - mymin, d=0.1)

    yf = fftpack.fft(k)
    return xf[:len(xf)//2][1:], yf[:len(yf)//2][1:]

def plot_phase_space(path, data_number, xmin, xmax, ymin, ymax, is_ion, direction):
    """
    Generates a 2D histogram plot representing the phase space.

    Parameters:
    path (str): Path to the data file.
    data_number (int): Data file number.
    xmin, xmax, ymin, ymax (float): Spatial boundaries for the plot.
    is_ion (bool): True for ion data, False for electron data.
    direction (str): Direction ('x', 'y', or 'z') for velocity data.

    Returns:
    None
    """
    file_name = f"prtl.tot.{data_number:03d}"
    particle_type = 'i' if is_ion else 'e'
    particle_data = {'x': f'x{particle_type}', 'y': f'y{particle_type}', 'v': f'v{direction}{particle_type}'}
    
    with h5py.File(path + file_name, 'r') as file:
        x, y, v = file[particle_data['x']][:], file[particle_data['y']][:], file[particle_data['v']][:]
    
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    filtered_x, filtered_v = x[mask], v[mask]

    x_bins = np.linspace(xmin, xmax, xmax - xmin)
    v_bins = np.linspace(-0.4 if is_ion else -4, 0.4 if is_ion else 4, 100 if is_ion else 300)

    plt.rcParams.update({'font.size': 40})
    plt.figure(figsize=(50, 10))
    plt.hist2d(filtered_x, filtered_v, bins=[x_bins, v_bins], norm=mpl.colors.LogNorm(), cmap='plasma')
    plt.xlabel(f"c/$\omega$$_{pe}$")
    plt.ylabel(f"p$_{particle_type}$$_{direction}/m_{particle_type}c$", fontdict={'size': 60})
    plt.show()

def calculate_energy_components(path, data_number):
    """
    Calculates energy components from particle and field data.

    Parameters:
    path (str): Path to the data file.
    data_number (int): Data file number.

    Returns:
    Tuple (kinetic_energy, magnetic_field_energy, electric_field_energy)
    """
    prt_file = f"prtl.tot.{data_number:03d}"
    fld_file = f"flds.tot.{data_number:03d}"
    param_file = f"param.{data_number:03d}"
    
    with h5py.File(path + prt_file, 'r') as prt, h5py.File(path + fld_file, 'r') as fld, h5py.File(path + param_file, 'r') as param:
        mx0, my0 = param['mx0'][0], param['my0'][0]
        kinetic_energy = np.sum((prt['gammae'][:] - 1) * 0.45**2 * param['me'][0] + (prt['gammai'][:] - 1) * 0.45**2 * param['mi'][0])
        magnetic_field_energy = np.sum(fld['bx'][0, 2:my0 - 3, 2:mx0 - 3]**2 + fld['by'][0, 2:my0 - 3, 2:mx0 - 3]**2 + fld['bz'][0, 2:my0 - 3, 2:mx0 - 3]**2)
        electric_field_energy = np.sum(fld['ex'][0, 2:my0 - 3, 2:mx0 - 3]**2 + fld['ey'][0, 2:my0 - 3, 2:mx0 - 3]**2 + fld['ez'][0, 2:my0 - 3, 2:mx0 - 3]**2)

    return kinetic_energy, magnetic_field_energy, electric_field_energy

def calculate_velocity_distribution(path, data_number, xmin, xmax, ymin, ymax, is_ion):
    """
    Calculates the velocity distribution histogram.

    Parameters:
    path (str): Path to the data file.
    data_number (int): Data file number.
    xmin, xmax, ymin, ymax (float): Spatial boundaries for the calculation.
    is_ion (bool): True for ions, False for electrons.

    Returns:
    Tuple of arrays (velocity_bins, histogram_values) for each velocity component.
    """
    prt_file = f"prtl.tot.{data_number:03d}"
    particle_type = 'i' if is_ion else 'e'

    with h5py.File(path + prt_file, 'r') as file:
        x = file[f'x{particle_type}'][:]
        u, v, w = file[f'u{particle_type}'][:], file[f'v{particle_type}'][:], file[f'w{particle_type}'][:]

    mask = (x >= xmin) & (x <= xmax)
    filtered_u, filtered_v, filtered_w = u[mask], v[mask], w[mask]

    hist_u, bins_u = np.histogram(filtered_u, bins=50)
    hist_v, bins_v = np.histogram(filtered_v, bins=50)
    hist_w, bins_w = np.histogram(filtered_w, bins=50)
    
    return np.array([bins_u[:-1], hist_u, bins_v[:-1], hist_v, bins_w[:-1], hist_w])