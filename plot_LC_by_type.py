#!/usr/bin/env python
import numpy as np
from pyfits import getdata
from glob import glob
import numpy as np
from multiprocessing import Pool
from scipy.optimize import leastsq
from numpy.lib import recfunctions
from useful import extinction_coeff
import esutil
from matplotlib import rc
from matplotlib.backends.backend_pdf import FigureCanvasPdf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPdf)
import pickle, os
from dust import getval
from esutil.coords import eq2gal
from matplotlib import pyplot as plt
from esutil.coords import sphdist
import sys, getopt
from esutil.numpy_util import match

###########################################################################################################################################

def model(ind):
    # model parameters
    period = true_params['P'][ind]
    gA = true_params['gA'][ind]
    rA = true_params['rA'][ind]
    iA = true_params['iA'][ind]
    zA = true_params['zA'][ind]
    gr0 = true_params['g0'][ind] - true_params['r0'][ind]
    ri0 = true_params['r0'][ind] - true_params['i0'][ind]
    rz0 = true_params['r0'][ind] - true_params['z0'][ind]
    # add ab templates
    if true_params['gT'][ind] > 99:
        print true_params['rT'][ind] - 100
        print templates_r.shape
        template_g = templates_g[:, true_params['gT'][ind] - 100]
        template_r = templates_r[:, true_params['rT'][ind] - 100]
        template_i = templates_i[:, true_params['iT'][ind] - 100]
        template_z = templates_z[:, true_params['zT'][ind] - 100]
    else:
        template_g = templates_g[:, -2 + true_params['gT'][ind]]
        template_r = templates_r[:, -2 + true_params['rT'][ind]]
        template_i = templates_i[:, -2 + true_params['iT'][ind]]
        template_z = templates_z[:, -2 + true_params['zT'][ind]]
    return period, gA, rA, iA, zA, gr0, ri0, rz0, template_g, template_r, template_i, template_z

###########################################################################################################################################

def plot_lcs(chi2_dof, period, phi, r0, F, ind, typ, starID, P_true, P_template):
    a = 3.47
    fig = plt.figure(figsize=(a,a), num=1)
    ax = fig.add_subplot(111)
    mjdg, gLC, gErr, mjdr, rLC, rErr, mjdi, iLC, iErr, mjdz, zLC, zErr, mjdy, yLC, yErr = lc_data
    _, gA, rA, iA, zA, gr0, ri0, rz0, template_g, template_r, template_i, template_z = model(ind)
    # calculate phases
    phase_g = (mjdg - phi)/period - np.floor((mjdg - phi)/period)
    phase_r = (mjdr - phi)/period - np.floor((mjdr - phi)/period)
    phase_i = (mjdi - phi)/period - np.floor((mjdi - phi)/period)
    phase_z = (mjdz - phi)/period - np.floor((mjdz - phi)/period)
    phase_y = (mjdy - phi)/period - np.floor((mjdy - phi)/period)
    # calculate model
    model_g = F*gA*template_g + gr0 + r0
    model_r = F*rA*template_r + r0
    model_i = F*iA*template_i + r0 - ri0
    model_z = F*zA*template_i + r0 - rz0

    plt.clf()
    plt.figure(figsize=(14,10))

    plt.errorbar(phase_g, gLC, yerr=gErr, fmt='o', ecolor='g', mfc='g')
    plt.errorbar(phase_r, rLC, yerr=rErr, fmt='o', ecolor='r', mfc='r')
    plt.errorbar(phase_i, iLC, yerr=iErr, fmt='o', ecolor='y', mfc='y')
    plt.errorbar(phase_z, zLC, yerr=zErr, fmt='o', ecolor='k', mfc='k')
    plt.errorbar(phase_y, yLC, yerr=yErr, fmt='o', ecolor='b', mfc='b')
    plt.plot(phase_model, model_g, 'g-')
    plt.plot(phase_model, model_r, 'r-')
    plt.plot(phase_model, model_i, 'y-')
    plt.plot(phase_model, model_z, 'k-')
    plt.axis([0, 1, np.max([np.max(model_g), np.max(model_r), np.max(model_i), np.max(model_z)])+0.1, np.min([np.min(model_g), np.min(model_r), np.min(model_i), np.max(model_z)])-0.1])
    plt.xlabel('phase')
    plt.ylabel('magnitude')
    fig.subplots_adjust(left=0.18, bottom=0.15, right=0.96, top=0.97)
    
    plt.text(0.5,0.3,'chi2_dof:'+str(chi2_dof)+"\n"+"P_true:"+str(P_true)+"\n"+"P_best:"+str(period)+"\n"+"P_template:"+str(P_template)+"\n"+"templateID:"+str(ind)+"\n"+"RRlyrType:"+str(typ)+"\n"+"File#:"+str(starID), horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes)
#    plt.show()
    plt.draw()
    plt.savefig(typ + '_%s_fitted_lightcurve.pdf' % starID)

###########################################################################################################################################

def load_PS1_data(lc_filename):
    # load cleaned PS1 data and correct magnitude errors
    lc_ps = np.genfromtxt(lc_filename, usecols=(1, 2, 3, 6, 8, 17), names='mag_err, ra, dec, filter, mjd_obs, ucalmag', dtype='f8, f8, f8, |S5, f8, f8')
    lc_ps['mag_err'] = np.sqrt((1.3*lc_ps['mag_err'])**2 + 0.015**2)
    # correct PS1 light curves for extinction
    gl, gb = eq2gal(lc_ps['ra'][0], lc_ps['dec'][0])
    EBV = getval(gl, gb)
    for band, c in zip(['g', 'r', 'i', 'z', 'y'], [3.172, 2.271, 1.682, 1.322, 1.087]):
        lc_ps['ucalmag'][lc_ps['filter'] == band] -= c*EBV
    # light curve data used in fitting
    lc_data = [lc_ps['mjd_obs'][lc_ps['filter'] == 'g'], lc_ps['ucalmag'][lc_ps['filter'] == 'g'], lc_ps['mag_err'][lc_ps['filter'] == 'g'], lc_ps['mjd_obs'][lc_ps['filter'] == 'r'], lc_ps['ucalmag'][lc_ps['filter'] == 'r'], lc_ps['mag_err'][lc_ps['filter'] == 'r'], lc_ps['mjd_obs'][lc_ps['filter'] == 'i'], lc_ps['ucalmag'][lc_ps['filter'] == 'i'], lc_ps['mag_err'][lc_ps['filter'] == 'i'], lc_ps['mjd_obs'][lc_ps['filter'] == 'z'], lc_ps['ucalmag'][lc_ps['filter'] == 'z'], lc_ps['mag_err'][lc_ps['filter'] == 'z'],lc_ps['mjd_obs'][lc_ps['filter'] == 'y'], lc_ps['ucalmag'][lc_ps['filter'] == 'y'], lc_ps['mag_err'][lc_ps['filter'] == 'y']]
    return lc_data

###########################################################################################################################################

# load all templates into an array
def load_templates(template_list):
    templates = np.zeros((500, template_list.size))
    for i, fname in enumerate(template_list):
        aux = getdata(fname)
        templates[:, i] = aux
    return templates

###########################################################################################################################################

#### main program #####

if __name__ == "__main__": 

  other_path = None
  rrlyr_path = None
  file_name = None
  home_path = None
  template_path = None

  if len(sys.argv)>1:
  
    # check the arguments 
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:r:n:t:d:",["input_file=","rrlyr_path=","non-rrlyr_path=","templates_path","data_tables_path"])

    except getopt.GetoptError:
   
      print "python prepare_for_plotting_withCwithF.py -i <input file full path> -r <rrlyr full path> -n <non-rrlyr full path> -t <templates path> -d <data tables (true params) full path>"
      sys.exit(2)
  
    for opt, arg in opts:
      if opt == "-h":
        print "python prepare_for_plotting_withCwithF.py -i <input file full path> -r <rrlyr full path> -n <non-rrlyr full path> -t <templates path> -d <data tables (true params) full path>"
        sys.exit(2)
      elif opt in ("-i", "--input_file"):
        file_name = arg
      elif opt in ("-r", "--rrlyr_path"):
        rrlyr_path = arg
      elif opt in ("-n", "--non-rrlyr_path"):
        other_path = arg
      elif opt in ("-d", "--data_tables_path"):
        home_path = arg
      elif opt in ("-t", "--templates_path"):
        template_path = arg


    # use TeX typesetting for plots
    rc('text', usetex=False)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':12})
    rc('xtick.major', size='6')
    rc('xtick.minor', size='3')
    rc('ytick.major', size='6')
    rc('ytick.minor', size='3')
    rc('lines', linewidth=1)
    rc('axes', linewidth=1)

    # load light curve parameters of 
    true_params = np.genfromtxt(home_path + '/table2.dat', names='id, type, P, gA, g0, gE, gT, rA, r0, rE, rT, iA, i0, iE, iT, zA, z0, zE, zT', usecols=(0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), dtype='u4, |S10, f8, f8, f8, f8, u2, f8, f8, f8, u2, f8, f8, f8, u2, f8, f8, f8, u2')

    # select RRab star-templates
    ab = true_params['type'] == 'ab'
    ab = true_params[ab]
    # sort RRab star-templates by phi_08
    phi_08 = np.genfromtxt(template_path + '/objid_phi08_all.txt', names='id, phi_08g, phi_08r, phi_08i, phi_08z', dtype='u4, f8, f8, f8, f8')
    # match two lists using id
    m1, m2 = match(ab['id'], phi_08['id'])
    assert (m1.size == ab.size)
    ab = ab[m1]
    phi_08 = phi_08[m2]
    # sort in ascending order, starting with g band
    sI = np.argsort(phi_08, order=('phi_08g', 'phi_08r', 'phi_08i', 'phi_08z'))
    ab = ab[sI]

    # select 2 RRc star-templates
    c = (true_params['id'] == 1342595) | (true_params['id'] == 3031571)
    c = true_params[c]

    # merge RRab and RRc star-templates (RRc need to go at the end otherwise the code will break)
    true_params = np.concatenate((ab, c))

    phase_model = np.arange(500)*0.002
    template_list = np.concatenate((np.sort(glob(template_path + "/1??g.fits")), np.sort(glob(template_path + "/?g.fits"))))
    templates_g = load_templates(template_list)
    template_list = np.concatenate((np.sort(glob(template_path + "/1??r.fits")), np.sort(glob(template_path + "/?r.fits"))))
    templates_r = load_templates(template_list)
    template_list = np.concatenate((np.sort(glob(template_path + "/1??i.fits")), np.sort(glob(template_path + "/?i.fits"))))
    templates_i = load_templates(template_list)
    template_list = np.concatenate((np.sort(glob(template_path + "/1??z.fits")), np.sort(glob(template_path + "/?z.fits"))))
    templates_z = load_templates(template_list)


    # reading data from file which contains main table
    main_table = np.genfromtxt(file_name, names='Ra, Dec, RRLIndicator, RRLtype, P_true, P_best, chi2dof, N, tID, phi, r0, P_template, rExt, d, uF, gF, rF, iF, zF, ugmin, ugminErr, grmin, grminErr, file_number, F', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24), dtype='f8, f8, i4, |S10, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, |S30, f8', delimiter=",")

    for i in range(len(main_table)): 

      # checking if star is RRLyrae: 1 - yes, 0 - no
      if main_table["RRLIndicator"][i] == 1:   
        typ = main_table["RRLtype"][i]
        p_true = main_table["P_true"][i]
        data_path = rrlyr_path
      else:
        typ = "other"
        p_true = "N/A"
        data_path = other_path

      chi2_dof = main_table["chi2dof"][i]
      templateID = main_table["tID"][i]
      period = main_table["P_best"][i]
      phi = main_table["phi"][i]
      r0 = main_table["r0"][i]
      F = main_table["F"][i]
      p_template = main_table["P_template"][i]
    
      starID = main_table["file_number"][i]
      txt_f = data_path + "/ps1_-" + starID + "cleaned.txt"

      lc_data = load_PS1_data(txt_f)

      print starID
  
      plot_lcs(chi2_dof, period, phi, r0, F, templateID, typ, starID, p_true, p_template)

  else:
    
    print "Incorrect number of arguments! Correct usage:\n 'python prepare_for_plotting_withCwithF.py -i <input file full path> -r <rrlyr full path> -n <non-rrlyr full path> -t <templates path>'"
