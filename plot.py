#!/usr/bin/env python
import numpy as np
from pyfits import getdata
from glob import glob
import numpy as np
from multiprocessing import Pool
from scipy.optimize import leastsq
from numpy.lib import recfunctions
#from useful import extinction_coeff
#import esutil
from matplotlib import rc
from matplotlib.backends.backend_pdf import FigureCanvasPdf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPdf)
import pickle, os
#from dust import getval
#from esutil.coords import eq2gal
from matplotlib import pyplot as plt
import os.path
import math
import sys, getopt

###########################################################################################################################################

def graph1(chi2_rrlyr, chi2_other):

  # Plots histogram of log(chi_dof) per RRLyrae & Others (non-RRLyrae) 
  
  plt.hist(chi2_rrlyr, 30, color = 'b', label='rrlyr', normed=True, alpha=0.5, histtype = 'step')
  plt.hist(chi2_other, 30, color = 'r', label='other', normed=True, alpha=0.5, histtype = 'step')

  plt.title("Best chi dof for rrlyrae & others")
  plt.xlim(-1, 2)
  plt.xlabel("log(chi_dof)")
  plt.legend(loc='upper right')
  plt.show()
  plt.savefig("./graph1.png")

###########################################################################################################################################

def graph2(period_rrlyr, period_other):

  # Plots histogram of P_best (calculated) per RRLyrae & Others (non-RRLyrae) 
  
  plt.hist(period_rrlyr, 30, color = 'b', label='rrlyr', normed=True, alpha=0.5, histtype = 'step')
  plt.hist(period_other, 30, color = 'r', label='other', normed=True, alpha=0.5, histtype = 'step')
  plt.title("Best period for rrlyrae & others")
  plt.xlim(0.3, 0.9)
  plt.xlabel("period_best")
  plt.legend(loc='upper right')
  plt.show()
  plt.savefig("./graph2.png")

###########################################################################################################################################

def graph3(period_rrlyr, period_template_rrlyr, period_other, period_template_other):

  # Plots histogram of log(P_best/P_template) per RRLyrae & Others (non-RRLyrae) 

  p_rrlyr = period_rrlyr / period_template_rrlyr
  p_rrlyr_f = np.array([math.log10(x) for x in p_rrlyr])

  p_other = period_other / period_template_other
  p_other_f = np.array([math.log10(x) for x in p_other])

  plt.hist(p_rrlyr_f, 30, color = 'b', label='rrlyr', normed=True, alpha=0.5, histtype = 'step')
  plt.hist(p_other_f, 30, color = 'r', label='other', normed=True, alpha=0.5, histtype = 'step')
  plt.title("Period diff (best/template) for rrlyrae & others")
  plt.xlim(-0.15, 0.1)
  plt.xlabel("log(P_best/P_template)")
  plt.legend(loc='upper right')
  plt.show()
  plt.savefig("./graph3.png")

###########################################################################################################################################

def graph4(period_rrlyr, period_template_rrlyr, period_other, period_template_other, chi2_rrlyr, chi2_other):

  # Plots scatter plot of log(chi_dof) vs. log(P_best/P_template) per RRLyrae & Others (non-RRLyrae) 

  p_rrlyr = period_rrlyr / period_template_rrlyr
  p_rrlyr_f = np.array([math.log10(x) for x in p_rrlyr])

  p_other = period_other / period_template_other
  p_other_f = np.array([math.log10(x) for x in p_other])

  plt.scatter(chi2_rrlyr, p_rrlyr_f, s=10, c='b', marker="+", label='rrlyr')
  plt.scatter(chi2_other, p_other_f, s=10, c='r', marker="x", label='other')

  plt.title("Chi2_dof vs. Period diff (best/template) for rrlyrae & others")
  plt.xlim(-0.8, 2.9)
  plt.xlabel("log(chi2_dof)")
  plt.ylabel("log(P_best/P_template)")
  plt.legend(loc='upper right')
  plt.show()
  plt.savefig("./graph4.png")

###########################################################################################################################################

def graph5(period_rrlyr, period_true_rrlyr):

  # Plots scatter plot of P_true vs. log(P_best/P_true) per RRLyrae

  p_rrlyr = period_rrlyr / period_true_rrlyr
  p_rrlyr_f = np.array([math.log10(x) for x in p_rrlyr])

  plt.scatter(period_true_rrlyr, p_rrlyr_f, s=10, c='b', marker="+", label='rrlyr')

  plt.title("True period vs. Period diff (best/true) for rrlyrae ONLY")
  plt.xlim(0.15, 1)
  plt.xlabel("True period")
  plt.ylabel("log(P_best/P_true)")
  plt.legend(loc='upper right')
  plt.show()
  plt.savefig("./graph5.png")

###########################################################################################################################################

def graph_ROC(chi2_rrlyr_log, chi2_other_log, dof_rrlyr, dof_other, chi2_rrlyr, chi2_other):

  # Plots ROC for log(chi_dof) and significance

  chi2_range = np.sort(chi2_rrlyr)
  eff = []
  comp = []
  eff_sign = []
  eff_sign2 = []
  comp_sign = []
  comp_sign2 = []
  control_eff = []
  control_comp = []

  sigma_rrlyr = np.array([math.sqrt(float(2)/(x-4)) for x in dof_rrlyr])
  signif_rrlyr = (chi2_rrlyr - 1) / sigma_rrlyr

  sigma_other = np.array([math.sqrt(float(2)/(x-4)) for x in dof_other])
  signif_other = (chi2_other - 1) / sigma_other

  for T in np.arange(-1.2, 4.2, 0.1):  #chi2_range:
    num_rrlyr_sel = len(np.where(chi2_rrlyr_log<T)[0])
    num_other_sel = len(np.where(chi2_other_log<T)[0])

    total_rrlyr = len(chi2_rrlyr_log)
    if num_rrlyr_sel+num_other_sel != 0:
      eff_value = float(num_rrlyr_sel)/(num_rrlyr_sel+num_other_sel)
      eff.append(eff_value)
      comp_value = float(num_rrlyr_sel)/total_rrlyr
      comp.append(comp_value)

  for T in np.arange(-5, 100, 0.1):  
    num_rrlyr_sel_sign = len(np.where(signif_rrlyr<T)[0])
    num_other_sel_sign = len(np.where(signif_other<T)[0])

    total_rrlyr = len(chi2_rrlyr)
 
    if num_rrlyr_sel_sign+num_other_sel_sign != 0:
      eff_value_sign = float(num_rrlyr_sel_sign)/(num_rrlyr_sel_sign+num_other_sel_sign)
      eff_sign.append(eff_value_sign)
      comp_value_sign = float(num_rrlyr_sel_sign)/total_rrlyr
      comp_sign.append(comp_value_sign)
 
  for T in [0.0, 0.3, 0.7, 1.0]:
    num_rrlyr_sel = len(np.where(chi2_rrlyr_log<T)[0])
    num_other_sel = len(np.where(chi2_other_log<T)[0])
    total_rrlyr = len(chi2_rrlyr_log)
    if num_rrlyr_sel+num_other_sel != 0:
      eff_value = float(num_rrlyr_sel)/(num_rrlyr_sel+num_other_sel)
      comp_value = float(num_rrlyr_sel)/total_rrlyr
      control_eff.append(eff_value)
      control_comp.append(comp_value)
    
  comp = np.array(comp)
  eff = np.array(eff)
  control_eff = np.array(control_eff)
  control_comp = np.array(control_comp)
  plt.plot(comp, eff, lw=1, color='b')
  plt.plot(comp_sign, eff_sign, color = 'g')
#  plt.scatter(control_comp, control_eff, color='red', marker='o', s=50)

#  print control_eff, control_comp

  plt.xlabel("Completeness")
  plt.ylabel("Efficiency")
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.show()
  plt.savefig("./graph6-ROC.png")

###########################################################################################################################################

def graph7(chi2_rrlyr, chi2_other, dof_rrlyr, dof_other):

  # Plots histogram of significance per RRLyrae & Others (non-RRLyrae) 

  sigma_rrlyr = np.array([math.sqrt(float(2)/(x-4)) for x in dof_rrlyr])
  signif_rrlyr = (chi2_rrlyr - 1) / sigma_rrlyr

  sigma_other = np.array([math.sqrt(float(2)/(x-4)) for x in dof_other])
  signif_other = (chi2_other - 1) / sigma_other

  bins = np.linspace(-10, 100, 55)

  plt.hist(signif_rrlyr, bins, color = 'b', label='rrlyr', normed=True, histtype = 'step')
  plt.hist(signif_other, bins, color = 'r', label='other', normed=True, histtype = 'step')

  plt.title("Significance for rrlyrae & others")
  plt.xlim(-10, 100)
  plt.xlabel("significance")
  plt.legend(loc='upper right')
  plt.show()
  plt.savefig("./graph7.png")  

###########################################################################################################################################

def extract_data(fname):

  # extracts all necessary data for plotting from main table (i.e. file containing it)

  chi2_rrlyr = []
  t_rrlyr = []
  p_rrlyr = []
  dof_rrlyr = []
  p_true_rrlyr = []
  p_template_rrlyr = []


  chi2_other = []
  t_other = []
  p_other = []
  dof_other = []
  p_true_other = []
  p_template_other = []

 
  # reading data from file which contains main table
  main_table = np.genfromtxt(fname, names='Ra, Dec, RRLIndicator, RRLtype, P_true, P_best, chi2dof, N, tID, phi, r0, P_template, rExt, d, uF, gF, rF, iF, zF, ugmin, ugminErr, grmin, grminErr, file # ', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), dtype='f8, f8, i4, |S10, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8', delimiter=",")

  for i in range(len(main_table)): 

    # checking if star is RRLyrae: 1 - yes, 0 - no
    if main_table["RRLIndicator"][i] == 1:   

      chi2_rrlyr.append(main_table["chi2dof"][i])
      t_rrlyr.append(main_table["tID"][i])
      p_rrlyr.append(main_table["P_best"][i])
      dof_rrlyr.append(main_table["N"][i])
      p_true_rrlyr.append(main_table["P_true"][i])
      p_template_rrlyr.append(main_table["P_template"][i])
    
    elif main_table["RRLIndicator"][i] == 0:

      chi2_other.append(main_table["chi2dof"][i])
      t_other.append(main_table["tID"][i])
      p_other.append(main_table["P_best"][i])
      dof_other.append(main_table["N"][i])
      p_true_other.append(main_table["P_true"][i])
      p_template_other.append(main_table["P_template"][i])
      
  return np.array(chi2_rrlyr), np.array(t_rrlyr), np.array(p_rrlyr), np.array(dof_rrlyr), np.array(p_true_rrlyr), np.array(p_template_rrlyr), np.array(chi2_other), np.array(t_other), np.array(p_other), np.array(dof_other), np.array(p_true_other), np.array(p_template_other)

###########################################################################################################################################

#### main program #####

if __name__ == "__main__":   

#  data_path = "/homec/sfb881/sfb044/projects/rrlyr/PS_SANDRA/"
#  file_name = "main_table.txt"
#  data = data_path + file_name


  # checking if input file is supplied as argument
  if len(sys.argv) > 1:

    try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:",["ifile="])

    except getopt.GetoptError:
      print "python plot.py -i <inputfile>"
      sys.exit(2)

    for opt, arg in opts:
      if opt == "-h":
        print "python plot.py -i <inputfile>"
        sys.exit(2)
      elif opt in ("-i", "--ifile"):
        data = arg

    # these flags are currently useless, but might be useful if C templates are added or F is fitted too
    Cflag = 0
    Fflag = 0
  
    if Cflag == 0 and Fflag == 0: 

      # retrieve data
      chi2_dof_rrlyr, template_rrlyr, period_rrlyr, dof_rrlyr, period_true_rrlyr, period_template_rrlyr, chi2_dof_other, template_other, period_other, dof_other, period_true_other, period_template_other = extract_data(data)


    # calculate log(chi2) for plotting
    chi2_dof_rrlyr_log = np.array([math.log10(x) for x in chi2_dof_rrlyr])
    chi2_dof_other_log = np.array([math.log10(x) for x in chi2_dof_other])

    # plot

    graph1(chi2_dof_rrlyr_log, chi2_dof_other_log)
    graph2(period_rrlyr, period_other)
    graph3(period_rrlyr, period_template_rrlyr, period_other, period_template_other)
    graph4(period_rrlyr, period_template_rrlyr, period_other, period_template_other,chi2_dof_rrlyr_log, chi2_dof_other_log)
    graph5(period_rrlyr, period_true_rrlyr)
    graph_ROC(chi2_dof_rrlyr_log, chi2_dof_other_log, dof_rrlyr, dof_other, chi2_dof_rrlyr, chi2_dof_other)
    graph7(chi2_dof_rrlyr, chi2_dof_other, dof_rrlyr, dof_other)

  else:

    print "Missing argument! Correct usage: python plot.py -i <inputfile full path>"

