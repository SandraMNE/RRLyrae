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
import pickle, os, cPickle
from dust import getval
from esutil.coords import eq2gal
from matplotlib import pyplot as plt
import os.path
import cPickle
from esutil.coords import sphdist
import sys, getopt
import gzip
from esutil.numpy_util import match


##############################################################################################################################################

def extract_data(pkl_fname):

    # extract all relevant data from processed pkl files
    f = gzip.open(pkl_fname,'rb')
#    f = open(pkl_fname,"r")
    res_all = cPickle.load(f)
    f.close()

    templateID = []
    chi2 = []
    period = []
    phi = []
    r = []
    dof = []
    F = []

    for res in res_all:
        templateID.append(res[0])
        chi2.append(res[1])  
        period.append(res[2])  
        phi.append(res[3][0])
        r.append(res[3][1])  
        F.append(res[3][2])  # currently not available
#        dof.append(res[4])

    period = np.array(period)
    templateID = np.array(templateID)
    chi2 = np.array(chi2)
    phi = np.array(phi)
    r = np.array(r)
    dof = np.array(dof)
    f = np.array(F)

    return period, templateID, chi2, phi, r, F

##############################################################################################################################################

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
    lc_data = [lc_ps['mjd_obs'][lc_ps['filter'] == 'g'], lc_ps['ucalmag'][lc_ps['filter'] == 'g'], lc_ps['mag_err'][lc_ps['filter'] == 'g'], lc_ps['mjd_obs'][lc_ps['filter'] == 'r'], lc_ps['ucalmag'][lc_ps['filter'] == 'r'], lc_ps['mag_err'][lc_ps['filter'] == 'r'], lc_ps['mjd_obs'][lc_ps['filter'] == 'i'], lc_ps['ucalmag'][lc_ps['filter'] == 'i'], lc_ps['mag_err'][lc_ps['filter'] == 'i'], lc_ps['mjd_obs'][lc_ps['filter'] == 'z'], lc_ps['ucalmag'][lc_ps['filter'] == 'z'], lc_ps['mag_err'][lc_ps['filter'] == 'z']]
    return lc_data

##############################################################################################################################################

#### main program #####

if __name__ == "__main__":  
  
  #suffix = "_withN"
  #data_other = "/homec/sfb881/sfb044/projects/rrlyr/PS_SANDRA/processed/lc_cleaned_other/*" + suffix + ".pkl"
  #data_rrlyr = "/homec/sfb881/sfb044/projects/rrlyr/PS_SANDRA/processed/lc_cleaned_rrlyr/*" + suffix + ".pkl"
  #file_name = "main_table.txt"
  #home_data = '/homec/sfb881/sfb044/papers/halo_sub'
  #

  suffix = None
  other_path = None
  rrlyr_path = None
  home_path = None
  file_name = None

  if len(sys.argv)>1:
  
    # check the arguments 
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hs:r:n:d:t:o:",["suffix=","rrlyr_path=","non-rrlyr_path=","data_tables_path=","templates_path","output_file="])

    except getopt.GetoptError:
   
      print "python prepare_for_plotting_withCwithF.py -s <pkl file suffix> -r <rrlyr full path> -n <non-rrlyr full path> -d <data tables (true params) full path> -t <templates path> -o <output file full path>"
      sys.exit(2)
  
    for opt, arg in opts:
      if opt == "-h":
        print "python prepare_for_plotting_withCwithF.py -s <pkl file suffix> -r <rrlyr full path> -n <non-rrlyr full path> -d <data tables (true params) full path> -t <templates path> -o <output file full path>"
        sys.exit(2)
      elif opt in ("-s", "--suffix"):
        suffix = arg
      elif opt in ("-r", "--rrlyr_path"):
        rrlyr_path = arg
      elif opt in ("-n", "--non-rrlyr_path"):
        other_path = arg
      elif opt in ("-d", "--data_tables_path"):
        home_path = arg
      elif opt in ("-t", "--templates_path"):
        template_path = arg
      elif opt in ("-o", "--output_file"):
        file_name = arg


    #making list of all pickle files to be processed
  
    data_rrlyr = rrlyr_path + "/*" + suffix
    fnames_r = np.sort(glob(data_rrlyr))
    
    if other_path != None:
      data_other = other_path + "/*" + suffix
      fnames_o = np.sort(glob(data_other))
      fnames = np.concatenate((fnames_r, fnames_o))
    else:
      fnames = fnames_r      


    names_other = []
    chi2_best_other = []
    period_best_other = []
    template_best_other = []
    dof_best_other = []
    phi_best_other = []
    r0_best_other = []
    F_best_other = []


    names_rrlyr = []
    chi2_best_rrlyr = []
    period_best_rrlyr = []
    template_best_rrlyr = []
    dof_best_rrlyr = []
    phi_best_rrlyr = []
    r0_best_rrlyr = []
    F_best_rrlyr = []


    table2 = np.genfromtxt('%s/table2.dat' % home_path, names='id, type, P, rA', usecols=(0, 1, 2, 11), dtype='u4, |S10, f8, f8')

    # load light curve parameters of SDSS S82 RR Lyrae stars
    true_params = np.genfromtxt('%s/table2.dat' % home_path, names='id, type, P, gA, g0, gE, gT, rA, r0, rE, rT, iA, i0, iE, iT, zA, z0, zE, zT', usecols=(0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), dtype='u4, |S10, f8, f8, f8, f8, u2, f8, f8, f8, u2, f8, f8, f8, u2, f8, f8, f8, u2')

    ab = true_params['type'] == 'ab'
    ab = true_params[ab]
    # sort RRab star-templates by phi_08
    phi_08 = np.genfromtxt('%s/objid_phi08_all.txt' % template_path, names='id, phi_08g, phi_08r, phi_08i, phi_08z', dtype='u4, f8, f8, f8, f8')
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

    # load from table3 ra & dec
    table3 = np.genfromtxt('%s/table3.dat' % home_path, usecols=(0,1,2,3,4,6,7,8,9,10,12,13,14,15), names='id, ra, dec, rExt, d, uF, gF, rF, iF, zF, ugmin, ugminErr, grmin, grminErr')
  
    f_tbl = open(file_name,"w")

    f_tbl.write("#Ra, Dec, RRLIndicator, RRLtype, P_true, P_best, chi2dof, N, tID, phi, r0, P_template, rExt, d, uF, gF, rF, iF, zF, 
ugmin, ugminErr, grmin, grminErr, file_number, F\n")


    c = 0

    # calculating values for each star
    for f in fnames:

      path = f[0:f.rfind("/")]

      txt_f = f.replace(suffix,".txt")

      RaDec = np.genfromtxt(txt_f, usecols=(2, 3), names='ra, dec', dtype='f8, f8')[0]
      N = sum(1 for line in open(txt_f, 'r'))
      print N
      ra = RaDec[0]
      dec = RaDec[1]
      starID = f[f.rfind("/")+1:].replace("ps1_-","").replace("cleaned","").replace(suffix,"")
  
      c = c + 1
      print "Currently processing", c, " out of ", len(fnames)

      # loading data from pkl file
      period, template, chi2, phi, r0, F = extract_data(f)

      # finding best of everything (for min chi2)
      p = period[np.argmin(chi2)]
      tID = template[np.argmin(chi2)]
      ch2 = chi2[np.argmin(chi2)]
      chi2dof = ch2/(N-4)
      phi_b = phi[np.argmin(chi2)]
      r0_b = r0[np.argmin(chi2)]
      F_b = F[np.argmin(chi2)]

      if path in data_rrlyr:
 
        # if star is RRLyrae
        RRLInd = 1
        P_template = true_params['P'][tID]
        ang_sep = sphdist(ra, dec, table3['ra'], table3['dec'])*3600
        ind = np.where(ang_sep < 1)[0]
        rExt = table3['rExt'][ind][0]
        d = table3['d'][ind][0]
        uF = table3['uF'][ind][0]
        gF = table3['gF'][ind][0]
        rF = table3['rF'][ind][0] 
        iF = table3['iF'][ind][0] 
        zF = table3['zF'][ind][0]
        ugmin = table3['ugmin'][ind][0]
        ugminErr = table3['ugminErr'][ind][0]
        grmin = table3['grmin'][ind][0]
        grminErr = table3['grminErr'][ind][0]    

        if ind != None:
          p_true = table2['P'][ind][0]
          typ = table2['type'][ind][0]

      else:

        RRLInd = 0
        P_template = true_params['P'][tID]
        p_true = -99999.99
        typ = None
        rExt = -99999.99
        d = -99999.99
        uF = -99999.99
        gF = -99999.99
        rF = -99999.99
        iF = -99999.99
        zF = -99999.99
        ugmin = -99999.99
        ugminErr = -99999.99
        grmin = -99999.99
        grminErr = -99999.99 

     
      
      f_tbl.write(str(ra)+","+str(dec)+","+str(RRLInd)+","+str(typ)+","+str(p_true)+","+str(p)+","+str(chi2dof)+","+str(N)+","+str(tID)+","+str(phi_b)+","+str(r0_b)+","+str(P_template)+","+str(rExt)+","+str(d)+","+str(uF)+","+str(gF)+","+str(rF)+","+str(iF)+","+str(zF)+","+str(ugmin)+","+str(ugminErr)+","+str(grmin)+","+str(grminErr)+","+str(starID)+","+str(F_b)+"\n")

    f_tbl.close()  

  else:

    print "Incorrect number of arguments! Correct usage:\n 'python prepare_for_plotting_withCwithF.py -r <rrlyr full path> -n <non-rrlyr full path> -d <data tables (true params) full > -t <templates path> -i <input file full path>'"
