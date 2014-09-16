#!/usr/bin/env python
import numpy as np
from pyfits import getdata
from glob import glob
import numpy as np
from multiprocessing import Pool
from scipy.optimize import leastsq
from numpy.lib import recfunctions
import pickle
import cPickle
import sys, os, commands
from esutil.coords import eq2gal
from dust import getval

home_dir = "/homec/sfb881/sfb044"
projects_dir = "/homec/sfb881/sfb044"

# load the list of filenames to be processed
lightcurves = np.loadtxt(sys.argv[1], dtype='|S200')
n_proc = lightcurves.size

# load light curve parameters of SDSS S82 RR Lyrae stars
true_params = np.genfromtxt('%s/papers/halo_sub/table2.dat' % home_dir, names='id, type, P, gA, g0, gE, gT, rA, r0, rE, rT, iA, i0, iE, iT, zA, z0, zE, zT', usecols=(0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), dtype='u4, |S10, f8, f8, f8, f8, u2, f8, f8, f8, u2, f8, f8, f8, u2, f8, f8, f8, u2')

# select only RRab stars
ab = true_params['type'] == 'ab'
# select 2 type c stars
#c = (true_params['id'] == 1342595) | (true_params['id'] == 3031571)
#true_params = true_params[ab | c]
true_params = true_params[ab]

# load all templates into an array
def load_templates(template_list):
    templates = np.zeros((500, template_list.size))
    for i, fname in enumerate(template_list):
        aux = getdata(fname)
        templates[:, i] = aux
    return templates

phase_model = np.arange(500)*0.002
template_list = np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/1??g.fits" % projects_dir))
#template_list = np.concatenate((template_list, np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/?g.fits" % projects_dir))))
templates_g = load_templates(template_list)
template_list = np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/1??r.fits" % projects_dir))
#template_list = np.concatenate((template_list, np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/?r.fits" % projects_dir))))
templates_r = load_templates(template_list)
template_list = np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/1??i.fits" % projects_dir))
#template_list = np.concatenate((template_list, np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/?i.fits" % projects_dir))))
templates_i = load_templates(template_list)
template_list = np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/1??z.fits" % projects_dir))
#template_list = np.concatenate((template_list, np.sort(glob("%s/projects/rrlyr/final_templates/bspline_templates/?z.fits" % projects_dir))))
templates_z = load_templates(template_list)

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
    # add templates
    if true_params['gT'][ind] > 99:
        template_g = templates_g[:, true_params['gT'][ind] - 100]
        template_r = templates_r[:, true_params['rT'][ind] - 100]
        template_i = templates_i[:, true_params['iT'][ind] - 100]
        template_z = templates_z[:, true_params['zT'][ind] - 100]
#    else:
#        sys.exit()
#        template_g = templates_g[:, -2 + true_params['gT'][ind]]
#        template_r = templates_r[:, -2 + true_params['rT'][ind]]
#        template_i = templates_i[:, -2 + true_params['iT'][ind]]
#        template_z = templates_z[:, -2 + true_params['zT'][ind]]
    return period, gA, rA, iA, zA, gr0, ri0, rz0, template_g, template_r, template_i, template_z

def errfunc(theta, period, lc_data, model_data):
    mjdg, gLC, gErr, mjdr, rLC, rErr, mjdi, iLC, iErr, mjdz, zLC, zErr = lc_data
    phi, r0 = theta
#    F = 1
    period_model, gA_model, rA_model, iA_model, zA_model, gr0_model, ri0_model, rz0_model, template_g, template_r, template_i, template_z = model_data

#    if not (-period < phi < period and np.min(rLC) - 0.5 < r0 < np.min(rLC) + 0.5 and F > 0 and F < 2):
    if not (-period < phi < period and np.min(rLC) - 0.5 < r0 < np.min(rLC) + 0.5):
      return np.inf + gLC

    # calculate model
    model_g = gA_model*template_g + gr0_model + r0
    model_r = rA_model*template_r + r0
    model_i = iA_model*template_i + r0 - ri0_model
    model_z = zA_model*template_z + r0 - rz0_model

    phase_g = (mjdg - phi)/period - np.floor((mjdg - phi)/period)
    phase_r = (mjdr - phi)/period - np.floor((mjdr - phi)/period)
    phase_i = (mjdi - phi)/period - np.floor((mjdi - phi)/period)
    phase_z = (mjdz - phi)/period - np.floor((mjdz - phi)/period)

    # interpolate observed phases onto the phase grid and calculate residuals between the observations and the model
    res_g = gLC - np.interp(phase_g, phase_model, model_g)
    res_r = rLC - np.interp(phase_r, phase_model, model_r)
    res_i = iLC - np.interp(phase_i, phase_model, model_i)
    res_z = zLC - np.interp(phase_z, phase_model, model_z)
    res = np.concatenate((res_g/gErr, res_r/rErr))
    res = np.concatenate((res, res_i/iErr))
    res = np.concatenate((res, res_z/zErr))
    return res

def probe_periods(ind, model_data, lc_data):
    # for a given template, returns the best-fit period, phi0, and r0
    chi2_all = []
    params_all = []
    period_all = []
    N_all = []
    mjdg, gLC, gErr, mjdr, rLC, rErr, mjdi, iLC, iErr, mjdz, zLC, zErr = lc_data
    #model_data = model(ind)
    period_model = model_data[0]
    period_grid = np.arange(period_model-0.1, period_model+0.1, 0.00002)
    for period in period_grid:
        phase_g = mjdg/period - np.floor(mjdg/period)
        phase_r = mjdr/period - np.floor(mjdr/period)
        phase_i = mjdi/period - np.floor(mjdi/period)
        phase_z = mjdz/period - np.floor(mjdz/period)
        peaks = np.array([gLC[np.argmin(gLC)], rLC[np.argmin(rLC)], iLC[np.argmin(iLC)], zLC[np.argmin(zLC)]])
        phases = np.array([phase_g[np.argmin(gLC)], phase_r[np.argmin(rLC)], phase_i[np.argmin(iLC)], phase_z[np.argmin(zLC)]])
        phi0 = phases[np.argmin(peaks)]
        if phi0 > 0.5:
            phi0 = (phi0-1.0)*period
        else:
            phi0 = phi0*period
        pinit = np.array([phi0, np.min(rLC)], dtype='f8')
        out = leastsq(errfunc, pinit, args=(period, lc_data, model_data), full_output=1)
        pfinal = out[0]
        covar = out[1]
        chi = errfunc(pfinal, period, lc_data, model_data)
        N = chi.shape[0]
        chi2 = np.sum(chi**2)
        chi2_all.append(chi2)
        period_all.append(period)
        #params_all.append(pfinal)
        N_all.append(N)

    chi2_all = np.array(chi2_all)
    period_all = np.array(period_all)
    chi2_min_ind = np.argmin(chi2_all)
    chi2_min = chi2_all[chi2_min_ind]
    period_best = period_all[chi2_min_ind]
    N_best = N_all[chi2_min_ind]

#    good = np.isfinite(chi2_all)
#    period_grid = period_grid[good]
#    chi2_all = chi2_all[good]
#    params_all = params_all[good]
#    best = np.argmin(chi2_all)

    return chi2_min, period_best, N_best


def calc_chi2_all_periods(ind, model_data, lc_data, period_best, res_all):
    # for a given template, returns the best-fit period, phi0, and r0
    chi2_all = []
    params_all = []
    period_all = []
    mjdg, gLC, gErr, mjdr, rLC, rErr, mjdi, iLC, iErr, mjdz, zLC, zErr = lc_data
#    period_model = model_data[0]
    period_grid = np.arange(period_best*(1-0.15/100), period_best*(1+0.15/100), 0.00001)
#    period_grid = np.arange(period_best-0.1, period_best+0.1, 0.00001)
    for period in period_grid:
        phase_g = mjdg/period - np.floor(mjdg/period)
        phase_r = mjdr/period - np.floor(mjdr/period)
        phase_i = mjdi/period - np.floor(mjdi/period)
        phase_z = mjdz/period - np.floor(mjdz/period)
        peaks = np.array([gLC[np.argmin(gLC)], rLC[np.argmin(rLC)], iLC[np.argmin(iLC)], zLC[np.argmin(zLC)]])
        phases = np.array([phase_g[np.argmin(gLC)], phase_r[np.argmin(rLC)], phase_i[np.argmin(iLC)], phase_z[np.argmin(zLC)]])
        phi0 = phases[np.argmin(peaks)]
        if phi0 > 0.5:
            phi0 = (phi0-1.0)*period
        else:
            phi0 = phi0*period
        pinit = np.array([phi0, np.min(rLC)], dtype='f8')
        out = leastsq(errfunc, pinit, args=(period, lc_data, model_data), full_output=1)
        pfinal = out[0]
        covar = out[1]
        chi = errfunc(pfinal, period, lc_data, model_data)
        N = chi.shape[0]
        chi2 = np.sum(chi**2)
        pfinal = np.array(pfinal)
#        print ind, period, chi2, pfinal #, phase_g, phase_r, phase_i, phase_z
        res_all.append((ind, chi2, period, pfinal, N))
        chi2_all.append(chi2)
        params_all.append(pfinal)
        period_all.append(period)

    chi2_all = np.array(chi2_all)
    params_all = np.array(params_all)
    period_all = np.array(period_all)

#    good = np.isfinite(chi2_all)
#    period_grid = period_grid[good]
#    chi2_all = chi2_all[good]
#    params_all = params_all[good]
#    best = np.argmin(chi2_all)
     
#    print ind, np.min(chi2_all), period_all[np.argmin(chi2_all)]

    return res_all #ind, chi2_all[best], period_grid[best], params_all[best]



def process_lightcurve(lc_filename):
    name = os.path.splitext(lc_filename)[0]
    # load the light curve
    lc_data = load_PS1_data(lc_filename)
    # try each template
    res_all = []
    best_params = []
    for ind in np.arange(true_params.size)[::2]:
        model_data = model(ind)
#        period_best = model_data[0]
#        period_best = 0.46927009
#        period_best = 0.46926924602910919          
#        print ind, period_model, period_best, "diff", (period_model-period_best)*24*3600

        chi2_best, period_best, N_best = probe_periods(ind, model_data, lc_data)
        best_params.append((ind, chi2_best, period_best, N_best))
        res_all = calc_chi2_all_periods(ind, model_data, lc_data, period_best, res_all)
        res_all = calc_chi2_all_periods(ind+1, model_data, lc_data, period_best, res_all)
    # dump best-fit parameters for all tested templates
    f = open('%s_withN.pkl' % name, 'w')
    pickle.dump(res_all, f)
    f.close()
    f_best = open('%s_just_best.pkl' % name, 'w')
    pickle.dump(best_params, f_best)
    f_best.close()
    return lc_filename



#res_final = process_lightcurve(sys.argv[1])

# start workers
pool = Pool(processes=n_proc)

it = pool.imap_unordered(process_lightcurve, lightcurves)
for res in it:
    print res

# terminate workers
pool.terminate()



