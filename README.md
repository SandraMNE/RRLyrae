RRLyrae
=======

File hybrid_fitting_fast.py is slight modification of earlier code, taking every second template and twice larger step in 1st iteration. No C templates or fitting of amplitude is done. This code is adapted for running on cluster, although with small modifications, it can be run on standalone machine. Output of this phase are pickle files, containing fitted period, template, chi2, dof.

File prepare_for_plotting.py generates main table based on pickle files. Main table is comma separated text file. This script takes several parameters, which can be seen with option -h.

File plot.py plots seven different figures based on main table. For executing, it is sufficient to pass main table file path with option -i (again -h can be used for help).


