#!/bin/env python


import os
# from compile import *
import time

import numpy as np
import pandas as pd
import csv
import scipy.stats as st

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import linregress

plt.switch_backend('agg')

bar_width = 1
error_width = 2

pos = 0

compiled2=pd.read_csv('results/btw_item_corr.csv')
compiled=pd.read_csv('results/targ_comp_corr.csv')

compiled['difference'] = compiled['targ_corr'] - compiled['comp_corr']
compiled['targ_diff'] = compiled['targ_corr'] - compiled[['base1corr', 'base2corr']].mean(axis = 1)
compiled['comp_diff'] = compiled['comp_corr'] - compiled[['base1corr', 'base2corr']].mean(axis = 1)
compiled['other'] = compiled[['base1corr', 'base2corr']].mean(axis = 1)



# A few possible ROI lists
pooled = ['Occ', 'VT', 'MTL']
in_order = ['V1', 'LOC', 'IT', 'Fus', 'PHC', 'PRC', 'EC', 'HC']
roi_list = ['V1', 'LOC', 'IT', 'Fus', 'PHC']
hipp_list = ['EC', 'PRC', 'HC']

# Added V2
output_list = ['V1', 'V2', 'LOC', 'IT', 'Fus', 'PHC', 'PRC', 'EC', 'HC']

# Lists for dictionaries
all_rois = ['V1', 'LOC_FS', 'IT_FS', 'fusiform_FS', 'parahippo_FS', 'V2', 'mOFC_FS', 'ento_FS', 'PRC_FS', 'hipp_FS', ['V1', 'LOC_FS'], ['IT_FS', 'fusiform_FS', 'parahippo_FS'], ['ento_FS', 'PRC_FS', 'hipp_FS']]

roi_name = ['V1','LOC','IT','Fus', 'PHC', 'V2', 'mOFC', 'EC', 'PRC', 'HC', 'Occ', 'VT', 'MTL']

color_list = [(0.03921569, 0.10980392, 0.52156863), (0.21568627, 0.45490196, 0.63529412), (0.22745098, 0.62352941, 0.60784314), (0.71764706, 0.83137255, 0.61176471), (0.83921569, 0.83529412, 0.45882353), (0.28627451, 0.03921569, 0.23921569), (0.74117647, 0.08235294, 0.31372549), (0.41176471, 0.82352941, 0.90588235), (0.87843137, 0.89411765, 0.8), (0.98039216, 0.41176471, 0), (0.0627451, 0.0745098, 0.2), (0.44705882, 0, 0.22352941), (0.94117647, 0.70980392, 0.34117647)]

roi_dict = dict(zip(roi_name, all_rois))
color_dict = dict(zip(roi_name, color_list))


# Increments a global counter
def increment_pos():
	global pos
	pos += 1



def create_subplot_dict(input_list):
	subplot_dict = dict(zip(input_list, range(1,(len(input_list) +1))))
	return subplot_dict

def compute_mean(input, measure):
	input['mean'] = input[measure].mean()
	return input


def subj_mean(input, measure):
	input['subj_mean'] = input[measure].mean()
	return input



# Next two filter down the data for a particular ROI
def output_data(roi,compiled, measure):
	if type(roi) is list:
		compiled_data = compiled[compiled['roi'].isin(roi)]
	else:
		compiled_data = compiled[compiled['roi'].isin([roi])]
	compiled_data = compiled_data.groupby(['subject','run']).apply(compute_mean, measure)
	compiled_data = compiled_data.groupby(['subject']).apply(subj_mean, measure)
	g_m = compiled_data[measure].mean()
	compiled_data['grand_mean'] = g_m
	compiled_data['corrected'] = compiled_data[measure] - compiled_data['subj_mean'] + compiled_data['grand_mean']
	grouped = compiled_data.groupby(['run'])
	grouped2 = compiled_data.groupby(['run', 'time_point'])
	means = grouped2.aggregate(np.mean)
	count = len(compiled_data.subject.unique())
	sterr = []
	for num in range(1,5):
		runonly = compiled_data[compiled_data['run'] == num].groupby('subject').agg(np.mean)
		se = runonly['corrected'].std()/np.sqrt(count)
		sterr.append(se)
	coarsemeans = grouped.aggregate(np.mean)
	xcoord = np.arange(1,5)
	return means, coarsemeans, sterr, xcoord




def output_sterr(run, roi, compiled, measure):
	if type(roi) is list:
		compiled_data = compiled[compiled['roi'].isin(roi)]
	else:
		compiled_data = compiled[compiled['roi'].isin([roi])]
	compiled_data = compiled_data.groupby(['subject','run','time_point']).apply(compute_mean, measure)
	compiled_data = compiled_data.groupby(['subject']).apply(subj_mean, measure)
	g_m = compiled_data[measure].mean()
	compiled_data['grand_mean'] = g_m
	compiled_data['corrected'] = compiled_data[measure] - compiled_data['subj_mean'] + compiled_data['grand_mean']
	count = len(compiled_data.subject.unique())
	runonly = compiled_data[compiled_data['run'] == run]
	timesterr = []
	for TR in range(1,24):
		TRonly = runonly[runonly['time_point'] == TR].groupby('subject').agg(np.mean)
		se = TRonly['corrected'].std()/np.sqrt(count)
		timesterr.append(se)
	return timesterr



# Plots difference between target and competitor over time (or really any single line)
def plot_time(input_list, compiled, measure, yaxtitle, figtitle, yax, smoothing, errorbars):
	nums = range(1,24)
	for named_roi in input_list:
		roi = roi_dict[str(named_roi)]
		plt.figure(figsize=(8,6))
		rr,gg,bb = color_dict[str(named_roi)] 
		means, coarsemeans, sterr, xcoord = output_data(roi,compiled,measure)
		for run in range(1,5):
			subplot_num = 140 + run
			plt.subplot(subplot_num, axisbg= (rr,gg,bb,1))
			runonly=means.loc[run]
			series=runonly[[measure]]
			if smoothing != 0:
				series = pd.rolling_mean(series, window=smoothing, min_periods=1, center = True)
			plt.plot(nums, series[measure],label=str(run), color = 'k', linewidth = 4)
			if errorbars == 'error_bars':
				errors = output_sterr(run, roi, compiled, measure)
				plt.fill_between(nums, series[measure]-errors, series[measure]+errors, facecolor = (0.25098,0.25098,0.25098), alpha = 0.3, linestyle='-', edgecolor = 'k', linewidth = 2, interpolate = True)
			plt.xlim(1,23)
			plt.ylim(yax)
			plt.axhline(y=0,color='k',linewidth=0.5)
			if run == 1:
				plt.ylabel(yaxtitle)
			else:
				plt.yticks([])
			plt.tight_layout()
			plt.xlabel('Run ' + str(run))
		plt.savefig('plots/' + str(named_roi) + '_' + str(figtitle) + '.png')
		plt.clf()

# Plots difference between target and competitor by tun
def plot_run(input_list,compiled, measure, yaxtitle, figtitle, yax, line_width, style):
	subplot_dict = create_subplot_dict(input_list)
	width = len(input_list) * 2.5
	plt.figure(figsize=(width,6))
	for named_roi in input_list:
		roi = roi_dict[str(named_roi)]
		means, coarsemeans, sterr, xcoord = output_data(roi, compiled, measure)
		series=coarsemeans[measure]
		errors=sterr
		subplot_num = 100 + len(input_list*10) + (subplot_dict[named_roi])
		rr,gg,bb = color_dict[str(named_roi)] 
		plt.subplot(subplot_num)
		if style == 'white':
			borderline = 'w'
		else:
			borderline = (rr,gg,bb,1)
		plt.bar(xcoord, series, bar_width, color = (rr,gg,bb,0.7), yerr=errors, error_kw = {'ecolor': (rr,gg,bb,1), 'linewidth': error_width}, edgecolor = borderline, linewidth = line_width)
		plt.xticks(xcoord + bar_width / 2, ('1', '2', '3', '4'))
		plt.axhline(y=0,color='k',linewidth=0.5)
		plt.xlabel(str(named_roi))
		plt.xlim(0.5,5.5)
		plt.ylim(yax)
		if subplot_num % 10 == 1:
			plt.ylabel(yaxtitle)
		else:
			plt.yticks([])
		plt.tight_layout()
	plt.savefig('plots/' + str(figtitle) + '.png')
	plt.clf()

# Computes spearman, pearson and slope for each subject by run
def compute_run_corr(input_list,compiled, measure):
	correlations = pd.DataFrame(np.empty([len(compiled.subject.unique()) * len(input_list),5]))
	correlations.columns = ['subject','roi','pearson', 'slope', 'spearman']
	for named_roi in input_list:
		roi = roi_dict[str(named_roi)]
		if type(roi) is list:
			compiled_data = compiled[compiled['roi'].isin(roi)]
		else:
			compiled_data = compiled[compiled['roi'].isin([roi])]
		for sub in compiled_data.subject.unique():
			subject_data = compiled_data[compiled_data['subject'] == sub]
			subject_data = subject_data.groupby(['run']).apply(compute_mean, measure)
			grouped = subject_data.groupby(['run'])
			means = grouped.aggregate(np.mean)
			runnums = [1, 2, 3, 4]
			differences = means[['difference']]
			spear = pd.DataFrame(data={'runnum': runnums, 'diffs': differences['difference']})
			spearman = spear.corr(method='spearman').loc['runnum','diffs']
			slope,ww,xx,yy,zz = linregress(runnums, differences['difference'])
			corr = np.corrcoef(runnums, differences['difference'])[1,0]
			correlations.iloc[pos,0] = sub
			correlations.iloc[pos,1] = named_roi
			correlations.iloc[pos,2] = corr
			correlations.iloc[pos,3] = slope
			correlations.iloc[pos,4] = spearman
			increment_pos()
	correlations.to_csv('results/target_vs_competitor_similarity_trend_across_runs_by_roi.csv')
	


# Plots two lines over time (e.g. targ_corr and comp_corr)
def plot_two_time(input_list, compiled, measure1, measure2, yaxtitle, figtitle, yax, smoothing, errorbars):
	nums = range(1,24)
	for named_roi in input_list:
		roi = roi_dict[str(named_roi)]
		plt.figure(figsize=(8,6))
		rr,gg,bb = color_dict[str(named_roi)] 
		for run in range(1,5):
			measures = [measure1, measure2]
			linestyles = ['-','--']
			markers = ['.','D']
			mark_opac = [1,0.8]
			mark_size = [12,6]
			targ_comp = zip(measures,linestyles, markers, mark_opac, mark_size)
			for cond, linetype, mark, opac, m_s, in targ_comp:
				means, coarsemeans, sterr, xcoord = output_data(roi,compiled,cond)
				subplot_num = 140 + run
				plt.subplot(subplot_num, axisbg= (rr,gg,bb,1))
				runonly=means.loc[run]
				series=runonly[[cond]]
				if smoothing != 0:
					series = pd.rolling_mean(series, window=smoothing, min_periods=1, center = True)
				plt.plot(nums, series[cond],label=str(run), color = (0,0,0,opac), linewidth = 3, linestyle = linetype)
				if errorbars == 'error_bars':
					errors = output_sterr(run, roi, compiled, cond)
					plt.fill_between(nums, series[cond]-errors, series[cond]+errors, facecolor = (0.25098,0.25098,0.25098), alpha = 0.3, linestyle=linetype, edgecolor = 'k', linewidth = 2, interpolate = True)
			plt.xlim(1,23)
			plt.ylim(yax)
			plt.axhline(y=0,color='k',linewidth=0.5)
			if run == 1:
				plt.ylabel(yaxtitle)
				legend_targ = mlines.Line2D([], [], color='k', linestyle='-', label='Target')
				legend_comp = mlines.Line2D([], [], color='k', linestyle='--', label='Competitor')
				plt.legend(handles=[legend_targ, legend_comp], fontsize=9, handlelength = 2, loc = 2)
			else:
				plt.yticks([])
			plt.tight_layout()
			plt.xlabel('Run ' + str(run))
		plt.savefig('plots/' + str(named_roi) + '_' + str(figtitle) + '.png')
		plt.clf()

# Plots two lines over runs (e.g. targ_corr and comp_corr)
def plot_two_run(input_list,compiled, measure1, measure2, yaxtitle, figtitle, yax, line_width, style):
	subplot_dict = create_subplot_dict(input_list)
	width = len(input_list) * 2.5
	plt.figure(figsize=(width,6))
	for named_roi in input_list:
		roi = roi_dict[str(named_roi)]
		measures = [measure1, measure2]
		linestyles = ['-','--']
		markers = ['.','D']
		mark_opac = [1,0.8]
		mark_size = [12,6]
		targ_comp = zip(measures,linestyles, markers, mark_opac, mark_size)
		for cond, linetype, mark, opac, m_s, in targ_comp:
			means, coarsemeans, sterr, xcoord = output_data(roi, compiled, cond)
			series=coarsemeans[cond]
			errors=sterr
			subplot_num = 100 + len(input_list*10) + (subplot_dict[named_roi])
			rr,gg,bb = color_dict[str(named_roi)] 
			plt.subplot(subplot_num)
			plt.errorbar(xcoord, series, color = (rr,gg,bb,1), lw = line_width, ls = linetype, marker = mark, ms = m_s, mfc = (rr,gg,bb,opac), mec = 'w', mew = line_width/2, yerr=errors, ecolor=(rr,gg,bb,0.6), elinewidth = line_width/2)
		plt.xticks(xcoord, ('1', '2', '3', '4'))
		plt.axhline(y=0,color='k',linewidth=0.5)
		plt.xlabel(str(named_roi))
		plt.xlim(0,5)
		plt.ylim(yax)
		if subplot_num % 10 == 1:
			plt.ylabel(yaxtitle)
			legend_targ = mlines.Line2D([], [], color='k', linestyle='-', marker='.', ms=11, label='Target')
			legend_comp = mlines.Line2D([], [], color='k', linestyle='--', marker='D', ms=5, label='Competitor')
			plt.legend(handles=[legend_targ, legend_comp], fontsize=11, numpoints=1, handlelength = 4, loc = 2)
		else:
			plt.yticks([])
		plt.tight_layout()
	plt.savefig('plots/' + str(figtitle) + '.png')
	plt.clf()




# These would make all the plots in the slides and more.


#plot_run(roi_list,compiled2, 'pearson', 'Correlation Between Items During Drawing', 'drawsim_run', (0,0.5), 1, 'white')
#plot_time(roi_list,compiled2, 'pearson', 'Correlation Between Items During Drawing', 'drawsim_time_3', (0,0.6), 3, 'error_bars')

#plot_run(in_order, compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'all_diff_run', (-.02,0.02), 1, 'white')

#plot_run(roi_list,compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'vis_diff_run', (-.02,0.02), 1, 'white')
#plot_time(roi_list,compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'vis_diff_time_3', (-0.05,0.05), 3, 'error_bars')

#plot_run(hipp_list,compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'hipp_diff_run', (-.015,0.015), 1, 'white')
#plot_time(hipp_list,compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'hipp_diff_time_3', (-0.03,0.03), 3, 'error_bars')

#plot_two_run(roi_list, compiled, 'targ_diff', 'comp_diff', 'Correlation Difference (Targ or Comp - Untrained)', 'vis_untrainbase_run', (-.03,0.03), 2, 'white')
#plot_two_time(roi_list,compiled, 'targ_diff', 'comp_diff', 'Correlation Difference (Targ or Comp - Untrained)', 'vis_untrainbase_time_5', (-.05,0.03), 5, 'error_bars')

#plot_two_run(hipp_list, compiled, 'targ_diff', 'comp_diff', 'Correlation Difference (Targ or Comp - Untrained)', 'hipp_untrainbase_run', (-.03,0.03), 2, 'white')
#plot_two_time(hipp_list, compiled, 'targ_diff', 'comp_diff', 'Correlation Difference (Targ or Comp - Untrained)', 'hipp_untrainbase_time_5', (-.03,0.03), 5, 'error_bars')

#plot_two_run(roi_list, compiled, 'targ_corr', 'comp_corr', 'Correlation with Target vs. Competitor', 'vis_targVScomp_run', (-.03,0.15), 2, 'white')
#plot_two_time(roi_list,compiled, 'targ_corr', 'comp_corr', 'Correlation with Target vs. Competitor', 'vis_targVScomp_time_5', (-.1,0.3), 5, 'error_bars')

#plot_two_run(hipp_list, compiled, 'targ_corr', 'comp_corr', 'Correlation with Target vs. Competitor', 'hipp_targVScomp_run', (-.025,0.02), 2, 'white')
#plot_two_time(hipp_list,compiled, 'targ_corr', 'comp_corr', 'Correlation with Target vs. Competitor', 'hipp_targVScomp_time_5', (-.03,0.03), 5, 'error_bars')

#plot_run(pooled,compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'pooled_run', (-.012,0.012), 1, 'white')
#plot_time(pooled,compiled, 'difference', 'Correlation Difference (Target - Competitor)', 'pooled_time', (-0.05,0.05), 3, 'error_bars')

#plot_two_run(pooled, compiled, 'targ_corr', 'comp_corr', 'Correlation with Target vs. Competitor', 'pooled_two_run', (-.02,0.11), 2, 'white')
#plot_two_time(pooled,compiled, 'targ_corr', 'comp_corr', 'Correlation with Target vs. Competitor', 'pooled_two_time', (-.1,0.3), 5, 'error_bars')



#compute_run_corr(output_list, compiled, 'difference')
