###############TREMOR#################
# tremor.praat is a Praat[6.1.47] script (http://www.praat.org/)
# that extracts 18 measures of vocal tremor from a sound that
# captures a (sustained) phonation.
# Input: wav-file(s) / Praat Sound objects
# Output: text in textfiles or the Praat Info window
######################################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright: Markus Brückl (2011-2021)
# License: GNU GPL > v3 (http://www.gnu.org/licenses/gpl.html)
######################################


######################################
# Global Settings
######################################
form Tremor 3.05
   positive Program_mode 2
   word Console_and_run_mode_input_path ./sounds/
   positive Analysis_time_step_(s) 0.015
comment Arguments for initial pitch extraction
   positive Minimal_pitch_(Hz) 60
   positive Maximal_pitch_(Hz) 350
   positive Silence_threshold 0.03
   positive Voicing_threshold 0.3
   positive Octave_cost 0.01
   positive Octave-jump_cost 0.35
   positive Voiced_/_unvoiced_cost 0.14
comment Arguments for tremor extraction from contours
   positive Amplitude_extraction_method 2
   positive Minimal_tremor_frequency_(Hz) 1.5
   positive Maximal_tremor_frequency_(Hz) 15
   positive Contour_magnitude_threshold 0.01
   positive Tremor_cyclicality_threshold 0.15
   positive Frequency_tremor_octave_cost 0.01
   positive Amplitude_tremor_octave_cost 0.01
   positive Output_of_indeterminate_values 2
endform

conPath$ = console_and_run_mode_input_path$
ts = analysis_time_step; [s]

minPi = minimal_pitch; [Hz]
maxPi = maximal_pitch; [Hz]
silThresh = silence_threshold
voiThresh = voicing_threshold
ocCo = octave_cost
ocjCo = 'octave-jump_cost'
vuvCo = 'voiced_/_unvoiced_cost'

minTr = minimal_tremor_frequency; [Hz]
maxTr = maximal_tremor_frequency; [Hz]
tremthresh = tremor_cyclicality_threshold
tremMagThresh = contour_magnitude_threshold

ocFtrem = frequency_tremor_octave_cost
ocAtrem = amplitude_tremor_octave_cost

nan_out = output_of_indeterminate_values


include ./procedures/freqtrem.praat
include ./procedures/amptrem.praat
include ./procedures/readPitchOb.praat
include ./procedures/tremIntIndex.praat
include ./procedures/tremProdSum.praat
include ./procedures/coninout.praat
include ./procedures/tremHNR.praat

call cinout