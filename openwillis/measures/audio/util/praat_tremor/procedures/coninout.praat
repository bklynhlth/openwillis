###############TREMOR#################
# coninout.praat is a Praat[6.1.47] script (http://www.praat.org/) 
# that serves as a procedure within tremor.praat.
###############TREMOR#################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright 2011-2021 Markus Brückl
# License: GNU GPL v3 (http://www.gnu.org/licenses/gpl.html)
######################################
#
######################################
# In: Sound object (quasi-stationary sustained phonation) in
# Out: tremor measurements as text in Info window
######################################

procedure cinout

sndID = Read from file: conPath$

slength = Get total duration

call ftrem
call atrem

writeInfoLine:
..."'name$''tab$'
...'ftrm:3''tab$'
...'ftrc:3''tab$'
...'fmodN''tab$'
...'ftrf:3''tab$'
...'ftri:3''tab$'
...'ftrp:3''tab$'
...'ftrcip:3''tab$'
...'ftrps:3''tab$'
...'ftrHNR:2''tab$'
...'atrm:3''tab$'
...'atrc:3''tab$'
...'amodN''tab$'
...'atrf:3''tab$'
...'atri:3''tab$'
...'atrp:3''tab$'
...'atrcip:3''tab$'
...'atrps:3''tab$'
...'atrHNR:2'"

endproc