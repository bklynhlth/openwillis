###############TREMOR#################
# tremHNR.praat is a Praat[6.1.47] script (http://www.praat.org/) 
# that calculates Tremor Intensity Indices [%] within tremor.praat.
###############TREMOR#################
# Author: Markus Brückl (markus.brueckl@tu-berlin.de)
# Copyright 2011-2021 Markus Brückl
# License: GNU GPL v3 (http://www.gnu.org/licenses/gpl.html)
######################################

procedure tremHNR

hnrts = 1/minTr
periodPerWin = slength * minTr * (2/3)

#pause 'hnrts', 'minTr', 'tremMagThresh', 'periodPerWin'

hnrID = To Harmonicity (cc): hnrts, minTr, tremMagThresh, periodPerWin
hnr = Get mean: 0, 0
#pause 'hnr:2' dB
removeObject: hnrID

endproc